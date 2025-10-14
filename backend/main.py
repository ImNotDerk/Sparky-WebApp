from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from dotenv import load_dotenv
import os
import logging

from checklist_manager import ChatChecklist
from input_evaluator import InputEvaluator

# --- Setup ---
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sparky API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Google GenAI client setup ---
PROJECT_ID = os.getenv("PROJECT_ID")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")
LOCATION = "us-central1"

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    http_options=HttpOptions(api_version="v1"),
)

MODEL_URI = f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}" # SPARKY model endpoint
logger.info(f"Using model endpoint: {MODEL_URI}")


# -- Create Chat Session ---
# This will be moved to after the topics is picked
conversation_history = []

chat = None

# chat = client.aio.chats.create(
#     model=MODEL_URI,
#     config=types.GenerateContentConfig(
#         safety_settings=[types.SafetySetting(
#             category="HARM_CATEGORY_HATE_SPEECH",
#             threshold="OFF"
#         ),types.SafetySetting(
#             category="HARM_CATEGORY_DANGEROUS_CONTENT",
#             threshold="OFF"
#         ),types.SafetySetting(
#             category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
#             threshold="OFF"
#         ),types.SafetySetting(
#             category="HARM_CATEGORY_HARASSMENT",
#             threshold="OFF"
#         )],
#     ),
#     history=conversation_history,
# )

# --- Request Schema ---
class GenerateRequestBody(BaseModel):
    prompt: str

# --- Chat State Management ---
checklist = ChatChecklist()

# --- Input Evaluator ---
input_evaluator = InputEvaluator()

# --- Routes ---
@app.get("/")
def root():
    return {"message": "Sparky API is running!"}

@app.get("/start_chat")
def start_chat():
    # chat = genai.start_chat(history=[])
    return {"message": "Chat started successfully!"}

@app.get("/chat_history")
async def get_chat_history():
    # If no history, return empty response
    if not conversation_history:
        return {"history": [], "total_tokens": 0}

    # Convert conversation_history into readable dicts
    history_data = [
        {
            "role": message.role,
            "text": " ".join(
                part.text for part in message.parts if getattr(part, "text", None)
            )
        }
        for message in conversation_history
    ]

    # Count total tokens (use conversation_history instead of chat.get_history())
    try:
        total_tokens_used = client.models.count_tokens(
            model=MODEL_URI,
            contents=chat.get_history()
        )
        total_tokens = getattr(total_tokens_used, "total_tokens", 0)
    except Exception:
        # If count_tokens fails, fallback to 0
        total_tokens = 0

    return {
        "history": history_data,
        "total_tokens": total_tokens
    }


@app.post("/send_message")
async def send_message(body: GenerateRequestBody):
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        # Append user message to conversation history
        conversation_history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        )

        next_step = checklist.next_step()

        if next_step == "got_name": # Ask for name
            if input_evaluator.is_empty_name_phrase(prompt):
                bot_reply = "Oops! I didn't catch your name. Can you say it again? 😊"

            name = input_evaluator.extract_name(prompt) # Can save this for saving of conversation history
            if name:
                checklist.data["child_name"] = name
                checklist.mark_done("got_name")
                bot_reply = f"Nice to meet you, {name}! What would you like to learn about today?"

            else:
                bot_reply = "Before we start, can you please tell me your name?"

        elif next_step == "picked_topic": # Pick a topic
            if input_evaluator.is_empty_topic_phrase(prompt):
                bot_reply = "Hmm, what topic would you like to learn about today?"

            topic = input_evaluator.extract_topic(prompt)
            if topic:
                checklist.data["topic"] = topic
                checklist.mark_done("picked_topic")
                bot_reply = f"Great choice! Let's start our adventure learning about **{topic}**. Are you ready?"

            else:
                bot_reply = "I didn't quite get that. What topic would you like to learn about today?"

        elif next_step == "story_started": # Start the story after topics is picked
            checklist.mark_done("story_started")

            # Start the chat from here with new systemInstructions with the given name and topic
            chat = client.aio.chats.create(
                model=MODEL_URI,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        f"You are a friendly Grade 3 peer tutor named SPARKY. "
                        f"You will be teaching {checklist.data['child_name']} Science concepts mainly about \"{checklist.data['topic']}\" in the Grade 3 level through interactive storytelling."
                    )
                ),
                history=conversation_history,
            )

            response = await chat.send_message(prompt)
            bot_reply = getattr(response, "text", "(no response generated)")

        else:
            # Normal chat after all steps
            response = await chat.send_message(prompt)
            bot_reply = getattr(response, "text", "(no response generated)")

        # Append bot reply to conversation history
        conversation_history.append(types.Content(
            role="model",
            parts=[types.Part(text=bot_reply)]
        ))

        return {"output": bot_reply}


    except Exception as e:
        logger.error("Error sending message", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_chat")
async def reset_chat():

    # Clear chat history in memory
    conversation_history.clear()

    # Recreate chat session
    chat = None

    # Reset checklist
    checklist.reset()

    return {"message": "Chat reset successfully!"}
