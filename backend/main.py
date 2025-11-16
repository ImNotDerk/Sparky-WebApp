from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from dotenv import load_dotenv
import os
import logging
import json
import uuid
import pathlib

from input_evaluator import InputEvaluator
from session_manager import ChatSessionManager
from chat_logic_service import ChatLogicService

# --- Setup ---
load_dotenv()
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
# logger.info(f"Using model endpoint: {MODEL_URI}")

# --- Load Stories ---

def load_all_stories(json_path: str) -> list[dict]:
    """Loads all story data from the JSON file."""
    # Build an absolute path to the JSON file
    base_dir = pathlib.Path(__file__).parent
    path = base_dir / json_path

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["stories"] # Return the list of stories

stories_data = load_all_stories("stories.json")

# --- Load Topics ---

def load_all_topics(json_path: str) -> list[dict]:
    """Loads all topic data from the JSON file."""
    # Build an absolute path to the JSON file
    base_dir = pathlib.Path(__file__).parent
    path = base_dir / json_path

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["topics"] # Return the list of topics

topics_data = load_all_topics("topics.json")

# --- Instantiate Services ---
session_manager = ChatSessionManager()
input_evaluator = InputEvaluator(stories_data=stories_data, topics_data=topics_data, genai_client=client, model_uri=MODEL_URI)
chat_logic = ChatLogicService(
    genai_client=client,
    model_uri=MODEL_URI,
    input_evaluator=input_evaluator,
    stories_data=stories_data,
    topics_data=topics_data
)

# --- Request Schemas ---
class GenerateRequestBody(BaseModel):
    session_id: str
    prompt: str

class ResetRequestBody(BaseModel):
    session_id: str

# --- Routes ---
@app.get("/")
def root():
    return {"message": "Sparky API is running!"}

@app.get("/start_chat")
def start_chat():
    """Generates a new, unique session ID for a new chat."""
    new_session_id = str(uuid.uuid4())
    session_manager.get_or_create_session(new_session_id)
    logger.info(f"New session created: {new_session_id}")
    return {"message": "New chat session created.", "session_id": new_session_id}

@app.get("/chat_history/{session_id}")
async def get_chat_history(session_id: str):
    """Gets the history for a specific session."""
    history = session_manager.get_history(session_id)
    if not history:
        return {"history": [], "total_tokens": 0}

    # Convert history to readable format
    history_data = [
        {
            "role": message.role,
            "text": " ".join(
                part.text for part in message.parts if getattr(part, "text", None)
            )
        }
        for message in history
    ]

    # Count total tokens
    try:
        total_tokens_used = await client.models.count_tokens(
            model=MODEL_URI,
            contents=history
        )
        total_tokens = getattr(total_tokens_used, "total_tokens", 0)
    except Exception:
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
        # 1. Get the user's current state from the session manager
        checklist = session_manager.get_checklist(body.session_id)
        session_data = session_manager.get_session_data(body.session_id)
        history = session_manager.get_history(body.session_id)

        # 2. Append the new user message to the history
        history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        )

        #3. Delegate ALL logic to the ChatLogicService
        bot_reply, choices_to_send, updated_checklist, updated_session_data = await chat_logic.process_message(
            checklist=checklist,
            session_data=session_data,
            history=history[:-1], # Exclude the latest user message for context
            user_prompt=prompt
        )

        # 4. Append the bot reply to the history
        history.append(types.Content(
            role="model",
            parts=[types.Part(text=bot_reply)]
        ))

        # 5. Save the updated state back to the session manager
        session_manager.save_session_data(
            session_id=body.session_id,
            checklist=updated_checklist,
            session_data=updated_session_data,
            history=history
        )

        return {"output": bot_reply, "choices": choices_to_send}

    except Exception as e:
        logger.error(f"Error sending message for session {body.session_id}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_chat")
async def reset_chat(body: ResetRequestBody):
    """Resets the chat session for a specific session ID."""
    session_manager.reset_session(body.session_id)
    logger.info(f"Session reset: {body.session_id}")
    return {"message": "Chat reset successfully!"}