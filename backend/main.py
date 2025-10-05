from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# from vertexai.generative_models import GenerativeModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
# PROJECT_ID = os.getenv("PROJECT_ID")
# LOCATION = "us-central1"

# try:
#     genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# except Exception as e:
#     print(f"Error configuring Generative AI: {e}")

# Request body type

load_dotenv()

class GenerateRequestBody(BaseModel):
    prompt: str

# --- Post endpoint ---
@app.post("/generate")
async def generate_content(body: GenerateRequestBody):
    prompt = body.prompt.strip()

    if not prompt:
        return {"error": "Prompt cannot be empty."}
    
    # tuned_model_name=os.getenv("MODEL_NAME")

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            config=types.GenerateContentConfig(
                system_instruction="You are a friendly Grade 3 peer tutor named SPARKY. Explain concepts short and simple in a way where a Grade 3 student can understand. Always start by asking their name.",
            ),
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ]

        )

        full_text = getattr(response, "text", None)
        if not full_text:
            full_text = "(no response generated)"

        return {"output": full_text}

    except Exception as e:
        print("Error Traceback:")
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/")
def root():

    return {"message": "hello world"}
