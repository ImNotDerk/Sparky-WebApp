from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from dotenv import load_dotenv
import os
import logging

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

MODEL_URI = f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"
logger.info(f"Using model endpoint: {MODEL_URI}")

# --- Request Schema ---
class GenerateRequestBody(BaseModel):
    prompt: str

# --- Routes ---
@app.get("/")
def root():
    return {"message": "Sparky API is running!"}

@app.post("/generate")
async def generate_content(body: GenerateRequestBody):
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        response = client.models.generate_content(
            model=MODEL_URI,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a friendly Grade 3 peer tutor named SPARKY. "
                    "Explain science concepts about Living Things short and simple so a Grade 3 student can understand. "
                    "Always start by asking their name."
                ),
            ),
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )

        full_text = getattr(response, "text", "(no response generated)")
        return {"output": full_text}

    except Exception as e:
        logger.error("Error generating content", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
