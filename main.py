import base64
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="IITM Multimodal QA API")

# 1. Enable CORS (Required by the Cloudflare Worker Grader)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Setup Gemini Client (Requires GEMINI_API_KEY environment variable)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 3. Define the exact schemas expected by the grader
class QARequest(BaseModel):
    image_base64: str
    question: str

class QAResponse(BaseModel):
    answer: str

# 4. The Endpoint exactly as specified
@app.post("/answer-image", response_model=QAResponse)
async def answer_image(payload: QARequest):
    try:
        # Strip any Base64 headers (e.g., "data:image/png;base64,") if the grader includes them
        b64_string = payload.image_base64
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
            
        image_bytes = base64.b64decode(b64_string)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 string")

    # Strict instructions to enforce the grader's formatting rules
    system_instruction = (
        "You are a strict data extraction assistant. "
        "Answer the user's question based on the provided image. "
        "CRITICAL RULE: If the answer is a numeric value (like a total, price, or score), "
        "you MUST return ONLY the raw number. Do NOT include currency symbols (like $ or ₹), "
        "units, or commas. Return just the digits and decimals (e.g., '4089.35')."
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
                payload.question
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0, # Keep temperature 0 for factual extraction
            )
        )
        
        # Clean up the output string to ensure it's just the answer
        clean_answer = response.text.strip().replace("`", "").replace('"', '')
        return QAResponse(answer=clean_answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))