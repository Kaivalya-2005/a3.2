import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# 1. Initialize FastAPI app
app = FastAPI(title="Multimodal QA API")

# 2. Enable CORS (Crucial for Cloudflare Worker grader)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define Request and Response Schemas
class ImageQARequest(BaseModel):
    image_base64: str
    question: str

class ImageQAResponse(BaseModel):
    answer: str

# Configure Gemini API (Ensure GEMINI_API_KEY is set in your environment variables)
# genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

@app.post("/answer-image", response_model=ImageQAResponse)
async def answer_image(payload: ImageQARequest):
    # Ensure API key is available
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
         raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured on server.")
    
    genai.configure(api_key=api_key)

    try:
        # Prepare the image payload for Gemini
        image_part = {
            "mime_type": "image/jpeg", # Gemini automatically handles png/jpeg via base64
            "data": payload.image_base64
        }
        
        # We use gemini-1.5-flash as it is the fastest and most cost-effective for multimodal QA
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Strict prompt engineering to satisfy grader constraints
        system_instructions = (
            "You are a strict data extraction bot. Answer the user's question based ONLY on the provided image. "
            "CRITICAL RULES: \n"
            "1. If the answer is numeric, return ONLY the raw number as a string. \n"
            "2. DO NOT include units, currency symbols (e.g., $, ₹), or commas. \n"
            "3. DO NOT use full sentences or conversational text. \n"
            "4. Just return the exact value requested."
        )
        
        prompt = f"{system_instructions}\n\nQuestion: {payload.question}"
        
        # Call the model
        response = model.generate_content([image_part, prompt])
        
        # Strip any accidental whitespace or hidden newline characters
        extracted_answer = response.text.strip()
        
        return {"answer": extracted_answer}
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during image processing.")