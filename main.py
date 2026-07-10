import base64
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Modern Google GenAI SDK
from google import genai
from google.genai import types

app = FastAPI(title="IITM Combined Grading Cell API")

# 1. Enable CORS for both Cloudflare Worker graders
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Setup Gemini Client (Automatically reads GEMINI_API_KEY from environment)
client = genai.Client()

# =====================================================================
# TASK 1: Multimodal Image Question-Answering
# =====================================================================

class QARequest(BaseModel):
    image_base64: str
    question: str

class QAResponse(BaseModel):
    answer: str

@app.post("/answer-image", response_model=QAResponse)
async def answer_image(payload: QARequest):
    try:
        b64_string = payload.image_base64
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        image_bytes = base64.b64decode(b64_string)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 string")

    qa_instruction = (
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
                system_instruction=qa_instruction,
                temperature=0.0,
            )
        )
        clean_answer = response.text.strip().replace("`", "").replace('"', '')
        return QAResponse(answer=clean_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# TASK 2: Fixed Schema Invoice Extraction
# =====================================================================

class InvoiceRequest(BaseModel):
    invoice_text: str

class InvoiceResponse(BaseModel):
    invoice_no: Optional[str] = Field(None, description="The invoice number. Null if not found.")
    date: Optional[str] = Field(None, description="The date strictly in YYYY-MM-DD format. Null if not found.")
    vendor: Optional[str] = Field(None, description="The name of the vendor or issuing company. Null if not found.")
    amount: Optional[float] = Field(None, description="The subtotal amount BEFORE tax. Null if not found.")
    tax: Optional[float] = Field(None, description="The tax amount only. Null if not found.")
    currency: Optional[str] = Field(None, description="The currency (e.g., INR, USD). Null if not found.")

@app.post("/extract", response_model=InvoiceResponse)
async def extract_invoice(payload: InvoiceRequest):
    try:
        extract_instruction = (
            "Analyze the following invoice text and extract the required fields. "
            "Ensure the date is converted to YYYY-MM-DD format. "
            "Ensure 'amount' is strictly the subtotal before tax, and 'tax' is just the tax amount."
            f"\n\nInvoice Text:\n{payload.invoice_text}"
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=extract_instruction,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=InvoiceResponse,
                temperature=0.0,
            ),
        )
        return InvoiceResponse.model_validate_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))