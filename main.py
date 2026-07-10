import base64
import os
import json
from typing import Optional, List, Type
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model

# Modern Google GenAI SDK
from google import genai
from google.genai import types

app = FastAPI(title="IITM Combined Grading Cell API")

# 1. Enable CORS for all Cloudflare Worker graders
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Setup Gemini Client 
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
        "CRITICAL RULE: If the answer is a numeric value, "
        "you MUST return ONLY the raw number. Do NOT include currency symbols, "
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
    invoice_no: Optional[str] = Field(None)
    date: Optional[str] = Field(None)
    vendor: Optional[str] = Field(None)
    amount: Optional[float] = Field(None)
    tax: Optional[float] = Field(None)
    currency: Optional[str] = Field(None)

@app.post("/extract", response_model=InvoiceResponse)
async def extract_invoice(payload: InvoiceRequest):
    try:
        extract_instruction = (
            "Analyze the following invoice text and extract the required fields precisely. "
            "Convert dates to ISO YYYY-MM-DD. 'amount' is subtotal before tax. "
            "If a field cannot be found, use JSON null."
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
        
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            raw_text = "\n".join(lines[1:-1])
        
        return InvoiceResponse.model_validate_json(raw_text.strip())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


# =====================================================================
# TASK 3: Dynamic Schema Extraction (DataBridge Inc.)
# =====================================================================

def create_dynamic_model(schema_dict: dict) -> Type[BaseModel]:
    """Dynamically builds a Pydantic model based on the requested JSON schema types."""
    type_mapping = {
        "string": str,
        "integer": int,
        "float": float,
        "boolean": bool,
        "date": str,
        "array[string]": List[str],
        "array[integer]": List[int]
    }
    
    fields = {}
    for key, field_type in schema_dict.items():
        # Map the requested string type to an actual Python type, defaulting to str
        py_type = type_mapping.get(field_type, str)
        
        # Add special formatting hints into the Pydantic description
        desc = "Return JSON null if the value is missing."
        if field_type == "date":
            desc = "ISO format YYYY-MM-DD. Return JSON null if missing."
            
        # All fields are optional (default=None) so Pydantic safely handles missing data by outputting null
        fields[key] = (Optional[py_type], Field(default=None, description=desc))
        
    return create_model('DynamicSchemaModel', **fields)


@app.post("/dynamic-extract")
async def dynamic_extract(payload: dict):
    try:
        text = payload.get("text", "")
        schema_dict = payload.get("schema", {})
        
        # 1. Build the dynamic schema class
        DynamicModel = create_dynamic_model(schema_dict)
        
        # 2. Instruct the model
        instruction = (
            "You are a dynamic ETL parsing agent. Extract data from the provided text to exactly match the requested schema. "
            "Rules:\n"
            "1. Return ONLY the keys requested in the schema.\n"
            "2. If a value is missing, return JSON null (not the string 'null').\n"
            "3. Dates must be formatted as YYYY-MM-DD.\n"
            f"\nText to parse:\n{text}"
        )
        
        # 3. Request structured output matching the dynamic Pydantic class
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=instruction,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DynamicModel,
                temperature=0.0,
            ),
        )
        
        # 4. Clean up any markdown blocks
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            raw_text = "\n".join(lines[1:-1]).strip()

        # 5. Validate through Pydantic and dump to a dictionary.
        # This guarantees exactly the keys requested are present, and missing ones are natively `null`.
        validated_data = DynamicModel.model_validate_json(raw_text)
        return validated_data.model_dump()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dynamic extraction failed: {str(e)}")

from typing import Dict, List, Any

# =====================================================================
# TASK 4: Korean Audio Dataset API
# =====================================================================

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

# This precisely matches the 13 keys from the Japanese/Korean prompt debugger
class AudioStatsResponse(BaseModel):
    rows: int
    columns: List[str]
    mean: Dict[str, Any]
    std: Dict[str, Any]
    variance: Dict[str, Any]
    min: Dict[str, Any]
    max: Dict[str, Any]
    median: Dict[str, Any]
    mode: Dict[str, Any]
    range: Dict[str, Any]
    allowed_values: Dict[str, Any]
    value_range: Dict[str, Any]
    correlation: List[Any]

@app.post("/analyze-audio", response_model=AudioStatsResponse)
async def analyze_audio(payload: AudioRequest):
    try:
        # 1. Clean and decode the base64 audio
        b64_string = payload.audio_base64
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
            
        audio_bytes = base64.b64decode(b64_string)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio string")

    # 2. Instruct Gemini to listen to the Korean audio and map it to our stats schema
    audio_instruction = (
        "You are an expert multilingual data analyst. "
        "Listen to the provided Korean audio file, which describes a dataset and its statistical properties. "
        "Extract the statistical values mentioned in the audio and map them strictly to the requested JSON schema. "
        "Translate the concepts accurately to fill in fields like mean, std (standard deviation), variance, etc. "
        "If a specific statistic is not mentioned in the audio, return an empty dictionary {} or array [] as appropriate for that field."
    )

    try:
        # 3. Use Gemini's multimodal audio capabilities
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                # We use audio/mp3 as a safe default; Gemini natively processes the binary
                types.Part.from_bytes(data=audio_bytes, mime_type='audio/mp3'), 
                audio_instruction
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AudioStatsResponse,
                temperature=0.0, # Zero temperature to prevent hallucinated math
            )
        )
        
        # 4. Clean up any markdown codeblock backticks if present
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            raw_text = "\n".join(lines[1:-1]).strip()

        # Parse and return the structured data
        parsed_json = json.loads(raw_text)
        return AudioStatsResponse.model_validate(parsed_json)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")