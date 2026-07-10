import base64
import os
import json
import asyncio
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
# =====================================================================
# TASK 5: Invoice Intelligence (Dynamic JSON Schema)
# =====================================================================

class InvoiceIntelRequest(BaseModel):
    document_id: str
    text: str
    schema: dict  # The grader sends the exact JSON schema definition here

@app.post("/invoice-intelligence")
async def invoice_intelligence(payload: InvoiceIntelRequest):
    # 1. Prepare the instruction, injecting the exact schema into the prompt
    instruction = (
        "You are an expert invoice extraction AI. "
        "Extract the information from the provided invoice text according to the exact JSON schema provided below. "
        "CRITICAL RULES:\n"
        "1. Strictly return valid JSON matching this schema.\n"
        "2. Do not include any extra keys.\n"
        "3. Ensure data types match (e.g., numbers are integers/floats, not strings, if specified).\n\n"
        f"=== REQUIRED JSON SCHEMA ===\n{json.dumps(payload.schema, indent=2)}\n\n"
        f"=== INVOICE TEXT ===\n{payload.text}"
    )

    # 2. Retry loop to protect against 429 RESOURCE_EXHAUSTED errors
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=instruction,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0, # Zero temperature for factual data extraction
                )
            )
            
            # Clean up potential markdown formatting
            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```json"):
                    raw_text = "\n".join(lines[1:-1]).strip()
                else:
                    raw_text = "\n".join(lines[1:-1]).strip()

            # Return the generated JSON (FastAPI handles serialization automatically)
            return json.loads(raw_text)
            
        except Exception as e:
            error_msg = str(e)
            # Catch rate limits and wait before retrying
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Wait 1s, 2s, 4s, etc.
                    continue
            
            # If it's a structural error or we run out of retries, throw the 500
            raise HTTPException(status_code=500, detail=f"Extraction failed: {error_msg}")