import os
import io
import httpx
import openai
import instructor
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import FinancialSchemas
from pypdf import PdfReader


app = FastAPI()

# Allow CORS so your frontend can call this backend on-premise and on Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TURNSTILE_SECRET = os.getenv("TURNSTILE_SECRET")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the Instructor client
client = instructor.from_openai(openai.OpenAI(api_key=OPENAI_API_KEY))

async def verify_turnstile (token: str) -> bool: 
    if not TURNSTILE_SECRET:
        raise HTTPException(
            status_code=500, detail="TURNSTILE_SECRET is missing on the server"
        )

    url = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    payload = {"secret": TURNSTILE_SECRET, "response": token}

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(url, data=payload)
        result = response.json()
        return result.get("success", False)


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),   
    turnstile_token: str = Form(...)
):
    
    isvalid = await verify_turnstile (turnstile_token)
    if not isvalid:
        raise HTTPException(
            status_code=400, detail="Invalid Cloudflare Turnstile token"
        )


    try: 
        # Read the file content as text
        content = await file.read()

        
        pdf_file = io.BytesIO(content)
        reader = PdfReader(pdf_file)


        document_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                document_text += text + "\n"

        # Call the LLM with structured output using the Pydantic model
        response = client.chat.completions.create(
            model="gpt-4o",
            response_model=FinancialSchemas,
            max_tokens=1000,
            messages=[
                {
                    "role": "user", 
                    "content": f"Extract financial metrics from this text: {document_text}"
                }
            ],
        ) 

        # Return the structured data as a JSON dictionary
        return response.model_dump()

    except Exception as e:
        
        # Return a 500 error if processing fails
        raise HTTPException(
            status_code=500, detail=f"OpenAI extraction failed: {str(e)}"
        )