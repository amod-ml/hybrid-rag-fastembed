from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .structlogger import logger
from .controllers import categorize_medical_condition
from pydantic import BaseModel

app = FastAPI()

logger.warning("This is DEV Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/status")
async def status():
    return {"status": "ok"}

class QuestionInput(BaseModel):
    question: str

@app.post("/categorize")
async def categorize_question(input_data: QuestionInput):
    try:
        categorization = await categorize_medical_condition(input_data.question)
        return categorization
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("Unexpected error in categorize_question", error=str(e))
        raise HTTPException(status_code=500, detail="An unexpected error occurred")




