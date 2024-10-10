from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from .structlogger import logger
from .controllers import (
    categorize_medical_condition,
    save_categorized_question,
    get_questions_by_category,
)
from .schemas import (
    MedicalCategory,
    QuestionInput,
    MedicalConditionCategorization,
    CategoryAndSaveResponse,
    QuestionResponse,
)
from typing import List
import uuid

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


@app.post("/categorize", response_model=MedicalConditionCategorization)
async def categorize_question(input_data: QuestionInput):
    try:
        categorization = await categorize_medical_condition(input_data.question)
        return MedicalConditionCategorization(categories=categorization.categories)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("Unexpected error in categorize_question", error=str(e))
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.post("/categorize-and-save", response_model=CategoryAndSaveResponse)
async def categorize_and_save_question(input_data: QuestionInput):
    try:
        question_uuid = str(uuid.uuid4())
        categorization = await categorize_medical_condition(input_data.question)

        question_data = {
            "uuid": question_uuid,
            "question": input_data.question,
            "categories": [category.value for category in categorization.categories],
        }

        await save_categorized_question(question_data)

        return CategoryAndSaveResponse(
            message="Question categorized and saved successfully", uuid=question_uuid
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("Unexpected error in categorize_and_save_question", error=str(e))
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.get("/questions", response_model=List[QuestionResponse])
async def get_questions_for_category(
    category: MedicalCategory = Query(
        ..., description="Medical category to filter questions"
    ),
):
    try:
        questions = await get_questions_by_category(category.value)
        return [QuestionResponse(**q) for q in questions]
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("Unexpected error in get_questions_for_category", error=str(e))
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
