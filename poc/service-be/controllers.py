from .services import get_openai_client, get_mongodb_client
from .structlogger import logger
from fastapi import HTTPException
from pydantic import ValidationError
from typing import List, Dict
from .schemas import MedicalCategory, MedicalConditionCategorization



async def categorize_medical_condition(question: str) -> MedicalConditionCategorization:
    client = await get_openai_client()
    system_prompt = """
    You are a medical expert tasked with categorizing medical questions. 
    Categorize the given question into one or more of the following categories:
    - Cardiovascular
    - Dermatology
    - Neurology
    - Oncology
    - Pediatrics
    - Endocrinology
    - Pulmonology
    - Other

    A question can belong to one or multiple categories. If the categorization is ambiguous 
    and cannot be put into any of the specific categories, categorize it as 'Other'. 
    This should be done as a last resort. The category 'Other' strictly cannot be accompanied by any other categories.

    Respond with a JSON object containing a 'categories' key with an array of category names.
    """
    try:
        logger.info("Categorizing medical condition", question=question)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}"},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "Medical_Condition_Categorization",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "categories": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [cat.value for cat in MedicalCategory],
                                },
                            }
                        },
                        "required": ["categories"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        )
        try:
            logger.debug("Categorization response", response=response)
            categorization = MedicalConditionCategorization.parse_raw(
                response.choices[0].message.content
            )
            logger.info("Categorization successful", categorization=categorization)
            return categorization
        except ValidationError as e:
            logger.error("Invalid response format", error=e)
            raise HTTPException(
                status_code=400, detail=f"Invalid response format: {str(e)}"
            )
    except Exception as e:
        logger.error("Error categorizing medical condition", error=e)
        raise HTTPException(
            status_code=500, detail=f"Error categorizing medical condition: {str(e)}"
        )


async def save_categorized_question(question_data: Dict):
    try:
        client = await get_mongodb_client()
        db = client.get_database("medical_questions")
        collection = db.get_collection("categorized_questions")
        
        result = await collection.insert_one(question_data)
        
        if result.inserted_id:
            logger.info("Question saved successfully", uuid=question_data["uuid"])
            return True
        else:
            logger.error("Failed to save question", uuid=question_data["uuid"])
            raise HTTPException(status_code=500, detail="Failed to save question")
    except Exception as e:
        logger.error("Error saving categorized question", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error saving categorized question: {str(e)}")


async def get_questions_by_category(category: str) -> List[Dict]:
    try:
        client = await get_mongodb_client()
        db = client.get_database("medical_questions")
        collection = db.get_collection("categorized_questions")
        
        query = {"categories": category}
        cursor = collection.find(query)
        
        questions = await cursor.to_list(length=None)
        
        logger.info(f"Retrieved {len(questions)} questions for category: {category}")
        return questions
    except Exception as e:
        logger.error(f"Error retrieving questions for category {category}", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error retrieving questions: {str(e)}")
