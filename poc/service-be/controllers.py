from .services import get_openai_client
from .structlogger import logger
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError
from typing import List


class MedicalConditionCategorization(BaseModel):
    categories: List[str]


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
    This should be done as a last resort.

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
                                    "enum": [
                                        "Cardiovascular",
                                        "Dermatology",
                                        "Neurology",
                                        "Oncology",
                                        "Pediatrics",
                                        "Endocrinology",
                                        "Pulmonology",
                                        "Other",
                                    ],
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
