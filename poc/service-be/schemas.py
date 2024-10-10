from enum import Enum
from pydantic import BaseModel, Field
from typing import List


class MedicalCategory(str, Enum):
    CARDIOVASCULAR = "Cardiovascular"
    DERMATOLOGY = "Dermatology"
    NEUROLOGY = "Neurology"
    ONCOLOGY = "Oncology"
    PEDIATRICS = "Pediatrics"
    ENDOCRINOLOGY = "Endocrinology"
    PULMONOLOGY = "Pulmonology"
    OTHER = "Other"


class QuestionInput(BaseModel):
    question: str


class MedicalConditionCategorization(BaseModel):
    categories: List[MedicalCategory]


class CategoryAndSaveResponse(BaseModel):
    message: str
    uuid: str = Field(..., description="UUID of the saved question")


class QuestionResponse(BaseModel):
    uuid: str
    question: str
    categories: List[MedicalCategory]
