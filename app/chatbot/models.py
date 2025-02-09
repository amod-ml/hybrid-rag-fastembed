from pydantic import BaseModel
from typing import List

class FileUploadResponse(BaseModel):
    filename: str
    message: str


class Chunk(BaseModel):
    text: str


class ChunkList(BaseModel):
    chunks: List[Chunk]


class ChatRequest(BaseModel):
    conversation_id: str
    query: str


class ChatResponse(BaseModel):
    message: str
