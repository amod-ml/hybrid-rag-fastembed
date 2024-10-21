from fastapi import APIRouter, Depends, HTTPException
from ..utils.openai import get_openai_client
from ..models import ChatRequest, ChatResponse
from ..core.chat_controller import process_chat

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, openai_client=Depends(get_openai_client)):
    try:
        return await process_chat(request)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
