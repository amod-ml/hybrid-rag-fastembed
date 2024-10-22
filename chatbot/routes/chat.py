from fastapi import APIRouter, Depends, HTTPException
from ..utils.openai import get_openai_client
from ..models import ChatRequest, ChatResponse
from ..core.chat_controller import process_chat

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, openai_client=Depends(get_openai_client)):
    """
    Handles incoming chat requests by processing the query and returning the response.

    Parameters:
    -----------
    request : ChatRequest
        The incoming chat request containing the query and conversation ID.

    openai_client:
        Dependency to get the OpenAI client for sending the query.

    Returns:
    --------
    ChatResponse:
        The assistant's reply based on the user's query and conversation history.
    """
    try:
        return await process_chat(request)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
