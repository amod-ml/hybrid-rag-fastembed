from ..utils.conversation_manager import ConversationManager
from ..utils.openai import get_openai_client
from ..models import ChatRequest, ChatResponse
from fastapi import HTTPException

conversation_manager = ConversationManager()

async def process_chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id
    query = request.query

    # Add user message to conversation history
    conversation_manager.add_message(conversation_id, "user", query)

    # Get conversation history
    history = conversation_manager.get_history(conversation_id)

    # Prepare messages for OpenAI API
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    if history:
        messages.extend(history)
    else:
        messages.append({"role": "user", "content": query})

    try:
        # Call OpenAI API
        openai_client = await get_openai_client()
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Extract assistant's reply
        assistant_reply = response.choices[0].message.content

        # Add assistant's reply to conversation history
        conversation_manager.add_message(conversation_id, "assistant", assistant_reply)

        return ChatResponse(message=assistant_reply)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat completion: {str(e)}")

