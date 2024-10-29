import json
from typing import List, Dict
from ..utils.conversation_manager import ConversationManager
from ..utils.openai import get_openai_client
from ..utils.qdrant import search_similar_chunks
from ..utils.structlogger import logger
from ..models import ChatRequest, ChatResponse
from fastapi import HTTPException

conversation_manager = ConversationManager()


async def get_classification_prompt(query: str, history_text: str) -> str:
    return f"""
    You are an AI assistant for a medical question-answer system. Your task is to determine if the current query requires searching a specialized medical document database, considering the conversation history.

    Current Query: {query}
    Conversation History:
    {history_text}

    Based on this context, determine whether retrieval of additional information from the medical database is necessary to answer the query accurately.
    Please respond with a JSON object containing two fields:
    1. "search_required": a boolean indicating whether a search is needed
    2. "search_query": if search is required, provide a relevant search query based on the conversation history and current query; otherwise, set to null

    Ensure your response is in valid JSON format.
    """


async def classify_query(client, prompt: str) -> Dict:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "Query_Classification_Result",
            "schema": {
                "type": "object",
                "properties": {
                    "search_required": {"type": "boolean"},
                    "search_query": {"type": ["string", "null"]},
                },
                "required": ["search_required", "search_query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            response_format=response_format,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in query classification: {str(e)}"
        )


async def generate_response(client, messages: List[Dict[str, str]]) -> str:
    try:
        response = await client.chat.completions.create(
            model="gpt-4o", messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in response generation: {str(e)}"
        )


async def get_embeddings(client, text: str) -> List[float]:
    try:
        response = await client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating embeddings: {str(e)}"
        )


def mmr(
    query_embedding: List[float],
    embeddings: List[List[float]],
    texts: List[str],
    k: int = 3,
    lambda_param: float = 0.5,
) -> List[str]:
    """Maximal Marginal Relevance algorithm for diverse retrieval"""
    selected = []
    remaining_ids = list(range(len(embeddings)))

    for _ in range(k):
        if not remaining_ids:
            break

        # Compute relevance scores
        relevance_scores = [
            sum(a * b for a, b in zip(query_embedding, embeddings[idx]))
            for idx in remaining_ids
        ]

        if not selected:
            # Select the most relevant document for the first iteration
            best_idx = max(
                range(len(relevance_scores)), key=lambda i: relevance_scores[i]
            )
        else:
            # Compute diversity scores
            diversity_scores = [
                max(
                    sum(
                        (a - b) ** 2
                        for a, b in zip(embeddings[idx], embeddings[sel_idx])
                    )
                    for sel_idx in selected
                )
                for idx in remaining_ids
            ]

            # Compute MMR scores
            mmr_scores = [
                lambda_param * relevance_scores[i]
                - (1 - lambda_param) * diversity_scores[i]
                for i in range(len(remaining_ids))
            ]
            best_idx = max(range(len(mmr_scores)), key=lambda i: mmr_scores[i])

        selected.append(remaining_ids[best_idx])
        remaining_ids.pop(best_idx)

    return [texts[idx] for idx in selected]


async def process_chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id
    query = request.query

    # Create a new conversation if it doesn't exist
    if conversation_id not in conversation_manager.conversations:
        conversation_manager.create_conversation(conversation_id)

    # Get conversation history (question-answer pairs)
    qa_pairs = conversation_manager.get_qa_pairs(conversation_id)
    history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs])

    client = await get_openai_client()
    assistant_reply = ""  # Initialize reply variable

    try:
        classification_prompt = await get_classification_prompt(query, history_text)
        classification_result = await classify_query(client, classification_prompt)

        if classification_result["search_required"]:
            # Perform similarity search using Qdrant
            query_embedding = await get_embeddings(
                client, classification_result["search_query"]
            )
            search_results = search_similar_chunks(
                "medical_document_repository", query_embedding, limit=3
            )

            # Extract texts from search results
            chunk_texts = [result.payload["text"] for result in search_results]
            search_query = classification_result["search_query"]

            rag_is_required = await determine_rag_response(search_query, chunk_texts)
            if rag_is_required["result"]:
                assistant_reply = await get_rag_response(
                    client, query, history_text, chunk_texts
                )
            else:
                # Use general knowledge response if RAG is not required
                system_message = f"""You are a helpful medical assistant. 
                Please answer the user's query based on your general knowledge.
                Conversation history:
                {history_text}
                """
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"User query: {query}"},
                ]
                assistant_reply = await generate_response(client, messages)
        else:
            # No search required, use general knowledge
            system_message = f"""You are a helpful medical assistant. 
            Please answer the user's query based on your general knowledge.
            Conversation history:
            {history_text}
            """
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"User query: {query}"},
            ]
            assistant_reply = await generate_response(client, messages)

        # Store the conversation
        conversation_manager.add_message_pair(conversation_id, query, assistant_reply)

        # Always return a ChatResponse object
        return ChatResponse(message=assistant_reply)

    except Exception as e:
        logger.error(f"Error in process_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


async def determine_rag_response(
    search_query: str, chunk_texts: List[str]
) -> Dict[str, bool]:
    client = await get_openai_client()

    system_message = f"""You are an AI assistant that determines if the provided texts has context to accurately answer the user's medical question.

    User query: {search_query}

    Relevant information:
    {' '.join(chunk_texts)}

    If yes you will return "result": "true", if no you will return "result": "false" in JSON format.
    """

    messages = [{"role": "system", "content": system_message}]

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "RAG_Response",
            "schema": {
                "type": "object",
                "properties": {"result": {"type": "boolean"}},
                "required": ["result"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
        response_format=response_format,
    )
    logger.info(
        "RAG determination",
        result=json.loads(response.choices[0].message.content)["result"],
    )
    return json.loads(response.choices[0].message.content)


async def get_rag_response(
    client, query: str, history_text: str, chunk_texts: List[str]
) -> str:
    system_message = f"""You are a helpful medical assistant. Please answer the user's query based on the following relevant information and your general knowledge.
        
        Conversation history:
        {history_text}
        
        Relevant information:
        {' '.join(chunk_texts)}

    Use this information to provide an accurate and helpful response."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"User query: {query}"},
    ]
    response = await generate_response(client, messages)
    return response
