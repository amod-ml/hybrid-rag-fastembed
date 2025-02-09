from ..utils.conversation_manager import ConversationManager
from ..utils.openai import get_openai_client
from ..utils.qdrant import qdrant_manager
from ..utils.structlogger import logger
from ..models import ChatRequest, ChatResponse
from fastapi import HTTPException
import json

conversation_manager = ConversationManager()

async def get_search_query_prompt(query: str, history_text: str) -> str:
    return f"""
    Based on the current query and conversation history, create a detailed comprehensive search query (these queries are in the context of a university) for retrieving relevant information.

    Current Query: {query}
    Conversation History:
    {history_text}

    Please respond with a JSON object containing a "search_query" field.
    """

async def process_chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id
    query = request.query
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "Search_Query",
            "schema": {
                "type": "object",
                "properties": {"search_query": {"type": "string"}},
                "required": ["search_query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    # Create or get conversation
    if conversation_id not in conversation_manager.conversations:
        conversation_manager.create_conversation(conversation_id)
    
    # Get conversation history
    qa_pairs = conversation_manager.get_qa_pairs(conversation_id)
    history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs])

    try:
        client = await get_openai_client()

        # Get optimized search query
        search_query_result = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": await get_search_query_prompt(query, history_text)}
            ],
            response_format=response_format
        )
        
        # Correctly parse the JSON response
        search_query_json = json.loads(search_query_result.choices[0].message.content)
        search_query = search_query_json["search_query"]

        # Verify collection before searching
        collection_name = "document_collection_hybrid"
        if not qdrant_manager.verify_collection(collection_name):
            logger.error("Invalid or empty collection")
            contexts = ["The knowledge base appears to be empty or unavailable."]
        else:
            # Perform hybrid search using QdrantManager
            search_results = qdrant_manager.search(
                collection_name=collection_name,
                query_text=search_query,
                limit=10
            )
            #logger.info(f"Search results: {search_results}")
            
            # Extract text from search results
            contexts = []
            for result in search_results:
                if result and 'document' in result:
                    contexts.append(result['document'])
            
            if not contexts:
                logger.warning("No relevant contexts found in search results")
                contexts = ["No relevant information found in the knowledge base."]
            
            # Generate response using context
            system_message = f"""
            You are a knowledgeable assistant that must follow a structured approach to respond to user queries based on the provided context and conversation history.

            Steps:
            
            1. **Determine Relevance**: Assess whether the provided context is directly relevant to the user’s query and the conversation history.
                - If the context is relevant to the query, proceed to generate the answer using the context.
                - If the context is irrelevant, ignore it and generate the answer solely based on your internal knowledge.
            
            2. **Generate Answer**:
                - If using the context, generate a response that accurately incorporates the information from the context to answer the query.
                - If the context is irrelevant, use only your internal knowledge to answer the question. Do not reference the context in this case.
            
            3. **Add Legal Disclaimer**:
                - If the answer is based on the provided context, end with: "This answer was provided by referring to a specialized knowledgebase."
                - If the answer is generated solely from internal knowledge, end with: "This answer was generated based on the internal knowledge of OpenAI's language model."
            
            Provided Context:
            {' '.join(contexts)}

            Conversation History:
            {history_text}

            Respond to the following query in a clear and accurate manner, following the above steps.
            """

            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
                ],
            )

            assistant_reply = response.choices[0].message.content
            
            # Store conversation
            conversation_manager.add_message_pair(conversation_id, query, assistant_reply)
            
            return ChatResponse(message=assistant_reply)

    except Exception as e:
        logger.error(f"Error in process_chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")
