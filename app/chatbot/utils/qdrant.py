import os
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
from .structlogger import logger

# Load environment variables
load_dotenv()

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)


def collection_exists(collection_name: str) -> bool:
    """
    Check if a collection exists in Qdrant.
    """
    try:
        collections = qdrant_client.get_collections()
        return any(
            collection.name == collection_name for collection in collections.collections
        )
    except Exception as e:
        logger.error(
            "Error checking collection existence",
            error=str(e),
            collection_name=collection_name,
        )
        raise


def create_collection(collection_name: str, vector_size: int = 1536):
    """
    Create a new collection in Qdrant.
    """
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Qdrant collection created", collection_name=collection_name)
    except Exception as e:
        logger.error(
            "Error creating Qdrant collection",
            error=str(e),
            collection_name=collection_name,
        )
        raise


def insert_chunks(
    collection_name: str, chunks: List[str], embeddings: List[List[float]]
):
    """
    Insert chunks and their embeddings into the specified collection.
    """
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(id=i, vector=embedding, payload={"text": chunk})
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ],
        )
        logger.info(
            "Chunks inserted into Qdrant",
            collection_name=collection_name,
            chunk_count=len(chunks),
        )
    except Exception as e:
        logger.error(
            "Error inserting chunks into Qdrant",
            error=str(e),
            collection_name=collection_name,
        )
        raise


def search_similar_chunks(
    collection_name: str, query_vector: List[float], limit: int = 3
):
    """
    Search for similar chunks in the specified collection.
    """
    search_result = qdrant_client.search(
        collection_name=collection_name, query_vector=query_vector, limit=limit
    )
    return search_result
