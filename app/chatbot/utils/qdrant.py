import os
from typing import List
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from tqdm import tqdm
from .structlogger import logger
from ..models import Chunk

load_dotenv()


class QdrantManager:
    DENSE_MODEL = "snowflake/snowflake-arctic-embed-xs"
    SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"

    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        # Set up models
        self.client.set_model(self.DENSE_MODEL)
        self.client.set_sparse_model(self.SPARSE_MODEL)

    def collection_exists(self, collection_name: str) -> bool:
        try:
            collections = self.client.get_collections()
            return any(
                collection.name == collection_name
                for collection in collections.collections
            )
        except Exception as e:
            logger.error("Error checking collection existence", error=str(e))
            raise

    def create_collection(self, collection_name: str):
        """Create a new collection with hybrid search support"""
        try:
            if not self.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=self.client.get_fastembed_vector_params(),
                    sparse_vectors_config=self.client.get_fastembed_sparse_vector_params(),
                )
                logger.info(
                    "Created collection with hybrid vectors",
                    collection_name=collection_name,
                )
        except Exception as e:
            logger.error("Error creating collection", error=str(e))
            raise

    def add_chunks(
        self, collection_name: str, chunks: List[Chunk], metadata: List[dict] = None
    ):
        """Add chunks with automatic vector generation"""
        try:
            # Prepare documents and metadata
            documents = [chunk.text for chunk in chunks]
            if metadata is None:
                metadata = [{"text": chunk.text} for chunk in chunks]

            # Add documents with progress bar
            self.client.add(
                collection_name=collection_name,
                documents=documents,
                metadata=metadata,
                ids=tqdm(range(len(documents))),
                parallel=0,  # Use all available CPU cores
            )
            logger.info(
                "Added chunks to collection",
                collection_name=collection_name,
                chunk_count=len(chunks),
            )
        except Exception as e:
            logger.error("Error adding chunks", error=str(e))
            raise

    def search(self, collection_name: str, query_text: str, limit: int = 5):
        """Simple search implementation following the tutorial"""
        try:
            logger.info(
                "Performing search",
                collection_name=collection_name,
                query_text=query_text
            )
            
            search_result = self.client.query(
                collection_name=collection_name,
                query_text=query_text,
                query_filter=None,
                limit=limit,
            )
            
            # Simply return the metadata list as shown in tutorial
            return [hit.metadata for hit in search_result]

        except Exception as e:
            logger.error("Error performing search", error=str(e))
            raise

    def verify_collection(self, collection_name: str):
        """Verify collection exists and has documents"""
        try:
            # Check if collection exists
            if not self.collection_exists(collection_name):
                logger.error(
                    "Collection does not exist",
                    collection_name=collection_name
                )
                return False
            
            # Get collection info
            collection_info = self.client.get_collection(collection_name)
            points_count = collection_info.points_count
            
            logger.info(
                "Collection verification",
                collection_name=collection_name,
                points_count=points_count,
                vectors_config=collection_info.config.params.vectors,
                sparse_vectors_config=collection_info.config.params.sparse_vectors
            )
            
            return points_count > 0
            
        except Exception as e:
            logger.error(
                "Error verifying collection",
                error=str(e),
                collection_name=collection_name
            )
            return False


# Initialize global instance
qdrant_manager = QdrantManager()
