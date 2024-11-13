from ..utils.text_extraction import (
    extract_text_from_txt,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_excel,
    extract_text_from_csv,
)
from fastapi import UploadFile, HTTPException
from typing import List
from ..utils.openai import get_openai_client, get_openai_encoder
from ..utils.qdrant import create_collection, insert_chunks, collection_exists
from ..utils.structlogger import logger  # Import the logger
from ..models import ChunkList, Chunk
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from semantic_chunkers import StatisticalChunker


COLLECTION_NAME = "document_collection"


async def process_file(file: UploadFile) -> ChunkList:
    file_extension = file.filename.split(".")[-1].lower()
    logger.info(f"Processing file: {file.filename}, extension: {file_extension}")

    extractor = get_extractor(file_extension)
    content = await extract_file_content(extractor, file)
    chunks = await semantic_chunking(content)
    ensure_collection_exists(COLLECTION_NAME)
    embeddings = await generate_embeddings(chunks)
    insert_chunks_into_qdrant(COLLECTION_NAME, chunks, embeddings)

    return ChunkList(chunks=[Chunk(text=chunk, metadata={}) for chunk in chunks])


def get_extractor(file_extension: str):
    extractors = {
        "txt": extract_text_from_txt,
        "pdf": extract_text_from_pdf,
        "docx": extract_text_from_docx,
        "doc": extract_text_from_docx,
        "xlsx": extract_text_from_excel,
        "xls": extract_text_from_excel,
        "csv": extract_text_from_csv,
    }
    extractor = extractors.get(file_extension)
    if not extractor:
        logger.error("Unsupported file format", extension=file_extension)
        raise HTTPException(
            status_code=402, detail=f"Unsupported file format: {file_extension}"
        )
    return extractor


async def extract_file_content(extractor, file: UploadFile) -> str:
    try:
        content = await extractor(file)
        logger.info("File content extracted successfully", filename=file.filename)
        return content
    except Exception as e:
        logger.error(
            "Error extracting file content", error=str(e), filename=file.filename
        )
        raise HTTPException(
            status_code=500, detail=f"Error extracting content from {file.filename}"
        )


def ensure_collection_exists(collection_name: str):
    try:
        if not collection_exists(collection_name):
            create_collection(collection_name)
            logger.info(f"Qdrant collection created: {collection_name}")
        else:
            logger.info(
                "Qdrant collection already exists", collection_name=collection_name
            )
    except Exception as e:
        logger.error(
            "Error checking/creating Qdrant collection",
            error=str(e),
            collection_name=collection_name,
        )
        raise HTTPException(status_code=500, detail="Error with Qdrant collection")


async def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    try:
        embeddings = await get_embeddings(chunks)
        logger.info("Embeddings generated", embedding_count=len(embeddings))
        return embeddings
    except Exception as e:
        logger.error("Error generating embeddings", error=str(e))
        raise HTTPException(status_code=500, detail="Error generating embeddings")


def insert_chunks_into_qdrant(
    collection_name: str, chunks: List[str], embeddings: List[List[float]]
):
    try:
        insert_chunks(collection_name, chunks, embeddings)
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
        raise HTTPException(
            status_code=500, detail="Error inserting chunks into Qdrant"
        )


def create_basic_chunks(text: str) -> List[str]:
    """
    Fallback chunking method using RecursiveCharacterTextSplitter
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.split_text(text)

        # Format chunks with basic metadata
        formatted_chunks = [
            f"{chunk}\n\nMetadata:\nSummary: Auto-generated chunk\nTags: automatic_split\nDocument Metadata: basic_split"
            for chunk in chunks
        ]

        logger.info(f"Created {len(formatted_chunks)} chunks using basic text splitter")
        return formatted_chunks

    except Exception as e:
        logger.error(f"Error in basic chunking: {str(e)}")
        raise


async def get_embeddings(chunks: List[str]) -> List[List[float]]:
    client = await get_openai_client()
    logger.info("Starting embedding generation", chunk_count=len(chunks))

    embeddings = []
    try:
        for chunk in chunks:
            response = await client.embeddings.create(
                model="text-embedding-3-small", input=chunk
            )
            embeddings.append(response.data[0].embedding)
        logger.info("Embedding generation completed", embedding_count=len(embeddings))
        return embeddings
    except Exception as e:
        logger.error("Error generating embeddings", error=str(e))
        raise


async def semantic_chunking(text: str) -> List[str]:
    try:
        encoder = await get_openai_encoder()
        chunker = StatisticalChunker(encoder=encoder, max_split_tokens=300)
        chunks = await chunker.acall(docs=[text])
        paragraphs = [" ".join(chunk.splits) for chunk in chunks[0]]
        return paragraphs
    except Exception as e:
        logger.error("Error during semantic chunking", error=str(e))
        raise
