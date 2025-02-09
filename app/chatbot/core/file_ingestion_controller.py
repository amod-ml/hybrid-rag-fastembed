from fastapi import UploadFile, HTTPException
from typing import List
from ..utils.openai import get_openai_encoder
from ..utils.qdrant import qdrant_manager
from ..utils.structlogger import logger
from ..models import ChunkList, Chunk
from semantic_chunkers import StatisticalChunker
from ..utils.text_extraction import (
    extract_text_from_txt,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_excel,
    extract_text_from_csv,
)

COLLECTION_NAME = "document_collection_hybrid"


async def process_file(file: UploadFile):
    """Process file and store in Qdrant with hybrid search"""
    try:
        # Get the appropriate extractor
        extractor = get_extractor(file.filename.split(".")[-1].lower())

        # Extract text content
        content = await extract_file_content(extractor, file)

        # Create semantic chunks
        chunks = await semantic_chunking(content)

        # Ensure collection exists
        ensure_collection_exists(COLLECTION_NAME)

        # Add chunks to Qdrant with metadata
        metadata = [
            {
                "filename": file.filename,
                "chunk_index": i,
                "preview": chunk.text[:100] + "..."
                if len(chunk.text) > 100
                else chunk.text,
            }
            for i, chunk in enumerate(chunks)
        ]

        # Use simplified Qdrant manager to add chunks
        qdrant_manager.add_chunks(
            collection_name=COLLECTION_NAME, chunks=chunks, metadata=metadata
        )

        return ChunkList(chunks=chunks)

    except Exception as e:
        logger.error("Error processing file", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


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
    """Ensure Qdrant collection exists using the manager"""
    try:
        if not qdrant_manager.collection_exists(collection_name):
            qdrant_manager.create_collection(collection_name)
            logger.info(
                "Created new Qdrant collection", collection_name=collection_name
            )
        else:
            logger.info(
                "Using existing Qdrant collection", collection_name=collection_name
            )
    except Exception as e:
        logger.error("Error with Qdrant collection", error=str(e))
        raise HTTPException(status_code=500, detail="Error with Qdrant collection")


async def semantic_chunking(text: str) -> List[Chunk]:
    """Create semantic chunks from text"""
    try:
        encoder = await get_openai_encoder()
        chunker = StatisticalChunker(encoder=encoder, max_split_tokens=200)
        chunks = await chunker.acall(docs=[text])
        return [Chunk(text=" ".join(chunk.splits)) for chunk in chunks[0]]
    except Exception as e:
        logger.error("Error during semantic chunking", error=str(e))
        raise
