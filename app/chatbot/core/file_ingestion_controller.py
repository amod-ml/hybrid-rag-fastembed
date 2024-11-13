import tiktoken
import json
import openai
import textwrap
from ..utils.text_extraction import (
    extract_text_from_txt,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_excel,
    extract_text_from_csv,
)
from fastapi import UploadFile, HTTPException
from typing import List
from ..utils.openai import get_openai_client
from ..utils.qdrant import create_collection, insert_chunks, collection_exists
from ..utils.structlogger import logger  # Import the logger
from ..models import ChunkList, Chunk
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
)
from json.decoder import JSONDecodeError
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


COLLECTION_NAME = "medical_document_collection"


async def process_file(file: UploadFile) -> ChunkList:
    file_extension = file.filename.split(".")[-1].lower()
    logger.info(f"Processing file: {file.filename}, extension: {file_extension}")

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

    try:
        content = await extractor(file)
        logger.info("File content extracted successfully", filename=file.filename)
        token_count = num_tokens_from_string(content, "o200k_base")
        logger.info(f"Token count calculated: {token_count}")

        if token_count <= 128000:
            try:
                chunks = await semantic_chunking(content)
                logger.info("Semantic chunking completed", chunk_count=len(chunks))
            except HTTPException as e:
                logger.error(f"Error during semantic chunking: {str(e)}")
                raise

            # Check if the collection exists, if not create it
            try:
                if not collection_exists(COLLECTION_NAME):
                    create_collection(COLLECTION_NAME)
                    logger.info(f"Qdrant collection created: {COLLECTION_NAME}")
                else:
                    logger.info(
                        "Qdrant collection already exists",
                        collection_name=COLLECTION_NAME,
                    )
            except Exception as e:
                logger.error(
                    "Error checking/creating Qdrant collection",
                    error=str(e),
                    collection_name=COLLECTION_NAME,
                )
                raise HTTPException(
                    status_code=500, detail="Error with Qdrant collection"
                )

            # Get embeddings for chunks
            try:
                embeddings = await get_embeddings(chunks)
                logger.info("Embeddings generated", embedding_count=len(embeddings))
            except Exception as e:
                logger.error("Error generating embeddings", error=str(e))
                raise HTTPException(
                    status_code=500, detail="Error generating embeddings"
                )

            # Insert chunks into Qdrant
            try:
                insert_chunks(COLLECTION_NAME, chunks, embeddings)
                logger.info(
                    "Chunks inserted into Qdrant",
                    collection_name=COLLECTION_NAME,
                    chunk_count=len(chunks),
                )
            except Exception as e:
                logger.error(
                    "Error inserting chunks into Qdrant",
                    error=str(e),
                    collection_name=COLLECTION_NAME,
                )
                raise HTTPException(
                    status_code=500, detail="Error inserting chunks into Qdrant"
                )

            return ChunkList(
                chunks=[Chunk(text=chunk, metadata={}) for chunk in chunks]
            )
        else:
            logger.warning(
                "File content exceeds maximum token limit", token_count=token_count
            )
            raise HTTPException(
                status_code=413, detail="File content exceeds maximum token limit"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing file", error=str(e), filename=file.filename)
        raise HTTPException(
            status_code=500, detail=f"Error processing {file_extension} file: {str(e)}"
        )


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def should_retry_semantic_chunking(exception):
    """
    Determine if semantic chunking should be retried based on specific error types
    """
    if isinstance(exception, openai.APIError):  # Server API errors
        return True
    if isinstance(exception, openai.APITimeoutError):  # Request timeout
        return True
    if isinstance(exception, openai.APIConnectionError):  # Network issues
        return True
    if isinstance(exception, openai.RateLimitError):  # 429 errors
        return True
    if isinstance(exception, openai.InternalServerError):  # 500 errors
        return True

    # Don't retry on client errors or JSON decode errors
    if isinstance(
        exception,
        (
            openai.BadRequestError,  # 400
            openai.AuthenticationError,  # 401
            openai.PermissionDeniedError,  # 403
            openai.NotFoundError,  # 404
            JSONDecodeError,  # JSON parsing errors
        ),
    ):
        return False

    return False


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


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception(should_retry_semantic_chunking),
)
async def semantic_chunking(content: str) -> List[str]:
    """
    Semantically chunk content with retry logic and fallback to basic chunking
    """
    client = await get_openai_client()
    logger.info("Starting semantic chunking")

    system_prompt = """You are an expert in semantic text analysis and chunking. Your task is to divide the given text into semantically coherent chunks, each with its own metadata. 
    Read the full text of the document. All the information presented in the full text must strictly be covered in the chunks as is without summarizing, editing or altering the text. Your task is to seperate the full text into chunks that are meaningful and coherent.
    
    Follow these guidelines:

    1. Divide the full text into units that are meaningful and coherent. All chunks combined must cover the total text present in the full text.
    2. Each chunk should be a self-contained unit of information. You must not summarize, edit or alter the text in the chunks.
    3. Chunk size can vary, but aim for chunks that are neither too short nor too long.
    4. For each chunk, provide:
    - The chunk text
    - A one-sentence summary
    - Relevant tags (keywords or phrases)
    - Document Metadata (title, author, section, subsection, year)
    5. If there are references section, contents, table of contents, abbreviations section, acknowledgements section, or other irrelevant parts of the text, ignore and exclude them from chunking.
    6. Avoid repetition of the same information.

    Please provide the chunked text in the JSON format specified in the response_format.

    Ensure that the entire text is covered and not altered in the chunks, and that the chunks are semantically meaningful and coherent."""

    # Define the JSON schema for the expected output
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "Semantic_Chunking_Result",
            "schema": {
                "type": "object",
                "properties": {
                    "chunks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "metadata": {
                                    "type": "object",
                                    "properties": {
                                        "summary": {"type": "string"},
                                        "tags": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "document_metadata": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": [
                                        "summary",
                                        "tags",
                                        "document_metadata",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "required": ["text", "metadata"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["chunks"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    try:
        logger.info("Sending request to OpenAI API")
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Here is the full text of the document: {textwrap.dedent(content)}",
                },
            ],
            response_format=response_format,
        )
        logger.info("Semantic chunking API call successful")

        try:
            result = json.loads(response.choices[0].message.content)
            # Format the chunks with metadata
            formatted_chunks = [
                f"{chunk['text']}\n\nMetadata:\nSummary: {chunk['metadata']['summary']}\nTags: {', '.join(chunk['metadata']['tags'])}\nDocument Metadata: {', '.join(chunk['metadata']['document_metadata'])}"
                for chunk in result["chunks"]
            ]
            logger.info(
                "Chunks formatted with metadata", chunk_count=len(formatted_chunks)
            )
            return formatted_chunks

        except JSONDecodeError as e:
            logger.warning(
                f"JSON decode error: {str(e)}. Falling back to basic chunking"
            )
            logger.debug(
                f"Failed response content: {response.choices[0].message.content}"
            )
            return create_basic_chunks(content)

    except Exception as e:
        logger.error(
            f"Error during semantic chunking: {type(e).__name__}", error=str(e)
        )
        if hasattr(e, "response"):
            logger.error(f"API response: {e.response.text}")

        # Fallback to basic chunking for any other errors
        logger.warning(
            "Falling back to basic chunking due to semantic chunking failure"
        )
        return create_basic_chunks(content)


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
