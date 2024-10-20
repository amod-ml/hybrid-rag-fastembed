from utils.text_extraction import (
    extract_text_from_txt,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_excel,
    extract_text_from_csv,
)
from fastapi import UploadFile, HTTPException
import tiktoken
from jinja2 import Environment, FileSystemLoader
import json
from typing import List
from ..utils.openai import get_openai_client


async def process_file(file: UploadFile) -> List[str]:
    file_extension = file.filename.split(".")[-1].lower()

    extractors = {
        "txt": extract_text_from_txt,
        "pdf": extract_text_from_pdf,
        "docx": extract_text_from_docx,
        "xlsx": extract_text_from_excel,
        "xls": extract_text_from_excel,
        "csv": extract_text_from_csv,
    }

    extractor = extractors.get(file_extension)
    if not extractor:
        raise HTTPException(
            status_code=402, detail=f"Unsupported file format: {file_extension}"
        )

    try:
        content = await extractor(file)
        token_count = num_tokens_from_string(content, "o200k_base")

        if token_count <= 128000:
            return await semantic_chunking(content)
        else:
            raise HTTPException(
                status_code=413, detail="File content exceeds maximum token limit"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing {file_extension} file: {str(e)}"
        )


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


async def semantic_chunking(content: str) -> List[str]:
    client = await get_openai_client()

    # Load the Jinja2 template
    env = Environment(loader=FileSystemLoader("chatbot/templates"))
    template = env.get_template("semantic_chunking_prompt.j2")

    # Render the template with the content
    system_prompt = template.render(content=content)

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
                                    },
                                    "required": ["summary", "tags"],
                                },
                            },
                            "required": ["text", "metadata"],
                        },
                    }
                },
                "required": ["chunks"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "assistant",
                "content": "Here is the chunked text in the requested format:",
            },
        ],
        response_format=response_format,
    )

    result = json.loads(response.choices[0].message.content)

    # Format the chunks with metadata
    formatted_chunks = [
        f"{chunk['text']}\n\nMetadata:\nSummary: {chunk['metadata']['summary']}\nTags: {', '.join(chunk['metadata']['tags'])}"
        for chunk in result["chunks"]
    ]

    return formatted_chunks
