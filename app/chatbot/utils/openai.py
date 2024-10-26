import os
import openai
from dotenv import load_dotenv
from fastapi import HTTPException
from .structlogger import logger

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def initialize_openai_client() -> openai.AsyncOpenAI:
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found in environment variables")
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    try:
        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize OpenAI client: {str(e)}"
        )


# Use this function to get the OpenAI client when needed
async def get_openai_client() -> openai.AsyncOpenAI:
    return await initialize_openai_client()
