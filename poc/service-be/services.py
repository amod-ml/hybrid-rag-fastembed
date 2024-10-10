from fastapi import HTTPException
from dotenv import load_dotenv
from .structlogger import logger
import openai
import os
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
import certifi
import asyncio

load_dotenv()

OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
MONGODB_URI: str = "mongodb+srv://amods:W0FefrWkryHjWWGx@amodscluster.qymv0.mongodb.net/?retryWrites=true&w=majority&appName=AmodsCluster"


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


async def initialize_mongodb_client() -> AsyncIOMotorClient:
    try:
        client = AsyncIOMotorClient(
            MONGODB_URI,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=30000,  # Increase timeout to 30 seconds
            connectTimeoutMS=30000,  # Add connect timeout
            socketTimeoutMS=30000,  # Add socket timeout
            maxPoolSize=1,  # Reduce pool size for testing
            retryWrites=True,
            w="majority",
        )
        # Use asyncio.wait_for to set a timeout for the ping operation
        await asyncio.wait_for(client.admin.command("ping"), timeout=30.0)
        logger.info("MongoDB client initialized successfully")
        return client
    except asyncio.TimeoutError:
        logger.error("Timeout while connecting to MongoDB")
        raise HTTPException(
            status_code=500, detail="Timeout while connecting to MongoDB"
        )
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB client: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize MongoDB client: {str(e)}"
        )


# Use this function to get the MongoDB client when needed
async def get_mongodb_client() -> AsyncIOMotorClient:
    return await initialize_mongodb_client()
