import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .utils.structlogger import logger
from .router import router
from .utils.conversation_manager import ConversationManager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

conversation_manager = ConversationManager(timeout=900)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(clear_inactive_conversations())

async def clear_inactive_conversations():
    while True:
        conversation_manager.clear_inactive_conversations()
        await asyncio.sleep(300)  # Check every 5 minutes

logger.info("Chatbot server started")
