import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils.structlogger import logger
from .router import router
from .utils.session_manager import SessionManager

app = FastAPI()


@app.get("/status")
async def status():
    return {"status": "alive"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

session_manager = SessionManager(timeout=900)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(session_manager.clear_inactive_sessions())


logger.info("Chatbot server started")
