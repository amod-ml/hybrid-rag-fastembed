import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .utils.structlogger import logger
from .router import router
from .utils.session_manager import SessionManager
from .core.file_ingestion_controller import process_file
from .models import FileUploadResponse

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


@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        chunk_list = await process_file(file)
        return FileUploadResponse(
            filename=file.filename,
            chunks_inserted=len(chunk_list.chunks),
            message="File processed and chunks inserted successfully"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
