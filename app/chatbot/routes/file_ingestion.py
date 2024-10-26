from fastapi import APIRouter, UploadFile, File, HTTPException
from ..models import FileUploadResponse
from ..core.file_ingestion_controller import process_file
from ..utils.structlogger import logger

router = APIRouter()


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        chunk_list = await process_file(file)
        return FileUploadResponse(
            filename=file.filename,
            chunks_inserted=len(chunk_list.chunks),
            message="File processed and chunks inserted successfully",
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
