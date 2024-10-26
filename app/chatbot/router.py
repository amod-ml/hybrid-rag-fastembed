from fastapi import APIRouter
from .routes import chat, file_ingestion, status

router = APIRouter()


router.include_router(
    file_ingestion.router,
    tags=["File Ingestion"],
)
router.include_router(
    chat.router,
    tags=["Chat"],
)
router.include_router(
    status.router,
    tags=["Status"],
)
