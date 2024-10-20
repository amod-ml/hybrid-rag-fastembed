from utils.text_extraction import (
    extract_text_from_txt,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_excel,
    extract_text_from_csv,
)
from fastapi import UploadFile, HTTPException


async def process_file(file: UploadFile) -> str:
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
        return await extractor(file)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing {file_extension} file: {str(e)}"
        )
