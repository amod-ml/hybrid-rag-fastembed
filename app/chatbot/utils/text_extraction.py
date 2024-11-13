import io
import csv
import base64
import regex as re
from docx import Document
from openpyxl import load_workbook
from fastapi import UploadFile
import fitz  # PyMuPDF
from .openai import get_openai_client
from .structlogger import logger
import os
from datetime import datetime
import textwrap
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
)
import openai


def should_retry_error(exception):
    """
    Determine if the error should trigger a retry based on specific error types and status codes
    """
    if isinstance(exception, openai.APIError):  # Server API errors
        return True
    if isinstance(exception, openai.APITimeoutError):  # Request timeout
        return True
    if isinstance(exception, openai.APIConnectionError):  # Network issues
        return True

    # Check for specific HTTP status codes that warrant retries
    if isinstance(exception, openai.BadRequestError):  # 400 errors
        return False  # Don't retry bad requests
    if isinstance(exception, openai.AuthenticationError):  # 401 errors
        return False  # Don't retry auth errors
    if isinstance(exception, openai.PermissionDeniedError):  # 403 errors
        return False  # Don't retry permission errors
    if isinstance(exception, openai.NotFoundError):  # 404 errors
        return False  # Don't retry not found errors
    if isinstance(exception, openai.RateLimitError):  # 429 errors
        return True  # Retry rate limit errors
    if isinstance(exception, openai.InternalServerError):  # 500 errors
        return True  # Retry internal server errors

    return False


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception(should_retry_error),
)
async def gpt4v_process_image(client, messages):
    """
    Process image with GPT-4V with retry logic for specific error conditions
    """
    try:
        return await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
    except Exception as e:
        logger.error(f"GPT-4V API error: {str(e)}")
        raise


async def extract_text_from_txt(file: UploadFile) -> str:
    content = await file.read()
    return content.decode("utf-8")


async def extract_text_from_pdf(file: UploadFile) -> str:
    """
    Extract text from PDF with fallback to OCR using GPT-4V for image-based PDFs
    """
    content = await file.read()
    pdf = fitz.open(stream=content, filetype="pdf")

    # First attempt: Direct text extraction
    text = ""
    for page in pdf:
        text += page.get_text()

    # Check if meaningful text was extracted using multiple criteria
    text_stripped = text.strip()

    # Regex patterns for meaningful text detection
    patterns = {
        "sentences": r"[A-Z][^.!?]*[.!?]",  # Looks for proper sentences
        "words": r"\b\w{2,}\b",  # Words with 2 or more characters
        "alphanumeric": r"[A-Za-z0-9]+",  # Any alphanumeric sequences
        "structured_data": r"[\t:,]\s*\w+",  # Common data delimiters with content
    }

    is_meaningful = (
        len(text_stripped) > 50  # Basic length check
        and bool(
            re.search(patterns["sentences"], text_stripped)
        )  # Has proper sentences
        and len(re.findall(patterns["words"], text_stripped))
        > 10  # Has sufficient words
    )

    if is_meaningful:
        logger.info("Successfully extracted meaningful text directly from PDF")
        return text

    logger.info(
        "Direct text extraction failed quality checks, attempting OCR with GPT-4V"
    )

    # Fallback: Convert pages to images and use GPT-4V
    try:
        client = await get_openai_client()
        combined_text = ""

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            # Convert page to PNG image
            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2)
            )  # 2x scaling for better quality
            img_data = pix.tobytes("png")

            # Convert to base64
            base64_image = base64.b64encode(img_data).decode("utf-8")

            # Improved prompt for better layout preservation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": textwrap.dedent(
                                """Please extract ALL text content from this image, maintaining the following:
                                        1. Preserve paragraph breaks and indentation
                                        2. Keep table-like structures aligned with tabs or spaces
                                        3. Maintain list formatting and numbering
                                        4. Preserve section headers and their hierarchy
                                        5. Keep any important line breaks that indicate document structure
                                        6. Retain column-based text layout where present

                                    Extract ONLY the text content - DO NOT ADD ANY DESCRIPTIONS, ANNOTATIONS, OR EXPLANATIONS. Format exactly as it appears."""
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ]

            # Get response from GPT-4V
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )

            page_text = response.choices[0].message.content
            combined_text += f"\n{page_text}\n"
            logger.info(f"Processed page {page_num + 1}/{len(pdf)} with GPT-4V")

        # Save debug output
        debug_dir = "debug_extractions"
        os.makedirs(debug_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{debug_dir}/extracted_text_{timestamp}.txt"

        # Find next available filename
        counter = 1
        while os.path.exists(filename):
            filename = f"{debug_dir}/extracted_text_{timestamp}_{counter}.txt"
            counter += 1

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Original Extraction Attempt:\n{'='*50}\n")
                f.write(text_stripped)
                f.write(f"\n\nGPT-4V OCR Extraction:\n{'='*50}\n")
                f.write(combined_text)
            logger.info(f"Debug output saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save debug output: {str(e)}")

        logger.info("Successfully extracted text using GPT-4V OCR")
        return combined_text.strip()

    except Exception as e:
        logger.error(f"Error during GPT-4V OCR processing: {str(e)}")
        return text_stripped


async def extract_text_from_docx(file: UploadFile) -> str:
    content = await file.read()
    doc = Document(io.BytesIO(content))
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])


async def extract_text_from_excel(file: UploadFile) -> str:
    content = await file.read()
    wb = load_workbook(filename=io.BytesIO(content))
    text = ""
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        for row in ws.iter_rows(values_only=True):
            text += "\t".join([str(cell) for cell in row if cell is not None]) + "\n"
    return text


async def extract_text_from_csv(file: UploadFile) -> str:
    content = await file.read()
    csv_content = content.decode("utf-8").splitlines()
    reader = csv.reader(csv_content)
    return "\n".join(["\t".join(row) for row in reader])
