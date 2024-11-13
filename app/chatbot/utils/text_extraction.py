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
        'sentences': r'[A-Z][^.!?]*[.!?]',  # Looks for proper sentences
        'words': r'\b\w{2,}\b',  # Words with 2 or more characters
        'alphanumeric': r'[A-Za-z0-9]+',  # Any alphanumeric sequences
        'structured_data': r'[\t:,]\s*\w+',  # Common data delimiters with content
    }
    
    is_meaningful = (
        len(text_stripped) > 50 and  # Basic length check
        bool(re.search(patterns['sentences'], text_stripped)) and  # Has proper sentences
        len(re.findall(patterns['words'], text_stripped)) > 10  # Has sufficient words
    )
    
    if is_meaningful:
        logger.info("Successfully extracted meaningful text directly from PDF")
        return text
    
    logger.info("Direct text extraction failed quality checks, attempting OCR with GPT-4V")
    
    # Fallback: Convert pages to images and use GPT-4V
    try:
        client = await get_openai_client()
        combined_text = ""
        
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            # Convert page to PNG image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better quality
            img_data = pix.tobytes("png")
            
            # Convert to base64
            base64_image = base64.b64encode(img_data).decode('utf-8')
            
            # Create message for GPT-4V
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please extract and return ONLY the text content from this image. Format it naturally with appropriate line breaks and spacing. Ignore any visual elements and optimize for retaining layout information."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
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
