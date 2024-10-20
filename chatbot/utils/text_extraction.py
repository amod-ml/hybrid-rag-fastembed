import io
import csv
from docx import Document
from openpyxl import load_workbook
from fastapi import UploadFile
import fitz  # PyMuPDF


async def extract_text_from_txt(file: UploadFile) -> str:
    content = await file.read()
    return content.decode("utf-8")


async def extract_text_from_pdf(file: UploadFile) -> str:
    content = await file.read()
    pdf = fitz.open(stream=content, filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text


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
