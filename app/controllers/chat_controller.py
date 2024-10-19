import asyncio
from typing import Dict
from motor.motor_asyncio import AsyncIOMotorClient
from app.services.mongodb_service import fetch_patient_data
from app.services.s3_service import fetch_files_from_s3
from app.services.openai_service import generate_ai_response
from app.utils.file_converter import convert_pdf_to_image

async def patient_chat_controller(user_id: str, conversation_id: str, query: str) -> Dict:
    """
    Handle patient chat requests by fetching necessary medical data 
    from MongoDB and S3 and passing it to the AI model.
    
    :param user_id: Patient's user ID (must start with PID)
    :param conversation_id: Unique conversation ID
    :param query: Query string from the patient
    :return: AI-generated response with patient context
    """
    if not user_id.startswith("PID"):
        raise ValueError("Invalid patient ID format")

    # Fetch patient data from MongoDB
    patient_data = await fetch_patient_data(user_id)
    
    # Fetch files from S3 and convert if necessary
    s3_files = await fetch_files_from_s3(user_id)
    converted_files = await asyncio.gather(*[convert_pdf_to_image(file) for file in s3_files if file.endswith('.pdf')])
    
    # Prepare context for AI
    context = {
        "patient_data": patient_data,
        "files": converted_files,
        "query": query,
        "conversation_id": conversation_id
    }
    
    # Pass all data to OpenAI for a response
    ai_response = await generate_ai_response(context)
    
    return {
        "response": ai_response,
        "conversation_id": conversation_id
    }
