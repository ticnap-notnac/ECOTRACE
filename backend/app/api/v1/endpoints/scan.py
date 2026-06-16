from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.database import get_db
from app.dependencies import get_current_user
from app.services.scan_service import ScanService
from app.schemas.scan import ScanResultResponse
import google.generativeai as genai
from app.config import settings

router = APIRouter()

@router.get("/models")
async def list_available_models():
    try:
        keys = settings.gemini_api_keys
        if not keys:
            keys = [settings.GEMINI_API_KEY]
        genai.configure(api_key=keys[0])
        models = [m.name for m in genai.list_models()]
        return {"models": models}
    except Exception as e:
        return {"error": str(e)}

@router.post("/analyze", response_model=ScanResultResponse)
async def analyze_scan(
    scan_type: str = Form(...),
    image: UploadFile = File(...),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    if scan_type not in ["receipt", "barcode", "packaging"]:
        raise HTTPException(status_code=400, detail="Invalid scan_type")
        
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image bytes
    image_bytes = await image.read()
    
    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 5MB)")

    result = await ScanService.process_scan(
        db=db,
        user_id=current_user.id,
        image_bytes=image_bytes,
        scan_type=scan_type,
        mime_type=image.content_type
    )
    
    return result

@router.get("/history", response_model=List[ScanResultResponse])
async def get_scan_history(
    limit: int = 10,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    return await ScanService.get_history(db, current_user.id, limit)
