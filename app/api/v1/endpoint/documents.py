import os
import tempfile
import uuid
import logging
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import AsyncQdrantClient

from app.db.postgres import get_db
from app.db.qdrant import get_qdrant
from app.services.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

router = APIRouter()

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    page_count: int
    chunk_count: int

@router.post(
    "/upload", 
    response_model=DocumentUploadResponse, 
    status_code=status.HTTP_201_CREATED
)
async def upload_pdf(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    qdrant: AsyncQdrantClient = Depends(get_qdrant)
):
    # 1. Validate file extension & MIME type
    if not file.filename.endswith(".pdf") or file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF documents are supported."
        )

    document_id = str(uuid.uuid4())
    temp_file_path = None

    try:
        # 2. Write incoming stream to a temporary disk location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name

        # 3. Instantiate and run the Ingestion Pipeline
        pipeline = IngestionPipeline(db_session=db, qdrant_client=qdrant)
        result = await pipeline.run(
            file_path=temp_file_path, 
            document_id=document_id, 
            original_filename=file.filename
        )

        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="completed",
            page_count=result.get("page_count", 0),
            chunk_count=result.get("chunk_count", 0)
        )

    except Exception as e:
        logger.error(f"Failed to process uploaded PDF {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while ingesting the document: {str(e)}"
        )

    finally:
        # 4. Always ensure the temporary file is deleted from disk
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
