"""
API routes for document management.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import shutil
from datetime import datetime

from config.settings import Config
from ..services.document_processor import Document, DocumentProcessor

# Set up logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentResponse(BaseModel):
    """Response model for document."""
    
    id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    created_at: str
    updated_at: Optional[str] = None
    chunk_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DocumentStatsResponse(BaseModel):
    """Response model for document statistics."""
    
    total: int
    indexed: int
    processing: int
    pending: int
    error: int
    by_type: Dict[str, int]
    recent_activity: List[Dict[str, Any]]


async def get_db_session() -> AsyncSession:
    """Get database session."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    
    db_url = Config.get_database_uri().replace("postgresql", "postgresql+asyncpg")
    engine = create_async_engine(db_url)
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with session_factory() as session:
        yield session
        await session.commit()


@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    status: Optional[str] = None,
    file_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db_session)
):
    """
    List documents with optional filters.
    
    Args:
        status: Filter by status
        file_type: Filter by file type
        limit: Maximum number of documents to return
        offset: Offset for pagination
        db: Database session
        
    Returns:
        List of documents
    """
    try:
        query = sa.select(Document).order_by(Document.created_at.desc())
        
        if status:
            query = query.where(Document.status == status)
        
        if file_type:
            query = query.where(Document.file_type == file_type)
        
        query = query.limit(limit).offset(offset)
        
        result = await db.execute(query)
        documents = result.scalars().all()
        
        return [doc.to_dict() for doc in documents]
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=DocumentStatsResponse)
async def get_document_stats(db: AsyncSession = Depends(get_db_session)):
    """
    Get document statistics.
    
    Args:
        db: Database session
        
    Returns:
        Document statistics
    """
    try:
        # Get total count
        total_query = sa.select(sa.func.count(Document.id))
        total_result = await db.execute(total_query)
        total = total_result.scalar() or 0
        
        # Get counts by status
        status_query = sa.select(Document.status, sa.func.count(Document.id)).group_by(Document.status)
        status_result = await db.execute(status_query)
        status_counts = dict(status_result.fetchall())
        
        # Get counts by file type
        type_query = sa.select(Document.file_type, sa.func.count(Document.id)).group_by(Document.file_type)
        type_result = await db.execute(type_query)
        type_counts = dict(type_result.fetchall())
        
        # Get recent activity (last 10 documents updated)
        activity_query = sa.select(Document).order_by(Document.updated_at.desc()).limit(10)
        activity_result = await db.execute(activity_query)
        recent_docs = activity_result.scalars().all()
        
        recent_activity = []
        for doc in recent_docs:
            action = "Unknown"
            
            if doc.status == "indexed":
                action = "Indexed document"
            elif doc.status == "processing":
                action = "Processing document"
            elif doc.status == "pending":
                action = "Added document"
            elif doc.status == "error":
                action = "Error processing document"
            
            details = f"{doc.filename}"
            if doc.status == "error" and doc.error:
                details += f" - {doc.error}"
            
            activity = {
                "timestamp": doc.updated_at.isoformat() if doc.updated_at else doc.created_at.isoformat(),
                "document_id": doc.id,
                "filename": doc.filename,
                "action": action,
                "details": details
            }
            
            recent_activity.append(activity)
        
        return {
            "total": total,
            "indexed": status_counts.get("indexed", 0),
            "processing": status_counts.get("processing", 0),
            "pending": status_counts.get("pending", 0),
            "error": status_counts.get("error", 0),
            "by_type": type_counts,
            "recent_activity": recent_activity
        }
        
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, db: AsyncSession = Depends(get_db_session)):
    """
    Get document by ID.
    
    Args:
        document_id: Document ID
        db: Database session
        
    Returns:
        Document details
    """
    try:
        query = sa.select(Document).where(Document.id == document_id)
        result = await db.execute(query)
        document = result.scalar_one_or_none()
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        return document.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Upload a document.
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded file
        db: Database session
        
    Returns:
        Uploaded document details
    """
    try:
        # Get upload directory
        upload_dir = os.environ.get("UPLOAD_DIR", "/app/uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save file to upload directory
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Generate ID
        import hashlib
        file_stat = os.stat(file_path)
        unique_string = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
        file_id = hashlib.md5(unique_string.encode()).hexdigest()
        
        # Get file type
        file_type = os.path.splitext(file.filename)[1].lower().lstrip(".")
        
        # Create document
        document = Document(
            id=file_id,
            filename=file.filename,
            original_path=file_path,
            file_type=file_type,
            file_size=file_stat.st_size,
            status="pending",
            metadata={
                "source": "api_upload",
                "content_type": file.content_type
            }
        )
        
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        return document.to_dict()
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Delete a document.
    
    Args:
        document_id: Document ID
        db: Database session
        
    Returns:
        Success message
    """
    try:
        # Get document
        query = sa.select(Document).where(Document.id == document_id)
        result = await db.execute(query)
        document = result.scalar_one_or_none()
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Delete file if it exists
        if document.original_path and os.path.exists(document.original_path):
            os.remove(document.original_path)
        
        if document.stored_path and os.path.exists(document.stored_path):
            os.remove(document.stored_path)
        
        # Delete from MinIO if available
        if document.minio_bucket and document.minio_object:
            try:
                # This should be improved with proper MinIO client handling
                from minio import Minio
                
                minio_endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
                minio_access_key = os.environ.get("MINIO_ACCESS_KEY", "minio_user")
                minio_secret_key = os.environ.get("MINIO_SECRET_KEY", "minio_password")
                minio_use_ssl = os.environ.get("MINIO_USE_SSL", "false").lower() == "true"
                
                minio_client = Minio(
                    minio_endpoint,
                    access_key=minio_access_key,
                    secret_key=minio_secret_key,
                    secure=minio_use_ssl,
                )
                
                await asyncio.to_thread(
                    minio_client.remove_object,
                    document.minio_bucket,
                    document.minio_object
                )
                
            except Exception as e:
                logger.error(f"Error deleting object from MinIO: {e}")
        
        # Delete document from database
        delete_query = sa.delete(Document).where(Document.id == document_id)
        await db.execute(delete_query)
        await db.commit()
        
        return {"message": f"Document {document_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{document_id}/reindex")
async def reindex_document(
    document_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Reindex a document.
    
    Args:
        document_id: Document ID
        db: Database session
        
    Returns:
        Success message
    """
    try:
        # Get document
        query = sa.select(Document).where(Document.id == document_id)
        result = await db.execute(query)
        document = result.scalar_one_or_none()
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Update status to pending
        document.status = "pending"
        document.error = None
        document.updated_at = datetime.utcnow()
        
        await db.commit()
        
        return {"message": f"Document {document_id} queued for reindexing"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reindexing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def register_routes(app):
    """Register document routes with the FastAPI app."""
    app.include_router(router)