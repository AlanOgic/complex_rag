"""
Enhanced API routes for document management with advanced processing features.
"""

import logging
import os
import json
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, BackgroundTasks, Path
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import shutil
import tempfile
from datetime import datetime, timedelta
import io

from config.settings import Config
from ..services.document_processor_v2 import Document, ProcessingHistory, ProcessingMetrics

# Set up logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/documents/v2", tags=["documents"])


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
    quality_score: Optional[float] = None
    language: Optional[str] = None
    ocr_applied: Optional[bool] = None
    ocr_quality: Optional[float] = None


class DocumentDetailResponse(DocumentResponse):
    """Detailed response model for document."""
    
    processing_info: Optional[Dict[str, Any]] = None
    processing_history: Optional[List[Dict[str, Any]]] = None


class DocumentStatsResponse(BaseModel):
    """Response model for document statistics."""
    
    total: int
    indexed: int
    processing: int
    pending: int
    error: int
    by_type: Dict[str, int]
    by_language: Dict[str, int]
    quality_stats: Dict[str, float]
    recent_activity: List[Dict[str, Any]]
    time_stats: Dict[str, float]


class ProcessingConfigResponse(BaseModel):
    """Response model for processing configuration."""
    
    enable_ocr: bool
    enable_quality_check: bool
    min_quality_score: float
    max_processing_attempts: int
    parallel_processing: int
    chunking_config: Dict[str, Any]
    quality_config: Dict[str, Any]
    ocr_config: Dict[str, Any]


class DocumentSearchParams(BaseModel):
    """Parameters for document search."""
    
    status: Optional[List[str]] = None
    file_type: Optional[List[str]] = None
    language: Optional[List[str]] = None
    min_quality: Optional[float] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    content_search: Optional[str] = None
    source: Optional[str] = None


class BulkDocumentAction(BaseModel):
    """Bulk action for documents."""
    
    document_ids: List[str]
    action: str  # "delete", "reindex", etc.


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


async def get_document_processing_config() -> Dict[str, Any]:
    """Get document processing configuration from environment variables."""
    config = {
        "enable_ocr": os.environ.get("ENABLE_OCR", "true").lower() == "true",
        "enable_quality_check": os.environ.get("ENABLE_QUALITY_CHECK", "true").lower() == "true",
        "min_quality_score": float(os.environ.get("MIN_QUALITY_SCORE", "0.5")),
        "max_processing_attempts": int(os.environ.get("MAX_PROCESSING_ATTEMPTS", "3")),
        "parallel_processing": int(os.environ.get("PARALLEL_PROCESSING", "4")),
        
        "chunking_config": {
            "chunk_size": int(os.environ.get("CHUNK_SIZE", str(Config.CHUNK_SIZE))),
            "chunk_overlap": int(os.environ.get("CHUNK_OVERLAP", str(Config.CHUNK_OVERLAP))),
            "strategy": os.environ.get("CHUNKING_STRATEGY", Config.CHUNKING_STRATEGY),
            "respect_boundaries": os.environ.get("RESPECT_BOUNDARIES", "true").lower() == "true"
        },
        
        "quality_config": {
            "min_content_length": int(os.environ.get("MIN_CONTENT_LENGTH", "50")),
            "max_noise_ratio": float(os.environ.get("MAX_NOISE_RATIO", "0.3")),
            "language_detection_threshold": float(os.environ.get("LANGUAGE_DETECTION_THRESHOLD", "0.8")),
            "deduplication_threshold": float(os.environ.get("DEDUPLICATION_THRESHOLD", "0.85"))
        },
        
        "ocr_config": {
            "ocr_engine": os.environ.get("OCR_ENGINE", "tesseract"),
            "languages": os.environ.get("OCR_LANGUAGES", "eng").split(","),
            "detect_tables": os.environ.get("DETECT_TABLES", "true").lower() == "true",
            "preserve_layout": os.environ.get("PRESERVE_LAYOUT", "true").lower() == "true",
            "confidence_threshold": float(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "60.0"))
        }
    }
    
    return config


@router.get("/config", response_model=ProcessingConfigResponse)
async def get_config():
    """
    Get document processing configuration.
    
    Returns:
        Document processing configuration
    """
    try:
        config = await get_document_processing_config()
        return config
        
    except Exception as e:
        logger.error(f"Error getting processing configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    status: Optional[str] = None,
    file_type: Optional[str] = None,
    language: Optional[str] = None,
    min_quality: Optional[float] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort_by: str = "created_at",
    sort_dir: str = "desc",
    db: AsyncSession = Depends(get_db_session)
):
    """
    List documents with optional filters.
    
    Args:
        status: Filter by status
        file_type: Filter by file type
        language: Filter by language
        min_quality: Filter by minimum quality score
        limit: Maximum number of documents to return
        offset: Offset for pagination
        sort_by: Field to sort by
        sort_dir: Sort direction (asc/desc)
        db: Database session
        
    Returns:
        List of documents
    """
    try:
        query = sa.select(Document)
        
        # Apply filters
        if status:
            query = query.where(Document.status == status)
        
        if file_type:
            query = query.where(Document.file_type == file_type)
        
        if language:
            query = query.where(Document.language == language)
        
        if min_quality is not None:
            query = query.where(Document.quality_score >= min_quality)
        
        # Apply sorting
        sort_column = getattr(Document, sort_by, Document.created_at)
        if sort_dir.lower() == "desc":
            sort_column = sort_column.desc()
        else:
            sort_column = sort_column.asc()
        
        query = query.order_by(sort_column)
        
        # Apply pagination
        query = query.limit(limit).offset(offset)
        
        result = await db.execute(query)
        documents = result.scalars().all()
        
        return [doc.to_dict() for doc in documents]
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=List[DocumentResponse])
async def search_documents(
    search_params: DocumentSearchParams,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort_by: str = "created_at",
    sort_dir: str = "desc",
    db: AsyncSession = Depends(get_db_session)
):
    """
    Advanced search for documents with multiple criteria.
    
    Args:
        search_params: Search parameters
        limit: Maximum number of documents to return
        offset: Offset for pagination
        sort_by: Field to sort by
        sort_dir: Sort direction (asc/desc)
        db: Database session
        
    Returns:
        List of matching documents
    """
    try:
        query = sa.select(Document)
        
        # Apply filters from search parameters
        if search_params.status:
            query = query.where(Document.status.in_(search_params.status))
        
        if search_params.file_type:
            query = query.where(Document.file_type.in_(search_params.file_type))
        
        if search_params.language:
            query = query.where(Document.language.in_(search_params.language))
        
        if search_params.min_quality is not None:
            query = query.where(Document.quality_score >= search_params.min_quality)
        
        if search_params.date_from:
            query = query.where(Document.created_at >= search_params.date_from)
        
        if search_params.date_to:
            query = query.where(Document.created_at <= search_params.date_to)
        
        if search_params.source:
            # Search in metadata.source field
            query = query.where(
                Document.metadata.op('->>')('source').contains(search_params.source)
            )
        
        # TODO: Implement content search with full-text search if needed
        
        # Apply sorting
        sort_column = getattr(Document, sort_by, Document.created_at)
        if sort_dir.lower() == "desc":
            sort_column = sort_column.desc()
        else:
            sort_column = sort_column.asc()
        
        query = query.order_by(sort_column)
        
        # Apply pagination
        query = query.limit(limit).offset(offset)
        
        result = await db.execute(query)
        documents = result.scalars().all()
        
        return [doc.to_dict() for doc in documents]
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=DocumentStatsResponse)
async def get_document_stats(
    time_period: str = "all",
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get comprehensive document statistics.
    
    Args:
        time_period: Time period for stats ("day", "week", "month", "all")
        db: Database session
        
    Returns:
        Document statistics
    """
    try:
        # Set up date filter based on time period
        date_filter = None
        if time_period == "day":
            date_filter = datetime.utcnow() - timedelta(days=1)
        elif time_period == "week":
            date_filter = datetime.utcnow() - timedelta(weeks=1)
        elif time_period == "month":
            date_filter = datetime.utcnow() - timedelta(days=30)
        
        # Get total count
        total_query = sa.select(sa.func.count(Document.id))
        if date_filter:
            total_query = total_query.where(Document.created_at >= date_filter)
        
        total_result = await db.execute(total_query)
        total = total_result.scalar() or 0
        
        # Get counts by status
        status_query = sa.select(Document.status, sa.func.count(Document.id)).group_by(Document.status)
        if date_filter:
            status_query = status_query.where(Document.created_at >= date_filter)
        
        status_result = await db.execute(status_query)
        status_counts = dict(status_result.fetchall())
        
        # Get counts by file type
        type_query = sa.select(Document.file_type, sa.func.count(Document.id)).group_by(Document.file_type)
        if date_filter:
            type_query = type_query.where(Document.created_at >= date_filter)
        
        type_result = await db.execute(type_query)
        type_counts = dict(type_result.fetchall())
        
        # Get counts by language
        lang_query = sa.select(Document.language, sa.func.count(Document.id)).group_by(Document.language)
        if date_filter:
            lang_query = lang_query.where(Document.created_at >= date_filter)
        
        lang_result = await db.execute(lang_query)
        lang_counts = dict(lang_result.fetchall())
        
        # Get quality statistics
        quality_query = sa.select(
            sa.func.avg(Document.quality_score).label("avg_quality"),
            sa.func.min(Document.quality_score).label("min_quality"),
            sa.func.max(Document.quality_score).label("max_quality"),
            sa.func.avg(Document.ocr_quality).label("avg_ocr_quality")
        )
        if date_filter:
            quality_query = quality_query.where(Document.created_at >= date_filter)
        
        quality_result = await db.execute(quality_query)
        quality_stats = dict(zip(["avg_quality", "min_quality", "max_quality", "avg_ocr_quality"], 
                                quality_result.first()))
        
        # Get processing time statistics from metrics
        time_query = sa.select(
            ProcessingMetrics.metric_name,
            sa.func.avg(ProcessingMetrics.value).label("avg_time")
        ).where(
            ProcessingMetrics.metric_type == "processing_time"
        ).group_by(ProcessingMetrics.metric_name)
        
        if date_filter:
            time_query = time_query.where(ProcessingMetrics.timestamp >= date_filter)
        
        time_result = await db.execute(time_query)
        time_stats = {row[0]: row[1] for row in time_result.fetchall()}
        
        # Get recent activity (last 10 documents or processing events)
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
                "details": details,
                "language": doc.language,
                "quality_score": doc.quality_score
            }
            
            recent_activity.append(activity)
        
        return {
            "total": total,
            "indexed": status_counts.get("indexed", 0),
            "processing": status_counts.get("processing", 0),
            "pending": status_counts.get("pending", 0),
            "error": status_counts.get("error", 0),
            "by_type": type_counts,
            "by_language": lang_counts,
            "quality_stats": quality_stats,
            "recent_activity": recent_activity,
            "time_stats": time_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document_detail(
    document_id: str,
    include_history: bool = False,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get detailed document information including processing history.
    
    Args:
        document_id: Document ID
        include_history: Whether to include processing history
        db: Database session
        
    Returns:
        Document details
    """
    try:
        # Get document
        query = sa.select(Document).where(Document.id == document_id)
        result = await db.execute(query)
        document = result.scalar_one_or_none()
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Create response
        response = document.to_dict()
        
        # Get processing history if requested
        if include_history:
            history_query = sa.select(ProcessingHistory).where(
                ProcessingHistory.document_id == document_id
            ).order_by(ProcessingHistory.started_at.desc())
            
            history_result = await db.execute(history_query)
            history = history_result.scalars().all()
            
            history_list = []
            for record in history:
                history_list.append({
                    "id": record.id,
                    "stage": record.stage,
                    "status": record.status,
                    "started_at": record.started_at.isoformat() if record.started_at else None,
                    "completed_at": record.completed_at.isoformat() if record.completed_at else None,
                    "duration": record.duration,
                    "details": record.details,
                    "error": record.error
                })
            
            response["processing_history"] = history_list
            
        # Include processing info if available
        response["processing_info"] = document.processing_info
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auto_process: bool = Query(True, description="Start processing immediately"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Upload a document.
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded file
        auto_process: Start processing immediately
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
        
        # Check if document already exists
        query = sa.select(Document).where(Document.id == file_id)
        result = await db.execute(query)
        existing_doc = result.scalar_one_or_none()
        
        if existing_doc:
            # Document already exists
            return existing_doc.to_dict()
        
        # Create document
        document = Document(
            id=file_id,
            filename=file.filename,
            original_path=file_path,
            file_type=file_type,
            file_size=file_stat.st_size,
            status="pending" if auto_process else "uploaded",
            metadata={
                "source": "api_upload",
                "content_type": file.content_type,
                "auto_process": auto_process
            }
        )
        
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        return document.to_dict()
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk", response_model=Dict[str, Any])
async def bulk_document_action(
    action_request: BulkDocumentAction,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Perform bulk action on multiple documents.
    
    Args:
        action_request: Bulk action request
        db: Database session
        
    Returns:
        Action result
    """
    try:
        if not action_request.document_ids:
            raise HTTPException(status_code=400, detail="No document IDs provided")
        
        action = action_request.action.lower()
        result = {
            "action": action,
            "total": len(action_request.document_ids),
            "successful": 0,
            "failed": 0,
            "details": []
        }
        
        if action == "delete":
            # Delete documents
            for doc_id in action_request.document_ids:
                try:
                    # Get document
                    query = sa.select(Document).where(Document.id == doc_id)
                    doc_result = await db.execute(query)
                    document = doc_result.scalar_one_or_none()
                    
                    if not document:
                        result["failed"] += 1
                        result["details"].append({
                            "id": doc_id,
                            "status": "failed",
                            "reason": "Document not found"
                        })
                        continue
                    
                    # Delete file if it exists
                    if document.original_path and os.path.exists(document.original_path):
                        os.remove(document.original_path)
                    
                    if document.stored_path and os.path.exists(document.stored_path):
                        os.remove(document.stored_path)
                    
                    # Delete document
                    delete_query = sa.delete(Document).where(Document.id == doc_id)
                    await db.execute(delete_query)
                    
                    result["successful"] += 1
                    result["details"].append({
                        "id": doc_id,
                        "status": "success"
                    })
                    
                except Exception as e:
                    result["failed"] += 1
                    result["details"].append({
                        "id": doc_id,
                        "status": "failed",
                        "reason": str(e)
                    })
            
            await db.commit()
            
        elif action == "reindex":
            # Reindex documents (mark as pending)
            for doc_id in action_request.document_ids:
                try:
                    # Get document
                    query = sa.select(Document).where(Document.id == doc_id)
                    doc_result = await db.execute(query)
                    document = doc_result.scalar_one_or_none()
                    
                    if not document:
                        result["failed"] += 1
                        result["details"].append({
                            "id": doc_id,
                            "status": "failed",
                            "reason": "Document not found"
                        })
                        continue
                    
                    # Reset processing info and errors
                    document.status = "pending"
                    document.error = None
                    document.processing_info = None
                    document.updated_at = datetime.utcnow()
                    
                    result["successful"] += 1
                    result["details"].append({
                        "id": doc_id,
                        "status": "success"
                    })
                    
                except Exception as e:
                    result["failed"] += 1
                    result["details"].append({
                        "id": doc_id,
                        "status": "failed",
                        "reason": str(e)
                    })
            
            await db.commit()
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported action: {action}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing bulk action: {e}")
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
        
        # Delete processing history
        history_delete_query = sa.delete(ProcessingHistory).where(
            ProcessingHistory.document_id == document_id
        )
        await db.execute(history_delete_query)
        
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
        document.processing_info = None
        document.updated_at = datetime.utcnow()
        
        await db.commit()
        
        return {"message": f"Document {document_id} queued for reindexing"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reindexing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Download the original document.
    
    Args:
        document_id: Document ID
        db: Database session
        
    Returns:
        Document file
    """
    try:
        # Get document
        query = sa.select(Document).where(Document.id == document_id)
        result = await db.execute(query)
        document = result.scalar_one_or_none()
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Get file path
        file_path = document.original_path or document.stored_path
        
        if not file_path or not os.path.exists(file_path):
            # Try to get from MinIO
            if document.minio_bucket and document.minio_object:
                try:
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
                    
                    # Get object and stream to temp file
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        await asyncio.to_thread(
                            minio_client.fget_object,
                            document.minio_bucket,
                            document.minio_object,
                            temp_file.name
                        )
                        
                        file_path = temp_file.name
                    
                except Exception as e:
                    logger.error(f"Error retrieving document from MinIO: {e}")
                    raise HTTPException(status_code=500, detail="Error retrieving document")
            else:
                raise HTTPException(status_code=404, detail="Document file not found")
        
        # Return file as streaming response
        return StreamingResponse(
            open(file_path, "rb"),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{document.filename}"'}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/processing_time")
async def get_processing_metrics(
    time_period: str = "day",
    metric_type: str = "processing_time",
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get document processing metrics.
    
    Args:
        time_period: Time period for metrics ("day", "week", "month", "all")
        metric_type: Type of metrics to retrieve
        db: Database session
        
    Returns:
        Processing metrics
    """
    try:
        # Set up date filter based on time period
        date_filter = None
        if time_period == "day":
            date_filter = datetime.utcnow() - timedelta(days=1)
        elif time_period == "week":
            date_filter = datetime.utcnow() - timedelta(weeks=1)
        elif time_period == "month":
            date_filter = datetime.utcnow() - timedelta(days=30)
        
        # Base query
        query = sa.select(
            ProcessingMetrics.metric_name,
            sa.func.avg(ProcessingMetrics.value).label("avg"),
            sa.func.min(ProcessingMetrics.value).label("min"),
            sa.func.max(ProcessingMetrics.value).label("max"),
            sa.func.count(ProcessingMetrics.id).label("count")
        ).where(
            ProcessingMetrics.metric_type == metric_type
        ).group_by(ProcessingMetrics.metric_name)
        
        if date_filter:
            query = query.where(ProcessingMetrics.timestamp >= date_filter)
        
        result = await db.execute(query)
        metrics = result.fetchall()
        
        # Format results
        formatted_metrics = {}
        for row in metrics:
            metric_name, avg_val, min_val, max_val, count = row
            formatted_metrics[metric_name] = {
                "avg": avg_val,
                "min": min_val,
                "max": max_val,
                "count": count
            }
        
        return {
            "time_period": time_period,
            "metric_type": metric_type,
            "metrics": formatted_metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting processing metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def register_routes(app):
    """Register document routes with the FastAPI app."""
    app.include_router(router)