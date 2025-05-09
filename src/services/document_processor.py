"""
Document processor service for asynchronous document processing.

This service monitors the uploads directory and MinIO buckets for new documents,
processes them, and indexes them in the RAG system.
"""

import os
import time
import asyncio
import logging
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import traceback
import shutil
import hashlib

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
import aiofiles
import aiofiles.os

# Local imports
from config.settings import Config
from ..connectors import FileConnector
from ..processors.chunker import DocumentChunker
from ..indexers.qdrant_indexer import QdrantIndexer

# Set up logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# SQLAlchemy models
Base = declarative_base()

class Document(Base):
    """Database model for document tracking."""
    
    __tablename__ = "documents"
    
    id = sa.Column(sa.String, primary_key=True)
    filename = sa.Column(sa.String, nullable=False)
    original_path = sa.Column(sa.String, nullable=True)
    file_type = sa.Column(sa.String, nullable=False)
    file_size = sa.Column(sa.Integer, nullable=False)
    stored_path = sa.Column(sa.String, nullable=True)
    minio_bucket = sa.Column(sa.String, nullable=True)
    minio_object = sa.Column(sa.String, nullable=True)
    chunk_count = sa.Column(sa.Integer, nullable=True)
    status = sa.Column(sa.String, nullable=False, default="pending")
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    updated_at = sa.Column(sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = sa.Column(sa.JSON, nullable=True)
    error = sa.Column(sa.String, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "chunk_count": self.chunk_count,
            "metadata": self.metadata
        }


class DocumentProcessor:
    """
    Service for processing documents asynchronously.
    
    Features:
    - Monitors uploads directory and MinIO buckets for new documents
    - Processes documents using appropriate connectors
    - Chunks and indexes documents in the RAG system
    - Tracks document processing status in the database
    """
    
    def __init__(self):
        """Initialize the document processor."""
        self.poll_interval = int(os.environ.get("PROCESSOR_POLL_INTERVAL", "5"))
        self.batch_size = int(os.environ.get("PROCESSOR_BATCH_SIZE", "10"))
        self.max_file_size_mb = float(os.environ.get("MAX_DOCUMENT_SIZE_MB", str(Config.MAX_DOCUMENT_SIZE_MB)))
        self.upload_dir = os.environ.get("UPLOAD_DIR", "/app/uploads")
        self.data_dir = os.environ.get("DATA_DIR", "/app/data")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        
        # MinIO configuration
        self.minio_endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
        self.minio_access_key = os.environ.get("MINIO_ACCESS_KEY", "minio_user")
        self.minio_secret_key = os.environ.get("MINIO_SECRET_KEY", "minio_password")
        self.minio_use_ssl = os.environ.get("MINIO_USE_SSL", "false").lower() == "true"
        self.minio_buckets = ["documents", "uploads"]
        
        # Database configuration
        self.db_url = Config.get_database_uri().replace("postgresql", "postgresql+asyncpg")
        
        # Create directories if they don't exist
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize components
        self.db_engine = None
        self.db_session = None
        self.minio_client = None
        self.chunker = None
        self.indexer = None
        self.file_connector = None
        self.initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize the document processor.
        
        Returns:
            True if successful
        """
        try:
            # Initialize database
            self.db_engine = create_async_engine(self.db_url)
            async with self.db_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self.db_session = sessionmaker(
                self.db_engine, class_=AsyncSession, expire_on_commit=False
            )
            
            # Initialize MinIO client
            try:
                from minio import Minio
                self.minio_client = Minio(
                    self.minio_endpoint,
                    access_key=self.minio_access_key,
                    secret_key=self.minio_secret_key,
                    secure=self.minio_use_ssl,
                )
                
                # Ensure buckets exist
                for bucket in self.minio_buckets:
                    if not await asyncio.to_thread(
                        self.minio_client.bucket_exists, bucket
                    ):
                        await asyncio.to_thread(
                            self.minio_client.make_bucket, bucket
                        )
                        logger.info(f"Created MinIO bucket: {bucket}")
            except ImportError:
                logger.warning("MinIO client not available. MinIO support disabled.")
                self.minio_client = None
            
            # Initialize chunker
            self.chunker = DocumentChunker(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
                strategy=Config.CHUNKING_STRATEGY,
                respect_boundaries=True
            )
            
            # Initialize indexer
            self.indexer = QdrantIndexer(
                url=os.environ.get("QDRANT_URL", Config.QDRANT_URL),
                collection_name=os.environ.get("QDRANT_COLLECTION_NAME", Config.QDRANT_COLLECTION_NAME),
                api_key=os.environ.get("QDRANT_API_KEY", Config.QDRANT_API_KEY),
                embedding_provider=os.environ.get("EMBEDDING_PROVIDER", Config.EMBEDDING_PROVIDER),
                embedding_model=os.environ.get("EMBEDDING_MODEL", Config.EMBEDDING_MODEL),
                embedding_dimension=int(os.environ.get("EMBEDDING_DIMENSION", str(Config.EMBEDDING_DIMENSION))),
                distance_metric=os.environ.get("VECTOR_DISTANCE_METRIC", Config.VECTOR_DISTANCE_METRIC),
                embedding_api_key=os.environ.get("OPENAI_API_KEY", Config.OPENAI_API_KEY),
                cache_dir=os.environ.get("EMBEDDING_CACHE_DIR", Config.EMBEDDING_CACHE_DIR),
            )
            
            if not await self.indexer.initialize():
                logger.error("Failed to initialize indexer")
                return False
            
            # Initialize file connector
            self.file_connector = FileConnector({
                "base_dirs": [self.upload_dir, self.processed_dir],
                "recursive": True,
                "max_file_size_mb": self.max_file_size_mb,
            })
            
            if not await self.file_connector.connect():
                logger.error("Failed to initialize file connector")
                return False
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing document processor: {e}")
            return False
    
    async def check_for_new_files(self) -> List[Path]:
        """
        Check for new files in the upload directory.
        
        Returns:
            List of new file paths
        """
        try:
            # Check local upload directory
            new_files = []
            
            for root, _, files in os.walk(self.upload_dir):
                for filename in files:
                    # Skip hidden files and temp files
                    if filename.startswith(".") or filename.endswith(".tmp"):
                        continue
                    
                    file_path = Path(os.path.join(root, filename))
                    
                    # Check if already in database
                    async with self.db_session() as session:
                        file_id = self._generate_file_id(file_path)
                        query = sa.select(Document).where(Document.id == file_id)
                        result = await session.execute(query)
                        doc = result.scalar_one_or_none()
                        
                        if not doc:
                            new_files.append(file_path)
            
            return new_files
            
        except Exception as e:
            logger.error(f"Error checking for new files: {e}")
            return []
    
    async def check_minio_for_new_files(self) -> List[Dict[str, Any]]:
        """
        Check for new files in MinIO uploads bucket.
        
        Returns:
            List of new file info dicts with bucket, object name
        """
        if not self.minio_client:
            return []
        
        try:
            # Get objects from uploads bucket
            objects = await asyncio.to_thread(
                self.minio_client.list_objects, "uploads", recursive=True
            )
            
            new_files = []
            
            for obj in objects:
                # Skip directories
                if obj.object_name.endswith('/'):
                    continue
                
                # Generate ID
                file_id = f"minio_{obj.bucket_name}_{obj.object_name}"
                file_id = hashlib.md5(file_id.encode()).hexdigest()
                
                # Check if already in database
                async with self.db_session() as session:
                    query = sa.select(Document).where(Document.id == file_id)
                    result = await session.execute(query)
                    doc = result.scalar_one_or_none()
                    
                    if not doc:
                        new_files.append({
                            "id": file_id,
                            "bucket": obj.bucket_name,
                            "object_name": obj.object_name,
                            "size": obj.size,
                            "last_modified": obj.last_modified
                        })
            
            return new_files
            
        except Exception as e:
            logger.error(f"Error checking MinIO for new files: {e}")
            return []
    
    async def get_pending_documents(self, limit: int) -> List[Document]:
        """
        Get pending documents from the database.
        
        Args:
            limit: Maximum number of documents to return
            
        Returns:
            List of pending documents
        """
        try:
            async with self.db_session() as session:
                query = sa.select(Document).where(
                    Document.status == "pending"
                ).order_by(Document.created_at).limit(limit)
                
                result = await session.execute(query)
                return list(result.scalars().all())
                
        except Exception as e:
            logger.error(f"Error getting pending documents: {e}")
            return []
    
    def _generate_file_id(self, file_path: Path) -> str:
        """
        Generate a unique ID for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Unique ID string
        """
        file_stat = file_path.stat()
        unique_string = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    async def register_new_files(self, file_paths: List[Path]) -> List[str]:
        """
        Register new files in the database.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of document IDs
        """
        try:
            registered_ids = []
            
            async with self.db_session() as session:
                for file_path in file_paths:
                    try:
                        # Generate ID
                        file_id = self._generate_file_id(file_path)
                        
                        # Get file info
                        filename = file_path.name
                        file_type = file_path.suffix.lower().lstrip(".")
                        file_size = file_path.stat().st_size
                        
                        # Create document
                        doc = Document(
                            id=file_id,
                            filename=filename,
                            original_path=str(file_path),
                            file_type=file_type,
                            file_size=file_size,
                            status="pending",
                            metadata={
                                "source": "local_upload"
                            }
                        )
                        
                        session.add(doc)
                        registered_ids.append(file_id)
                        
                    except Exception as e:
                        logger.error(f"Error registering file {file_path}: {e}")
                
                await session.commit()
            
            return registered_ids
            
        except Exception as e:
            logger.error(f"Error registering new files: {e}")
            return []
    
    async def register_minio_files(self, file_infos: List[Dict[str, Any]]) -> List[str]:
        """
        Register new MinIO files in the database.
        
        Args:
            file_infos: List of file info dicts
            
        Returns:
            List of document IDs
        """
        try:
            registered_ids = []
            
            async with self.db_session() as session:
                for file_info in file_infos:
                    try:
                        file_id = file_info["id"]
                        object_name = file_info["object_name"]
                        filename = os.path.basename(object_name)
                        file_type = os.path.splitext(filename)[1].lower().lstrip(".")
                        
                        # Create document
                        doc = Document(
                            id=file_id,
                            filename=filename,
                            file_type=file_type,
                            file_size=file_info["size"],
                            minio_bucket=file_info["bucket"],
                            minio_object=object_name,
                            status="pending",
                            metadata={
                                "source": "minio_upload",
                                "last_modified": file_info["last_modified"].isoformat()
                                if hasattr(file_info["last_modified"], "isoformat") else str(file_info["last_modified"])
                            }
                        )
                        
                        session.add(doc)
                        registered_ids.append(file_id)
                        
                    except Exception as e:
                        logger.error(f"Error registering MinIO file {file_info}: {e}")
                
                await session.commit()
            
            return registered_ids
            
        except Exception as e:
            logger.error(f"Error registering MinIO files: {e}")
            return []
    
    async def process_document(self, doc: Document) -> bool:
        """
        Process a document.
        
        Args:
            doc: Document to process
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Processing document {doc.id}: {doc.filename}")
            
            # Update status to processing
            async with self.db_session() as session:
                query = sa.select(Document).where(Document.id == doc.id).with_for_update()
                result = await session.execute(query)
                db_doc = result.scalar_one_or_none()
                
                if not db_doc:
                    logger.error(f"Document {doc.id} not found")
                    return False
                
                db_doc.status = "processing"
                await session.commit()
            
            # Create temp directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = None
                
                # Get file from source
                if doc.original_path:
                    # Local file
                    file_path = Path(doc.original_path)
                elif doc.minio_bucket and doc.minio_object and self.minio_client:
                    # MinIO file
                    temp_file = os.path.join(temp_dir, doc.filename)
                    
                    await asyncio.to_thread(
                        self.minio_client.fget_object,
                        doc.minio_bucket,
                        doc.minio_object,
                        temp_file
                    )
                    
                    file_path = Path(temp_file)
                
                if not file_path or not file_path.exists():
                    raise FileNotFoundError(f"Could not access file: {doc.filename}")
                
                # Process file with FileConnector
                document = await self._process_file_with_connector(file_path, doc)
                
                if not document:
                    raise ValueError(f"Failed to process file: {doc.filename}")
                
                # Chunk document
                chunks = self.chunker.chunk_document(document)
                
                if not chunks:
                    raise ValueError(f"No chunks generated for document: {doc.filename}")
                
                # Index chunks
                if not await self.indexer.add_chunks(chunks):
                    raise ValueError(f"Failed to index chunks for document: {doc.filename}")
                
                # Store processed document
                processed_path = os.path.join(self.processed_dir, doc.filename)
                
                if doc.original_path:
                    # Copy local file
                    shutil.copy2(doc.original_path, processed_path)
                
                # If MinIO is available, store in documents bucket
                if self.minio_client:
                    try:
                        if doc.original_path:
                            # Upload local file to MinIO
                            await asyncio.to_thread(
                                self.minio_client.fput_object,
                                "documents",
                                doc.filename,
                                doc.original_path
                            )
                        elif doc.minio_bucket and doc.minio_object:
                            # Copy within MinIO
                            source = f"{doc.minio_bucket}/{doc.minio_object}"
                            await asyncio.to_thread(
                                self.minio_client.copy_object,
                                "documents",
                                doc.filename,
                                source
                            )
                    except Exception as e:
                        logger.error(f"Error storing document in MinIO: {e}")
                
                # Update document status
                async with self.db_session() as session:
                    query = sa.select(Document).where(Document.id == doc.id).with_for_update()
                    result = await session.execute(query)
                    db_doc = result.scalar_one_or_none()
                    
                    if db_doc:
                        db_doc.status = "indexed"
                        db_doc.stored_path = processed_path
                        db_doc.chunk_count = len(chunks)
                        
                        if self.minio_client:
                            db_doc.minio_bucket = "documents"
                            db_doc.minio_object = doc.filename
                        
                        db_doc.updated_at = datetime.utcnow()
                        await session.commit()
                
                logger.info(f"Document {doc.id} processed successfully with {len(chunks)} chunks")
                return True
                
        except Exception as e:
            error_message = f"Error processing document {doc.id}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_message)
            
            # Update document status to error
            try:
                async with self.db_session() as session:
                    query = sa.select(Document).where(Document.id == doc.id).with_for_update()
                    result = await session.execute(query)
                    db_doc = result.scalar_one_or_none()
                    
                    if db_doc:
                        db_doc.status = "error"
                        db_doc.error = str(e)
                        db_doc.updated_at = datetime.utcnow()
                        await session.commit()
            except Exception as db_error:
                logger.error(f"Error updating document status: {db_error}")
            
            return False
    
    async def _process_file_with_connector(self, file_path: Path, doc: Document) -> Optional[Any]:
        """
        Process a file using the FileConnector.
        
        Args:
            file_path: Path to the file
            doc: Document record
            
        Returns:
            Processed document or None if failed
        """
        try:
            # Create custom file connector for this specific file
            file_config = {
                "base_dirs": [str(file_path.parent)],
                "include_patterns": [file_path.name],
                "recursive": False,
                "extract_metadata": True
            }
            
            file_connector = FileConnector(file_config)
            
            if not await file_connector.connect():
                raise RuntimeError(f"Could not connect to file: {file_path}")
            
            # Get document
            documents = await file_connector.get_documents(limit=1)
            
            if not documents or len(documents) == 0:
                raise ValueError(f"No documents found for file: {file_path}")
            
            return documents[0]
            
        except Exception as e:
            logger.error(f"Error processing file with connector: {e}")
            return None
    
    async def run(self) -> None:
        """Run the document processor loop."""
        if not self.initialized:
            if not await self.initialize():
                logger.error("Failed to initialize document processor")
                return
        
        try:
            while True:
                try:
                    # Check for new files in upload directory
                    new_files = await self.check_for_new_files()
                    
                    if new_files:
                        logger.info(f"Found {len(new_files)} new files in upload directory")
                        await self.register_new_files(new_files)
                    
                    # Check for new files in MinIO
                    if self.minio_client:
                        new_minio_files = await self.check_minio_for_new_files()
                        
                        if new_minio_files:
                            logger.info(f"Found {len(new_minio_files)} new files in MinIO")
                            await self.register_minio_files(new_minio_files)
                    
                    # Process pending documents
                    pending_docs = await self.get_pending_documents(self.batch_size)
                    
                    if pending_docs:
                        logger.info(f"Processing {len(pending_docs)} pending documents")
                        
                        for doc in pending_docs:
                            await self.process_document(doc)
                    
                except Exception as e:
                    logger.error(f"Error in document processor loop: {e}")
                
                # Wait for next cycle
                await asyncio.sleep(self.poll_interval)
                
        except KeyboardInterrupt:
            logger.info("Document processor stopped")
        except Exception as e:
            logger.error(f"Fatal error in document processor: {e}")
        finally:
            # Close connections
            if self.indexer:
                await self.indexer.close()
            
            if self.db_engine:
                await self.db_engine.dispose()


async def main():
    """Main entry point."""
    processor = DocumentProcessor()
    await processor.run()


if __name__ == "__main__":
    asyncio.run(main())