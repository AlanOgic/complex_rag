"""
Advanced document indexing for RAG systems.

This module provides comprehensive document indexing capabilities with
multiple processing stages, quality checks, and advanced features.
"""

import logging
import os
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import uuid
import hashlib
from pathlib import Path
import json

from .chunker import DocumentChunker
from .document_quality import DocumentQualityProcessor
from .ocr_processor import OCRProcessor
from ..indexers.qdrant_indexer import QdrantIndexer, DocumentChunk
from ..connectors.base import Document

logger = logging.getLogger(__name__)


class ProcessingStage:
    """Enumeration of document processing stages."""
    
    RECEIVED = "received"
    VALIDATION = "validation"
    PREPROCESSING = "preprocessing"
    EXTRACTION = "extraction"
    OCR = "ocr"
    QUALITY_CHECK = "quality_check"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class IndexProcessor:
    """
    Advanced document indexing processor with multiple stages and quality checks.
    
    Features:
    - Multi-stage document processing pipeline
    - Document validation and quality assessment
    - Content extraction with OCR support
    - Smart chunking with metadata enrichment
    - Vectorization and indexing
    - Processing status tracking
    - Resume-from-failure capability
    """
    
    def __init__(self,
                 indexer: QdrantIndexer,
                 chunking_config: Dict[str, Any] = None,
                 quality_config: Dict[str, Any] = None,
                 ocr_config: Dict[str, Any] = None,
                 min_quality_score: float = 0.5,
                 max_attempts: int = 3,
                 enable_ocr: bool = True,
                 enable_quality_check: bool = True,
                 cache_dir: Optional[str] = None,
                 log_level: str = "INFO"):
        """
        Initialize the index processor.
        
        Args:
            indexer: Initialized QdrantIndexer for storing embeddings
            chunking_config: Configuration for document chunking
            quality_config: Configuration for document quality processor
            ocr_config: Configuration for OCR processor
            min_quality_score: Minimum document quality score to process
            max_attempts: Maximum number of processing attempts
            enable_ocr: Whether to enable OCR processing
            enable_quality_check: Whether to enable document quality checks
            cache_dir: Directory for caching
            log_level: Logging level
        """
        self.indexer = indexer
        self.max_attempts = max_attempts
        self.min_quality_score = min_quality_score
        self.enable_ocr = enable_ocr
        self.enable_quality_check = enable_quality_check
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level))
        
        # Set up cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache/indexing")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize processors
        chunking_config = chunking_config or {}
        self.chunker = DocumentChunker(
            chunk_size=chunking_config.get("chunk_size", 500),
            chunk_overlap=chunking_config.get("chunk_overlap", 100),
            strategy=chunking_config.get("strategy", "fixed"),
            respect_boundaries=chunking_config.get("respect_boundaries", True)
        )
        
        quality_config = quality_config or {}
        self.quality_processor = DocumentQualityProcessor(
            min_content_length=quality_config.get("min_content_length", 50),
            max_noise_ratio=quality_config.get("max_noise_ratio", 0.3),
            language_detection_threshold=quality_config.get("language_detection_threshold", 0.8),
            deduplication_threshold=quality_config.get("deduplication_threshold", 0.85)
        )
        
        ocr_config = ocr_config or {}
        self.ocr_processor = OCRProcessor(
            ocr_engine=ocr_config.get("ocr_engine", "tesseract"),
            languages=ocr_config.get("languages", ["eng"]),
            detect_tables=ocr_config.get("detect_tables", True),
            preserve_layout=ocr_config.get("preserve_layout", True),
            confidence_threshold=ocr_config.get("confidence_threshold", 60.0),
            temp_dir=ocr_config.get("temp_dir", str(self.cache_dir / "ocr"))
        )
        
        # Initialize OCR processor if enabled
        if self.enable_ocr:
            asyncio.create_task(self.ocr_processor.initialize())
    
    async def process_document(self, document: Document) -> Tuple[bool, Dict[str, Any]]:
        """
        Process a document through the indexing pipeline.
        
        Args:
            document: Document to process
            
        Returns:
            Tuple of (success, processing_info)
        """
        processor_id = str(uuid.uuid4())
        stage = ProcessingStage.RECEIVED
        attempts = 0
        processing_info = {
            "processor_id": processor_id,
            "document_id": document.source_id,
            "source_type": document.source_type,
            "stage": stage,
            "start_time": time.time(),
            "attempts": attempts,
            "stages_completed": [],
            "current_stage": stage,
            "metadata": document.metadata.copy() if document.metadata else {}
        }
        
        try:
            self.logger.info(f"Started processing document {document.source_id} ({processor_id})")
            
            # Document validation stage
            processing_info["current_stage"] = ProcessingStage.VALIDATION
            if not document.content and not document.file_path:
                raise ValueError("Document must have either content or file_path")
            
            processing_info["stages_completed"].append(ProcessingStage.VALIDATION)
            
            # Preprocessing stage
            processing_info["current_stage"] = ProcessingStage.PREPROCESSING
            document, preprocessing_info = await self._preprocess_document(document)
            processing_info["preprocessing_info"] = preprocessing_info
            processing_info["stages_completed"].append(ProcessingStage.PREPROCESSING)
            
            # Content extraction stage
            processing_info["current_stage"] = ProcessingStage.EXTRACTION
            document, extraction_info = await self._extract_content(document)
            processing_info["extraction_info"] = extraction_info
            processing_info["stages_completed"].append(ProcessingStage.EXTRACTION)
            
            # OCR stage (if needed)
            if self.enable_ocr and extraction_info.get("needs_ocr", False):
                processing_info["current_stage"] = ProcessingStage.OCR
                document, ocr_info = await self._apply_ocr(document)
                processing_info["ocr_info"] = ocr_info
                processing_info["stages_completed"].append(ProcessingStage.OCR)
            
            # Quality check stage
            if self.enable_quality_check:
                processing_info["current_stage"] = ProcessingStage.QUALITY_CHECK
                document, quality_info = await self._check_quality(document)
                processing_info["quality_info"] = quality_info
                
                # Check if document passes quality threshold
                if not quality_info.get("passes_threshold", True):
                    self.logger.warning(
                        f"Document {document.source_id} failed quality check with score "
                        f"{quality_info.get('quality_score', 0)}"
                    )
                    processing_info["status"] = "failed"
                    processing_info["error"] = "Document failed quality check"
                    processing_info["current_stage"] = ProcessingStage.FAILED
                    return False, processing_info
                
                processing_info["stages_completed"].append(ProcessingStage.QUALITY_CHECK)
            
            # Chunking stage
            processing_info["current_stage"] = ProcessingStage.CHUNKING
            chunks, chunking_info = await self._chunk_document(document)
            processing_info["chunking_info"] = chunking_info
            processing_info["stages_completed"].append(ProcessingStage.CHUNKING)
            
            # Embedding and Indexing stage
            processing_info["current_stage"] = ProcessingStage.EMBEDDING
            success, embedding_info = await self._embed_and_index_chunks(chunks)
            processing_info["embedding_info"] = embedding_info
            
            if not success:
                self.logger.error(f"Failed to embed and index document {document.source_id}")
                processing_info["status"] = "failed"
                processing_info["error"] = "Failed to embed and index document"
                processing_info["current_stage"] = ProcessingStage.FAILED
                return False, processing_info
            
            processing_info["stages_completed"].append(ProcessingStage.EMBEDDING)
            processing_info["stages_completed"].append(ProcessingStage.INDEXING)
            
            # Final metadata and completion
            processing_info["current_stage"] = ProcessingStage.COMPLETED
            processing_info["status"] = "success"
            processing_info["end_time"] = time.time()
            processing_info["processing_time"] = processing_info["end_time"] - processing_info["start_time"]
            processing_info["chunk_count"] = len(chunks)
            
            # Log completion
            self.logger.info(
                f"Completed processing document {document.source_id} ({processor_id}) "
                f"with {len(chunks)} chunks in {processing_info['processing_time']:.2f}s"
            )
            
            return True, processing_info
            
        except Exception as e:
            self.logger.error(f"Error processing document {document.source_id}: {str(e)}")
            
            # Increment attempts
            attempts += 1
            processing_info["attempts"] = attempts
            
            # Check if we should retry
            if attempts < self.max_attempts:
                self.logger.info(f"Retrying document {document.source_id} (attempt {attempts + 1}/{self.max_attempts})")
                return await self.process_document(document)
            
            # Record failure
            processing_info["status"] = "failed"
            processing_info["error"] = str(e)
            processing_info["current_stage"] = ProcessingStage.FAILED
            processing_info["end_time"] = time.time()
            processing_info["processing_time"] = processing_info["end_time"] - processing_info["start_time"]
            
            return False, processing_info
    
    async def _preprocess_document(self, document: Document) -> Tuple[Document, Dict[str, Any]]:
        """
        Preprocess document before extraction.
        
        Args:
            document: Document to preprocess
            
        Returns:
            Tuple of (preprocessed_document, preprocessing_info)
        """
        preprocessing_info = {
            "original_metadata": document.metadata.copy() if document.metadata else {}
        }
        
        try:
            # Generate a deterministic document ID if not present
            if not document.source_id:
                content_hash = hashlib.md5(document.content[:1000].encode()).hexdigest() if document.content else ""
                path_hash = hashlib.md5(str(document.file_path).encode()).hexdigest() if document.file_path else ""
                document.source_id = f"doc_{content_hash or path_hash}"
            
            # Extract file type from metadata or file path
            file_type = None
            
            if document.metadata and "file_type" in document.metadata:
                file_type = document.metadata["file_type"]
            elif document.file_path:
                file_type = os.path.splitext(document.file_path)[1].lower().lstrip(".")
            
            if file_type:
                if not document.metadata:
                    document.metadata = {}
                document.metadata["file_type"] = file_type
            
            # Check if document needs OCR
            needs_ocr = False
            
            if file_type in ["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif"]:
                needs_ocr = True
            
            preprocessing_info["file_type"] = file_type
            preprocessing_info["needs_ocr"] = needs_ocr
            
            return document, preprocessing_info
            
        except Exception as e:
            self.logger.error(f"Error preprocessing document: {str(e)}")
            preprocessing_info["error"] = str(e)
            return document, preprocessing_info
    
    async def _extract_content(self, document: Document) -> Tuple[Document, Dict[str, Any]]:
        """
        Extract content from document.
        
        Args:
            document: Document to extract content from
            
        Returns:
            Tuple of (document_with_content, extraction_info)
        """
        extraction_info = {
            "successful": False,
            "needs_ocr": False
        }
        
        try:
            # If document already has content, we're done
            if document.content:
                extraction_info["successful"] = True
                extraction_info["source"] = "provided"
                extraction_info["content_length"] = len(document.content)
                return document, extraction_info
            
            # If no file path, we can't extract content
            if not document.file_path:
                raise ValueError("Document has no content or file path")
            
            # Get file type
            file_type = None
            if document.metadata and "file_type" in document.metadata:
                file_type = document.metadata["file_type"]
            else:
                file_type = os.path.splitext(document.file_path)[1].lower().lstrip(".")
            
            # Text-based formats
            if file_type in ["txt", "md", "json", "csv", "xml", "html", "htm"]:
                async with aiofiles.open(document.file_path, "r", encoding="utf-8", errors="replace") as f:
                    document.content = await f.read()
                
                extraction_info["successful"] = True
                extraction_info["source"] = "file_read"
                extraction_info["content_length"] = len(document.content)
                
            # Potentially OCR-needed formats
            elif file_type in ["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif"]:
                # Try direct PDF text extraction first
                if file_type == "pdf":
                    try:
                        from PyPDF2 import PdfReader
                        
                        reader = await asyncio.to_thread(PdfReader, document.file_path)
                        content = ""
                        
                        for page in reader.pages:
                            page_text = await asyncio.to_thread(page.extract_text)
                            if page_text:
                                content += page_text + "\n\n"
                        
                        if content and len(content.strip()) > 100:
                            document.content = content
                            extraction_info["successful"] = True
                            extraction_info["source"] = "pdf_extract"
                            extraction_info["content_length"] = len(content)
                            return document, extraction_info
                    except Exception as e:
                        self.logger.debug(f"PDF extraction failed: {str(e)}")
                
                # Mark for OCR
                extraction_info["needs_ocr"] = True
                extraction_info["file_type"] = file_type
                
            # Office documents
            elif file_type in ["docx", "doc", "pptx", "ppt", "xlsx", "xls"]:
                try:
                    import textract
                    
                    content = await asyncio.to_thread(
                        textract.process,
                        document.file_path,
                        encoding='utf-8'
                    )
                    
                    if isinstance(content, bytes):
                        content = content.decode('utf-8', errors='replace')
                    
                    document.content = content
                    extraction_info["successful"] = True
                    extraction_info["source"] = "textract"
                    extraction_info["content_length"] = len(content)
                    
                except Exception as e:
                    self.logger.error(f"Textract extraction failed: {str(e)}")
                    extraction_info["needs_ocr"] = True
                    extraction_info["file_type"] = file_type
            
            else:
                self.logger.warning(f"Unsupported file type: {file_type}")
                extraction_info["error"] = f"Unsupported file type: {file_type}"
            
            return document, extraction_info
            
        except Exception as e:
            self.logger.error(f"Error extracting content: {str(e)}")
            extraction_info["error"] = str(e)
            return document, extraction_info
    
    async def _apply_ocr(self, document: Document) -> Tuple[Document, Dict[str, Any]]:
        """
        Apply OCR to extract text from images or PDFs.
        
        Args:
            document: Document to process with OCR
            
        Returns:
            Tuple of (document_with_content, ocr_info)
        """
        ocr_info = {
            "successful": False
        }
        
        if not self.enable_ocr:
            ocr_info["error"] = "OCR is disabled"
            return document, ocr_info
        
        if not document.file_path:
            ocr_info["error"] = "Document has no file path for OCR"
            return document, ocr_info
        
        try:
            file_type = os.path.splitext(document.file_path)[1].lower().lstrip(".")
            
            # Process based on file type
            if file_type == "pdf":
                result = await self.ocr_processor.process_pdf(document.file_path)
            else:
                result = await self.ocr_processor.process_image(document.file_path)
            
            if "error" in result:
                ocr_info["error"] = result["error"]
                return document, ocr_info
            
            # Update document with OCR text
            document.content = result["text"]
            
            # Update metadata with OCR info
            if not document.metadata:
                document.metadata = {}
            
            document.metadata["ocr_applied"] = True
            document.metadata["ocr_engine"] = result.get("source", self.ocr_processor.ocr_engine)
            document.metadata["ocr_confidence"] = result.get("confidence", 0.0)
            
            if "pages" in result:
                document.metadata["page_count"] = result["pages"]
            
            ocr_info.update(result)
            ocr_info["successful"] = True
            ocr_info["content_length"] = len(document.content)
            
            return document, ocr_info
            
        except Exception as e:
            self.logger.error(f"Error applying OCR: {str(e)}")
            ocr_info["error"] = str(e)
            return document, ocr_info
    
    async def _check_quality(self, document: Document) -> Tuple[Document, Dict[str, Any]]:
        """
        Check and enhance document quality.
        
        Args:
            document: Document to check
            
        Returns:
            Tuple of (enhanced_document, quality_info)
        """
        quality_info = {}
        
        if not self.enable_quality_check:
            quality_info["quality_check_disabled"] = True
            quality_info["passes_threshold"] = True
            return document, quality_info
        
        if not document.content:
            quality_info["error"] = "Document has no content for quality check"
            quality_info["passes_threshold"] = False
            return document, quality_info
        
        try:
            # Analyze document quality
            quality_analysis = self.quality_processor.analyze_document_quality(document.content)
            quality_info.update(quality_analysis)
            
            # Check if document passes quality threshold
            quality_info["passes_threshold"] = quality_analysis["quality_score"] >= self.min_quality_score
            
            # If document has quality issues but is still above threshold, try to enhance
            if quality_info["passes_threshold"] and quality_analysis["quality_score"] < 0.7:
                enhanced_content, enhancement_info = self.quality_processor.enhance_document(document.content)
                
                # Update document with enhanced content if improved
                if enhancement_info["improved"]:
                    document.content = enhanced_content
                    quality_info["enhancement_applied"] = True
                    quality_info["enhancement_info"] = enhancement_info
                else:
                    quality_info["enhancement_applied"] = False
            
            # Update document metadata with quality info
            if not document.metadata:
                document.metadata = {}
            
            document.metadata["quality_score"] = quality_analysis["quality_score"]
            document.metadata["language"] = quality_analysis["language"]
            
            if quality_analysis["has_structure"]:
                document.metadata["has_structure"] = True
            
            return document, quality_info
            
        except Exception as e:
            self.logger.error(f"Error checking document quality: {str(e)}")
            quality_info["error"] = str(e)
            quality_info["passes_threshold"] = True  # Don't block processing on quality check errors
            return document, quality_info
    
    async def _chunk_document(self, document: Document) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """
        Chunk document into smaller pieces for embedding.
        
        Args:
            document: Document to chunk
            
        Returns:
            Tuple of (document_chunks, chunking_info)
        """
        chunking_info = {
            "successful": False
        }
        
        if not document.content:
            chunking_info["error"] = "Document has no content for chunking"
            return [], chunking_info
        
        try:
            # Apply chunking
            chunks = self.chunker.chunk_document(document)
            
            chunking_info["successful"] = True
            chunking_info["chunk_count"] = len(chunks)
            
            # Calculate average chunk length
            if chunks:
                avg_chunk_length = sum(len(chunk.content) for chunk in chunks) / len(chunks)
                chunking_info["avg_chunk_length"] = avg_chunk_length
            
            return chunks, chunking_info
            
        except Exception as e:
            self.logger.error(f"Error chunking document: {str(e)}")
            chunking_info["error"] = str(e)
            return [], chunking_info
    
    async def _embed_and_index_chunks(self, chunks: List[DocumentChunk]) -> Tuple[bool, Dict[str, Any]]:
        """
        Generate embeddings and index document chunks.
        
        Args:
            chunks: Document chunks to embed and index
            
        Returns:
            Tuple of (success, embedding_info)
        """
        embedding_info = {
            "successful": False
        }
        
        if not chunks:
            embedding_info["error"] = "No chunks to embed"
            return False, embedding_info
        
        try:
            # Add chunks to indexer
            success = await self.indexer.add_chunks(chunks)
            
            if not success:
                embedding_info["error"] = "Failed to add chunks to indexer"
                return False, embedding_info
            
            embedding_info["successful"] = True
            embedding_info["chunks_indexed"] = len(chunks)
            embedding_info["embedding_provider"] = self.indexer.embedding_provider
            embedding_info["embedding_model"] = self.indexer.embedding_model_name
            
            return True, embedding_info
            
        except Exception as e:
            self.logger.error(f"Error embedding and indexing chunks: {str(e)}")
            embedding_info["error"] = str(e)
            return False, embedding_info


# Import here for asynchronous file operations
import aiofiles