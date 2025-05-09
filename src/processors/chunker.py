"""
Document chunking utilities.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union, Set
import uuid
import hashlib

from ..connectors.base import Document
from ..indexers.qdrant_indexer import DocumentChunk

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Handles document chunking with various strategies.
    
    Features:
    - Multiple chunking strategies (fixed size, semantic, etc.)
    - Overlap handling
    - Metadata management
    - Special content handling (tables, code blocks, etc.)
    """
    
    def __init__(self,
                 chunk_size: int = 500,
                 chunk_overlap: int = 100,
                 strategy: str = "fixed",
                 respect_boundaries: bool = True):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Target size of each chunk (in characters)
            chunk_overlap: Overlap between consecutive chunks
            strategy: Chunking strategy ("fixed", "sentence", "paragraph", "semantic")
            respect_boundaries: Whether to respect paragraph and sentence boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.respect_boundaries = respect_boundaries
    
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """
        Split a document into chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        if not document.content:
            return []
        
        # Choose chunking strategy
        if self.strategy == "fixed":
            chunks = self._fixed_size_chunking(document.content)
        elif self.strategy == "sentence":
            chunks = self._sentence_chunking(document.content)
        elif self.strategy == "paragraph":
            chunks = self._paragraph_chunking(document.content)
        elif self.strategy == "semantic":
            chunks = self._semantic_chunking(document.content)
        else:
            # Default to fixed size
            chunks = self._fixed_size_chunking(document.content)
        
        # Create DocumentChunk objects with metadata
        result = []
        base_metadata = document.metadata or {}
        
        for i, chunk_text in enumerate(chunks):
            # Create a deterministic chunk ID
            content_hash = hashlib.md5((document.source_id + chunk_text[:100]).encode()).hexdigest()
            source_id_cleaned = document.source_id.replace('/', '_')
            source_id_cleaned = source_id_cleaned.replace('\\', '_')
            chunk_id = f"{source_id_cleaned}_{content_hash}"
            
            # Create metadata with chunk info
            metadata = base_metadata.copy()
            metadata.update({
                "chunk_index": i,
                "chunk_count": len(chunks)
            })
            
            # Add special content flags
            metadata.update(self._detect_special_content(chunk_text))
            
            # Create the chunk
            chunk = DocumentChunk(
                content=chunk_text,
                metadata=metadata,
                source_type=document.source_type,
                source_id=document.source_id,
                chunk_id=chunk_id
            )
            
            result.append(chunk)
        
        return result
    
    def _fixed_size_chunking(self, text: str) -> List[str]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        if self.respect_boundaries:
            # Split on paragraphs first
            paragraphs = re.split(r'\n\s*\n', text)
            current_chunk = []
            current_size = 0
            
            for paragraph in paragraphs:
                # Skip empty paragraphs
                if not paragraph.strip():
                    continue
                
                paragraph_size = len(paragraph)
                
                # If adding this paragraph exceeds chunk size, finish the current chunk
                if current_size + paragraph_size > self.chunk_size and current_size > 0:
                    chunks.append('\n\n'.join(current_chunk))
                    
                    # Start a new chunk with overlap
                    overlap_size = 0
                    overlap_chunks = []
                    
                    # Add previous paragraphs until we reach the desired overlap
                    for prev_paragraph in reversed(current_chunk):
                        if overlap_size + len(prev_paragraph) <= self.chunk_overlap:
                            overlap_chunks.insert(0, prev_paragraph)
                            overlap_size += len(prev_paragraph)
                        else:
                            break
                    
                    current_chunk = overlap_chunks
                    current_size = overlap_size
                
                # Add the paragraph to the current chunk
                current_chunk.append(paragraph)
                current_size += paragraph_size
            
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
        else:
            # Simple fixed-size chunking without respecting boundaries
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                if chunk:
                    chunks.append(chunk)
        
        return chunks
    
    def _sentence_chunking(self, text: str) -> List[str]:
        """
        Split text into chunks at sentence boundaries.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        # Split into sentences
        # This regex is a simplistic sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds chunk size, finish current chunk
            if current_size + sentence_size > self.chunk_size and current_size > 0:
                chunks.append(' '.join(current_chunk))
                
                # Start a new chunk with overlap
                overlap_size = 0
                overlap_sentences = []
                
                # Add previous sentences until we reach the desired overlap
                for prev_sentence in reversed(current_chunk):
                    if overlap_size + len(prev_sentence) <= self.chunk_overlap:
                        overlap_sentences.insert(0, prev_sentence)
                        overlap_size += len(prev_sentence) + 1  # +1 for the space
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for the space
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _paragraph_chunking(self, text: str) -> List[str]:
        """
        Split text into chunks at paragraph boundaries.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Split on paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
            
            paragraph_size = len(paragraph)
            
            # If this paragraph alone exceeds chunk size, split it further
            if paragraph_size > self.chunk_size:
                # If we have accumulated content, add it as a chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split the large paragraph using sentence chunking
                paragraph_chunks = self._sentence_chunking(paragraph)
                chunks.extend(paragraph_chunks)
                continue
            
            # If adding this paragraph exceeds chunk size, finish the current chunk
            if current_size + paragraph_size > self.chunk_size and current_size > 0:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Add the paragraph to the current chunk
            current_chunk.append(paragraph)
            current_size += paragraph_size + 4  # +4 for the '\n\n' separator
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _semantic_chunking(self, text: str) -> List[str]:
        """
        Split text into chunks based on semantic boundaries.
        
        This is a placeholder - true semantic chunking would require ML models.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # For now, fall back to paragraph chunking
        # In a real implementation, this would use an ML model to detect topic changes
        return self._paragraph_chunking(text)
    
    def _detect_special_content(self, text: str) -> Dict[str, bool]:
        """
        Detect special content types in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with flags for special content
        """
        result = {
            "has_code": False,
            "has_table": False,
            "has_list": False,
            "has_urls": False,
            "has_json": False,
            "has_xml": False
        }
        
        # Detect code blocks (markdown style or indented)
        if re.search(r'```[\s\S]*?```', text) or re.search(r'    \w+', text):
            result["has_code"] = True
        
        # Detect tables (markdown, ascii, or pipe separated)
        if re.search(r'\|[\s\S]*?\|', text) or re.search(r'[+-]+\s*[+-]+', text):
            result["has_table"] = True
        
        # Detect lists
        if re.search(r'^\s*[\*\-+]\s', text, re.MULTILINE) or re.search(r'^\s*\d+\.\s', text, re.MULTILINE):
            result["has_list"] = True
        
        # Detect URLs
        if re.search(r'https?://\S+', text):
            result["has_urls"] = True
        
        # Detect JSON-like content
        if re.search(r'{\s*"\w+"\s*:', text) or re.search(r'{\s*\w+\s*:', text):
            result["has_json"] = True
        
        # Detect XML-like content
        if re.search(r'<\w+>[\s\S]*?</\w+>', text):
            result["has_xml"] = True
        
        return result