"""
Core RAG (Retrieval-Augmented Generation) system implementation.
"""

import logging
import os
import json
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Set, Tuple
import uuid
import hashlib

import numpy as np
from pydantic import BaseModel, Field

from .connectors import ConnectorFactory, BaseConnector, Document
from .processors.chunker import DocumentChunker
from .indexers.qdrant_indexer import QdrantIndexer
from .retrievers.hybrid_retriever import HybridRetriever
from .models.llm_manager import LLMManager
from config.settings import Config

logger = logging.getLogger(__name__)


class RAGQuery(BaseModel):
    """Representation of a query to the RAG system."""
    
    query: str
    sources: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    use_reranking: Optional[bool] = None
    use_multi_query: Optional[bool] = None
    include_sources: Optional[bool] = True
    citation_format: Optional[str] = "inline"
    stream: Optional[bool] = False


class RAGChunk(BaseModel):
    """Representation of a chunk returned by the RAG system."""
    
    content: str
    metadata: Dict[str, Any]
    score: float = 0.0
    source_type: str
    source_id: str
    chunk_id: str
    

class RAGResponse(BaseModel):
    """Response from the RAG system."""
    
    query: str
    answer: str
    chunks: List[RAGChunk] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComplexRAGSystem:
    """
    Complex RAG system integrating multiple data sources.
    
    Features:
    - Multi-source retrieval (emails, Odoo, Mattermost, files, SQL databases)
    - Advanced document processing including chunking and embedding
    - Hybrid retrieval methods (vector + keyword/BM25)
    - Reranking and multi-query expansion
    - Source attribution and citation
    - Query routing and weighting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAG system.
        
        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self.connectors: Dict[str, BaseConnector] = {}
        self.indexer = None
        self.chunker = None
        self.retriever = None
        self.llm_manager = None
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize all components of the RAG system.
        
        Returns:
            True if initialization was successful
        """
        try:
            # Create document chunker
            self.chunker = DocumentChunker(
                chunk_size=self.config.get("chunk_size", Config.CHUNK_SIZE),
                chunk_overlap=self.config.get("chunk_overlap", Config.CHUNK_OVERLAP)
            )
            
            # Create indexer (Qdrant)
            self.indexer = QdrantIndexer(
                url=self.config.get("qdrant_url", Config.QDRANT_URL),
                collection_name=self.config.get("collection_name", Config.QDRANT_COLLECTION_NAME),
                api_key=self.config.get("qdrant_api_key", Config.QDRANT_API_KEY),
                embedding_model=self.config.get("embedding_model", Config.EMBEDDING_MODEL),
                embedding_dimension=self.config.get("embedding_dimension", Config.EMBEDDING_DIMENSION)
            )
            
            # Create retriever
            self.retriever = HybridRetriever(
                indexer=self.indexer,
                max_chunks=self.config.get("max_relevant_chunks", Config.MAX_RELEVANT_CHUNKS),
                similarity_threshold=self.config.get("similarity_threshold", Config.SIMILARITY_THRESHOLD),
                source_weights=self.config.get("source_weights", Config.SOURCE_WEIGHTS),
                reranker_model=self.config.get("reranker_model", Config.RERANKER_MODEL) if 
                               self.config.get("reranker_enabled", Config.RERANKER_ENABLED) else None
            )
            
            # Create LLM manager
            self.llm_manager = LLMManager(
                provider=self.config.get("llm_provider", Config.LLM_PROVIDER),
                model=self.config.get("llm_model", Config.LLM_MODEL),
                api_key=self.config.get("openai_api_key", Config.OPENAI_API_KEY),
                temperature=self.config.get("llm_temperature", Config.LLM_TEMPERATURE),
                max_tokens=self.config.get("llm_max_tokens", Config.LLM_MAX_TOKENS)
            )
            
            # Initialize connectors based on configuration
            await self._initialize_connectors()
            
            # Initialize indexer
            await self.indexer.initialize()
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            return False
    
    async def _initialize_connectors(self) -> None:
        """Initialize all data source connectors."""
        # Email connector
        if all([Config.EMAIL_SERVER, Config.EMAIL_USER, Config.EMAIL_PASSWORD]):
            email_config = {
                "provider": Config.EMAIL_PROVIDER,
                "server": Config.EMAIL_SERVER,
                "port": Config.EMAIL_PORT,
                "username": Config.EMAIL_USER,
                "password": Config.EMAIL_PASSWORD,
                "folders": Config.EMAIL_FOLDERS,
                "use_ssl": Config.EMAIL_USE_SSL,
                "filter": Config.EMAIL_FILTER
            }
            email_connector = ConnectorFactory.create_connector("email", email_config)
            if email_connector:
                self.connectors["email"] = email_connector
        
        # Mattermost connector
        if all([Config.MATTERMOST_URL, Config.MATTERMOST_TOKEN]):
            mattermost_config = {
                "url": Config.MATTERMOST_URL,
                "token": Config.MATTERMOST_TOKEN,
                "team": Config.MATTERMOST_TEAM,
                "channels": Config.MATTERMOST_CHANNELS
            }
            mattermost_connector = ConnectorFactory.create_connector("mattermost", mattermost_config)
            if mattermost_connector:
                self.connectors["mattermost"] = mattermost_connector
        
        # Odoo connector
        if all([Config.ODOO_HOST, Config.ODOO_USER, Config.ODOO_PASSWORD]):
            odoo_config = {
                "host": Config.ODOO_HOST,
                "port": Config.ODOO_PORT,
                "database": Config.ODOO_DB,
                "username": Config.ODOO_USER,
                "password": Config.ODOO_PASSWORD,
                "protocol": Config.ODOO_PROTOCOL,
                "modules": Config.ODOO_MODULES
            }
            odoo_connector = ConnectorFactory.create_connector("odoo", odoo_config)
            if odoo_connector:
                self.connectors["odoo"] = odoo_connector
        
        # SQL Database connector
        if Config.SQL_DB_URI:
            sql_config = {
                "connection_string": Config.SQL_DB_URI,
                "schema": Config.SQL_DB_NAME,
                "text_columns_only": True
            }
            sql_connector = ConnectorFactory.create_connector("database", sql_config)
            if sql_connector:
                self.connectors["database"] = sql_connector
        
        # File connector
        file_config = {
            "base_dirs": [os.path.join(os.path.dirname(__file__), "..", "data")],
            "recursive": True,
            "extensions": ["text", "markdown", "json", "pdf"]
        }
        file_connector = ConnectorFactory.create_connector("file", file_config)
        if file_connector:
            self.connectors["file"] = file_connector
    
    async def index_all_sources(self, limit_per_source: int = 100) -> Dict[str, int]:
        """
        Index all documents from all sources.
        
        Args:
            limit_per_source: Maximum number of documents to retrieve per source
            
        Returns:
            Dictionary with number of documents indexed per source
        """
        if not self.is_initialized:
            if not await self.initialize():
                return {}
        
        results = {}
        
        for source_name, connector in self.connectors.items():
            try:
                logger.info(f"Indexing documents from {source_name}...")
                documents = await connector.get_documents(limit=limit_per_source)
                
                # Process and index the documents
                count = await self._process_and_index_documents(documents, source_name)
                results[source_name] = count
                
                logger.info(f"Indexed {count} documents from {source_name}")
            except Exception as e:
                logger.error(f"Error indexing {source_name}: {e}")
                results[source_name] = 0
        
        return results
    
    async def index_source(self, source_name: str, query: Optional[str] = None, 
                         limit: Optional[int] = None, **kwargs) -> int:
        """
        Index documents from a specific source.
        
        Args:
            source_name: Name of the source to index
            query: Optional query to filter documents
            limit: Maximum number of documents to retrieve
            **kwargs: Additional source-specific parameters
            
        Returns:
            Number of documents indexed
        """
        if not self.is_initialized:
            if not await self.initialize():
                return 0
        
        if source_name not in self.connectors:
            logger.error(f"Source not found: {source_name}")
            return 0
        
        try:
            connector = self.connectors[source_name]
            documents = await connector.get_documents(query=query, limit=limit, **kwargs)
            
            # Process and index the documents
            count = await self._process_and_index_documents(documents, source_name)
            logger.info(f"Indexed {count} documents from {source_name}")
            
            return count
            
        except Exception as e:
            logger.error(f"Error indexing {source_name}: {e}")
            return 0
    
    async def _process_and_index_documents(self, documents: List[Document], source_name: str) -> int:
        """
        Process and index a list of documents.
        
        Args:
            documents: List of documents to process and index
            source_name: Source name (for logging/tracking)
            
        Returns:
            Number of chunks indexed
        """
        if not documents:
            return 0
        
        # Chunk the documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        
        # Index the chunks
        if all_chunks:
            await self.indexer.add_chunks(all_chunks)
        
        return len(all_chunks)
    
    async def query(self, rag_query: Union[str, RAGQuery]) -> RAGResponse:
        """
        Query the RAG system.
        
        Args:
            rag_query: Query string or RAGQuery object
            
        Returns:
            RAGResponse with answer and relevant chunks
        """
        if not self.is_initialized:
            if not await self.initialize():
                raise RuntimeError("Failed to initialize RAG system")
        
        # Convert string to RAGQuery if needed
        if isinstance(rag_query, str):
            rag_query = RAGQuery(query=rag_query)
        
        query_text = rag_query.query
        
        # Get relevant chunks from the retriever
        relevant_chunks = await self.retriever.retrieve(
            query=query_text,
            sources=rag_query.sources,
            filters=rag_query.filters,
            limit=rag_query.limit,
            use_reranking=rag_query.use_reranking,
            use_multi_query=rag_query.use_multi_query
        )
        
        # Format relevant chunks for LLM context
        context = self._format_chunks_for_llm(relevant_chunks)
        
        # Generate answer using LLM
        answer = await self.llm_manager.generate_answer(
            query=query_text,
            context=context,
            include_citations=rag_query.include_sources,
            citation_format=rag_query.citation_format,
            stream=rag_query.stream
        )
        
        # Convert chunks to RAGChunks
        rag_chunks = [
            RAGChunk(
                content=chunk.content,
                metadata=chunk.metadata,
                score=chunk.score,
                source_type=chunk.source_type,
                source_id=chunk.source_id,
                chunk_id=chunk.chunk_id
            )
            for chunk in relevant_chunks
        ]
        
        # Extract unique sources
        sources = self._extract_unique_sources(relevant_chunks)
        
        # Create response
        response = RAGResponse(
            query=query_text,
            answer=answer,
            chunks=rag_chunks,
            sources=sources,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "source_count": len(sources),
                "chunk_count": len(rag_chunks)
            }
        )
        
        return response
    
    def _format_chunks_for_llm(self, chunks: List[Any]) -> str:
        """
        Format chunks for LLM context.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Formatted context string
        """
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Format chunk with source information
            source_info = ""
            if "file_name" in chunk.metadata:
                source_info = f"Source: {chunk.metadata['file_name']}"
            elif "url" in chunk.metadata:
                source_info = f"Source: {chunk.metadata['url']}"
            elif "email_id" in chunk.metadata:
                source_info = f"Source: Email from {chunk.metadata.get('from', 'unknown')}"
            elif "model" in chunk.metadata and "id" in chunk.metadata:
                source_info = f"Source: {chunk.metadata['model']} (ID: {chunk.metadata['id']})"
            elif "channel_name" in chunk.metadata:
                source_info = f"Source: Mattermost channel {chunk.metadata['channel_name']}"
            else:
                source_info = f"Source: {chunk.source_type} (ID: {chunk.source_id})"
            
            formatted_chunks.append(f"[{i+1}] {source_info}\n{chunk.content}\n")
        
        return "\n".join(formatted_chunks)
    
    def _extract_unique_sources(self, chunks: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract unique sources from chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            List of unique source information
        """
        unique_sources = {}
        
        for chunk in chunks:
            source_id = chunk.source_id
            if source_id not in unique_sources:
                source_info = {
                    "id": source_id,
                    "type": chunk.source_type,
                }
                
                # Add source-specific metadata
                if chunk.source_type == "file":
                    if "file_path" in chunk.metadata:
                        source_info["path"] = chunk.metadata["file_path"]
                    if "file_name" in chunk.metadata:
                        source_info["name"] = chunk.metadata["file_name"]
                
                elif chunk.source_type == "email":
                    if "subject" in chunk.metadata:
                        source_info["subject"] = chunk.metadata["subject"]
                    if "from" in chunk.metadata:
                        source_info["from"] = chunk.metadata["from"]
                    if "date" in chunk.metadata:
                        source_info["date"] = chunk.metadata["date"]
                
                elif chunk.source_type == "odoo":
                    if "model" in chunk.metadata:
                        source_info["model"] = chunk.metadata["model"]
                    if "id" in chunk.metadata:
                        source_info["record_id"] = chunk.metadata["id"]
                
                elif chunk.source_type == "mattermost":
                    if "channel_name" in chunk.metadata:
                        source_info["channel"] = chunk.metadata["channel_name"]
                    if "username" in chunk.metadata:
                        source_info["user"] = chunk.metadata["username"]
                
                elif chunk.source_type == "database":
                    if "table" in chunk.metadata:
                        source_info["table"] = chunk.metadata["table"]
                    if "primary_keys" in chunk.metadata:
                        source_info["primary_keys"] = chunk.metadata["primary_keys"]
                
                unique_sources[source_id] = source_info
        
        return list(unique_sources.values())
    
    async def close(self) -> None:
        """Close all connections and resources."""
        # Close all connectors
        for name, connector in self.connectors.items():
            try:
                await connector.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting {name}: {e}")
        
        # Close indexer
        if self.indexer:
            try:
                await self.indexer.close()
            except Exception as e:
                logger.error(f"Error closing indexer: {e}")
        
        # Close LLM manager
        if self.llm_manager:
            try:
                await self.llm_manager.close()
            except Exception as e:
                logger.error(f"Error closing LLM manager: {e}")
        
        self.is_initialized = False