"""
API for the Complex RAG system.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio
import uvicorn

from ..rag_system import ComplexRAGSystem, RAGQuery, RAGResponse, RAGChunk
from . import document_routes
from config.settings import Config

# Set up logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Complex RAG API",
    description="API for a multi-source Retrieval-Augmented Generation system",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register document routes
document_routes.register_routes(app)

# Create RAG system instance
rag_system = ComplexRAGSystem()
initialization_task = None


class QueryRequest(BaseModel):
    """Request for querying the RAG system."""
    
    query: str
    sources: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    use_reranking: Optional[bool] = None
    use_multi_query: Optional[bool] = None
    include_sources: Optional[bool] = True
    citation_format: Optional[str] = "inline"
    stream: Optional[bool] = False


class ChunkResponse(BaseModel):
    """Response representation of a document chunk."""
    
    content: str
    metadata: Dict[str, Any]
    score: float
    source_type: str
    source_id: str
    chunk_id: str


class QueryResponse(BaseModel):
    """Response for a RAG query."""
    
    query: str
    answer: str
    chunks: Optional[List[ChunkResponse]] = None
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IndexResponse(BaseModel):
    """Response for indexing operations."""
    
    indexed_count: int
    source: str
    status: str
    message: Optional[str] = None


class SourceListResponse(BaseModel):
    """Response for listing available sources."""
    
    sources: List[str]
    available_sources: List[Dict[str, Any]]


async def get_or_initialize_rag():
    """
    Get or initialize the RAG system.
    
    Returns:
        Initialized RAG system
    """
    global initialization_task, rag_system
    
    if not rag_system.is_initialized:
        if initialization_task is None or initialization_task.done():
            initialization_task = asyncio.create_task(rag_system.initialize())
        
        try:
            await initialization_task
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize RAG system: {str(e)}"
            )
    
    return rag_system


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global initialization_task
    initialization_task = asyncio.create_task(rag_system.initialize())


@app.on_event("shutdown")
async def shutdown_event():
    """Close the RAG system on shutdown."""
    await rag_system.close()


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return {
        "status": "ok",
        "version": "0.1.0",
        "rag_initialized": rag_system.is_initialized
    }


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest, rag: ComplexRAGSystem = Depends(get_or_initialize_rag)
):
    """
    Query the RAG system.
    
    Args:
        request: Query request
        rag: RAG system instance
        
    Returns:
        Query response with answer and relevant chunks
    """
    try:
        # Convert request to RAGQuery
        rag_query = RAGQuery(
            query=request.query,
            sources=request.sources,
            filters=request.filters,
            limit=request.limit,
            use_reranking=request.use_reranking,
            use_multi_query=request.use_multi_query,
            include_sources=request.include_sources,
            citation_format=request.citation_format,
            stream=request.stream
        )
        
        # Run query
        response = await rag.query(rag_query)
        
        # Convert response to API model
        chunks = None
        if response.chunks:
            chunks = [
                ChunkResponse(
                    content=chunk.content,
                    metadata=chunk.metadata,
                    score=chunk.score,
                    source_type=chunk.source_type,
                    source_id=chunk.source_id,
                    chunk_id=chunk.chunk_id
                )
                for chunk in response.chunks
            ]
        
        return QueryResponse(
            query=response.query,
            answer=response.answer,
            chunks=chunks,
            sources=response.sources,
            metadata=response.metadata
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/{source_name}", response_model=IndexResponse)
async def index_source(
    source_name: str,
    query: Optional[str] = None,
    limit: Optional[int] = Query(None, description="Maximum number of documents to index"),
    background_tasks: BackgroundTasks = None,
    rag: ComplexRAGSystem = Depends(get_or_initialize_rag)
):
    """
    Index documents from a specific source.
    
    Args:
        source_name: Name of the source to index
        query: Optional query to filter documents
        limit: Maximum number of documents to index
        background_tasks: FastAPI background tasks
        rag: RAG system instance
        
    Returns:
        Indexing response
    """
    if source_name not in rag.connectors:
        raise HTTPException(status_code=404, detail=f"Source '{source_name}' not found")
    
    try:
        # Run indexing
        count = await rag.index_source(source_name, query=query, limit=limit)
        
        return IndexResponse(
            indexed_count=count,
            source=source_name,
            status="success",
            message=f"Indexed {count} documents from {source_name}"
        )
    except Exception as e:
        logger.error(f"Error indexing {source_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index", response_model=Dict[str, IndexResponse])
async def index_all_sources(
    limit_per_source: int = Query(100, description="Maximum number of documents per source"),
    background_tasks: BackgroundTasks = None,
    rag: ComplexRAGSystem = Depends(get_or_initialize_rag)
):
    """
    Index documents from all sources.
    
    Args:
        limit_per_source: Maximum number of documents per source
        background_tasks: FastAPI background tasks
        rag: RAG system instance
        
    Returns:
        Dictionary with indexing responses for each source
    """
    try:
        # Run indexing
        results = await rag.index_all_sources(limit_per_source=limit_per_source)
        
        # Convert results to API responses
        responses = {}
        for source, count in results.items():
            responses[source] = IndexResponse(
                indexed_count=count,
                source=source,
                status="success",
                message=f"Indexed {count} documents from {source}"
            )
        
        return responses
    except Exception as e:
        logger.error(f"Error indexing all sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources", response_model=SourceListResponse)
async def list_sources(rag: ComplexRAGSystem = Depends(get_or_initialize_rag)):
    """
    List available sources.
    
    Args:
        rag: RAG system instance
        
    Returns:
        List of available sources
    """
    sources = list(rag.connectors.keys())
    
    # Get detailed info for each source
    source_info = []
    for name, connector in rag.connectors.items():
        info = connector.get_source_info()
        source_info.append(info)
    
    return SourceListResponse(
        sources=sources,
        available_sources=source_info
    )


@app.get("/stats", response_model=Dict[str, Any])
async def get_stats(rag: ComplexRAGSystem = Depends(get_or_initialize_rag)):
    """
    Get statistics about the RAG system.
    
    Args:
        rag: RAG system instance
        
    Returns:
        Dictionary with statistics
    """
    try:
        index_stats = await rag.indexer.get_stats()
        
        # Add source counts
        source_counts = {name: 1 for name in rag.connectors.keys()}
        
        return {
            "index_stats": index_stats,
            "source_counts": source_counts,
            "config": {
                "max_chunks": rag.retriever.max_chunks,
                "similarity_threshold": rag.retriever.similarity_threshold,
                "use_reranking": bool(rag.retriever.reranker_model),
                "embedding_model": rag.indexer.embedding_model_name
            }
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)