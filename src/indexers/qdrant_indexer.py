"""
Qdrant-based vector database indexer for storing and retrieving text chunks.
"""

import logging
import os
import json
import asyncio
from typing import Any, Dict, List, Optional, Union
import uuid
import time
import numpy as np

from ..models.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Representation of a document chunk for indexing and retrieval."""
    
    def __init__(self, 
                 content: str,
                 metadata: Dict[str, Any],
                 source_type: str,
                 source_id: str,
                 chunk_id: Optional[str] = None,
                 embedding: Optional[List[float]] = None,
                 score: float = 0.0):
        """
        Initialize a document chunk.
        
        Args:
            content: Text content of the chunk
            metadata: Additional metadata about the chunk
            source_type: Type of source (email, mattermost, odoo, etc.)
            source_id: ID of the source document
            chunk_id: Unique ID for this chunk (generated if not provided)
            embedding: Vector embedding of the content (optional)
            score: Similarity score (only used during retrieval)
        """
        self.content = content
        self.metadata = metadata
        self.source_type = source_type
        self.source_id = source_id
        self.chunk_id = chunk_id or str(uuid.uuid4())
        self.embedding = embedding
        self.score = score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            metadata=data["metadata"],
            source_type=data["source_type"],
            source_id=data["source_id"],
            chunk_id=data.get("chunk_id"),
            embedding=data.get("embedding"),
            score=data.get("score", 0.0)
        )


class QdrantIndexer:
    """
    Indexer for storing and retrieving text chunks using Qdrant vector database.
    
    Features:
    - Embedding generation with multiple provider options
    - Storage and retrieval of document chunks
    - Metadata filtering
    - Similarity search
    - Payload management
    """
    
    def __init__(self, 
                 url: str = "http://localhost:6333",
                 collection_name: str = "documents",
                 api_key: Optional[str] = None,
                 embedding_provider: str = "sentence_transformers",
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 embedding_dimension: int = 1024,
                 distance_metric: str = "cosine",
                 embedding_api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 batch_size: int = 32):
        """
        Initialize the Qdrant indexer.
        
        Args:
            url: Qdrant server URL
            collection_name: Name of the collection to use
            api_key: Optional API key for Qdrant
            embedding_provider: Embedding provider (sentence_transformers, openai, cohere, huggingface)
            embedding_model: Name or path of the embedding model
            embedding_dimension: Dimension of the embeddings
            distance_metric: Distance metric (cosine, dot, euclidean)
            embedding_api_key: API key for embedding service
            cache_dir: Directory for embedding cache
            batch_size: Batch size for embedding generation
        """
        self.url = url
        self.collection_name = collection_name
        self.api_key = api_key
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric
        self.batch_size = batch_size
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(
            provider=embedding_provider,
            model_name=embedding_model,
            dimension=embedding_dimension,
            api_key=embedding_api_key,
            cache_dir=cache_dir,
            batch_size=batch_size,
            normalize=True
        )
        
        self.client = None
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize the indexer, connecting to Qdrant and loading the embedding model.
        
        Returns:
            True if initialization was successful
        """
        try:
            # Import here to avoid dependency if not using this indexer
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
            
            # Connect to Qdrant
            logger.info(f"Connecting to Qdrant at {self.url}")
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
            
            # Initialize embedding manager
            if not await self.embedding_manager.initialize():
                logger.error("Failed to initialize embedding manager")
                return False
            
            # Update embedding dimension from embedding manager
            self.embedding_dimension = self.embedding_manager.dimension
            
            # Map distance metrics
            distance_map = {
                "cosine": models.Distance.COSINE,
                "dot": models.Distance.DOT,
                "euclidean": models.Distance.EUCLID
            }
            
            qdrant_distance = distance_map.get(self.distance_metric.lower(), models.Distance.COSINE)
            
            # Ensure collection exists
            collections = await asyncio.to_thread(self.client.get_collections)
            collection_exists = False
            
            for collection in collections.collections:
                if collection.name == self.collection_name:
                    collection_exists = True
                    break
            
            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dimension,
                        distance=qdrant_distance
                    )
                )
                
                # Create index for source_type field
                await asyncio.to_thread(
                    self.client.create_payload_index,
                    collection_name=self.collection_name,
                    field_name="source_type",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                
                # Create index for source_id field
                await asyncio.to_thread(
                    self.client.create_payload_index,
                    collection_name=self.collection_name,
                    field_name="source_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Qdrant indexer: {e}")
            return False
    
    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Add document chunks to the index.
        
        Args:
            chunks: List of document chunks to add
            
        Returns:
            True if successful
        """
        if not self.is_initialized:
            if not await self.initialize():
                return False
        
        try:
            # Generate embeddings for chunks without existing embeddings
            chunks_without_embeddings = [i for i, chunk in enumerate(chunks) if chunk.embedding is None]
            
            if chunks_without_embeddings:
                texts = [chunks[i].content for i in chunks_without_embeddings]
                embeddings = await self.embedding_manager.generate_embeddings(texts)
                
                # Update chunks with embeddings
                for idx, embedding_idx in enumerate(chunks_without_embeddings):
                    chunks[embedding_idx].embedding = embeddings[idx].tolist()
            
            # Prepare data for Qdrant
            points = []
            from qdrant_client.http import models
            
            for chunk in chunks:
                # Convert metadata to payload
                payload = {
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "source_type": chunk.source_type,
                    "source_id": chunk.source_id
                }
                
                points.append(models.PointStruct(
                    id=chunk.chunk_id,
                    vector=chunk.embedding,
                    payload=payload
                ))
            
            # Add to Qdrant in batches of 100
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                await asyncio.to_thread(
                    self.client.upsert,
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.info(f"Added {len(chunks)} chunks to Qdrant index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to Qdrant: {e}")
            return False
    
    async def search(self, 
                    query: str, 
                    limit: int = 10, 
                    filters: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Search for relevant chunks by embedding similarity.
        
        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters for metadata
            
        Returns:
            List of relevant document chunks
        """
        if not self.is_initialized:
            if not await self.initialize():
                return []
        
        try:
            # Generate embedding for query
            query_embedding = await self.embedding_manager.generate_embedding(query)
            
            # Convert filters to Qdrant format
            qdrant_filter = None
            if filters:
                from qdrant_client.http import models
                
                filter_conditions = []
                for key, value in filters.items():
                    if key == "source_type" and isinstance(value, (list, tuple)):
                        filter_conditions.append(
                            models.FieldCondition(
                                key="source_type",
                                match=models.MatchAny(any=value)
                            )
                        )
                    elif key == "source_id" and isinstance(value, (list, tuple)):
                        filter_conditions.append(
                            models.FieldCondition(
                                key="source_id",
                                match=models.MatchAny(any=value)
                            )
                        )
                    elif key == "metadata":
                        # Handle metadata filters
                        for meta_key, meta_value in value.items():
                            filter_path = f"metadata.{meta_key}"
                            if isinstance(meta_value, (list, tuple)):
                                filter_conditions.append(
                                    models.FieldCondition(
                                        key=filter_path,
                                        match=models.MatchAny(any=meta_value)
                                    )
                                )
                            else:
                                filter_conditions.append(
                                    models.FieldCondition(
                                        key=filter_path,
                                        match=models.MatchValue(value=meta_value)
                                    )
                                )
                
                if filter_conditions:
                    qdrant_filter = models.Filter(
                        must=filter_conditions
                    )
            
            # Perform search
            search_result = await asyncio.to_thread(
                self.client.search,
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                query_filter=qdrant_filter
            )
            
            # Convert results to DocumentChunks
            chunks = []
            for result in search_result:
                payload = result.payload
                chunk = DocumentChunk(
                    content=payload["content"],
                    metadata=payload["metadata"],
                    source_type=payload["source_type"],
                    source_id=payload["source_id"],
                    chunk_id=str(result.id),
                    score=result.score
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return []
    
    async def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """
        Delete chunks matching the filters.
        
        Args:
            filters: Filters to match chunks for deletion
            
        Returns:
            Number of deleted chunks
        """
        if not self.is_initialized:
            if not await self.initialize():
                return 0
        
        try:
            # Convert filters to Qdrant format
            from qdrant_client.http import models
            
            filter_conditions = []
            for key, value in filters.items():
                if key == "source_type" and isinstance(value, (list, tuple)):
                    filter_conditions.append(
                        models.FieldCondition(
                            key="source_type",
                            match=models.MatchAny(any=value)
                        )
                    )
                elif key == "source_id" and isinstance(value, (list, tuple)):
                    filter_conditions.append(
                        models.FieldCondition(
                            key="source_id",
                            match=models.MatchAny(any=value)
                        )
                    )
                elif key == "chunk_ids" and isinstance(value, (list, tuple)):
                    # Delete specific chunk IDs
                    result = await asyncio.to_thread(
                        self.client.delete,
                        collection_name=self.collection_name,
                        points_selector=models.PointIdsList(
                            points=value
                        )
                    )
                    return result.deleted
            
            if filter_conditions:
                qdrant_filter = models.Filter(
                    must=filter_conditions
                )
                
                result = await asyncio.to_thread(
                    self.client.delete,
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(
                        filter=qdrant_filter
                    )
                )
                return result.deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Error deleting from Qdrant: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self.is_initialized:
            if not await self.initialize():
                return {}
        
        try:
            collection_info = await asyncio.to_thread(
                self.client.get_collection,
                collection_name=self.collection_name
            )
            
            # Get counts by source_type
            from qdrant_client.http import models
            
            source_types = set()
            counts = {}
            
            # First, get all unique source types
            scroll_result = await asyncio.to_thread(
                self.client.scroll,
                collection_name=self.collection_name,
                limit=100,
                with_payload=["source_type"],
                with_vectors=False
            )
            
            for point in scroll_result[0]:
                if "source_type" in point.payload:
                    source_types.add(point.payload["source_type"])
            
            # Then count each source type
            for source_type in source_types:
                count_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_type",
                            match=models.MatchValue(value=source_type)
                        )
                    ]
                )
                
                count = await asyncio.to_thread(
                    self.client.count,
                    collection_name=self.collection_name,
                    count_filter=count_filter
                )
                
                counts[source_type] = count.count
            
            # Get embedding information
            embedding_info = {
                "provider": self.embedding_provider,
                "model": self.embedding_model_name,
                "dimension": self.embedding_dimension
            }
            
            return {
                "vector_size": collection_info.config.params.vectors.size,
                "total_chunks": collection_info.vectors_count,
                "source_type_counts": counts,
                "indexing_is_on": collection_info.config.hnsw_config.on_disk,
                "embedding": embedding_info,
                "last_updated": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting Qdrant stats: {e}")
            return {}
    
    async def close(self) -> None:
        """Close connections and free resources."""
        # Close embedding manager
        if self.embedding_manager:
            try:
                await self.embedding_manager.close()
            except Exception as e:
                logger.error(f"Error closing embedding manager: {e}")
        
        # Close Qdrant client
        if self.client:
            try:
                await asyncio.to_thread(self.client.close)
            except Exception as e:
                logger.error(f"Error closing Qdrant client: {e}")
            finally:
                self.client = None
        
        self.is_initialized = False