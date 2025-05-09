"""
Hybrid retrieval system that combines vector similarity with other ranking methods.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import re
import numpy as np
from collections import defaultdict

from ..indexers.qdrant_indexer import QdrantIndexer, DocumentChunk

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval system for RAG.
    
    Combines:
    - Vector similarity search
    - BM25/keyword weighting
    - Source type weighting
    - Query expansion
    - Re-ranking
    """
    
    def __init__(self, 
                 indexer: QdrantIndexer,
                 max_chunks: int = 10,
                 similarity_threshold: float = 0.7,
                 source_weights: Optional[Dict[str, float]] = None,
                 reranker_model: Optional[str] = None,
                 use_hybrid_score: bool = True):
        """
        Initialize the hybrid retriever.
        
        Args:
            indexer: Vector database indexer
            max_chunks: Maximum number of chunks to return
            similarity_threshold: Minimum similarity score threshold
            source_weights: Weights for different source types
            reranker_model: Optional cross-encoder model for reranking
            use_hybrid_score: Whether to use hybrid scoring
        """
        self.indexer = indexer
        self.max_chunks = max_chunks
        self.similarity_threshold = similarity_threshold
        self.source_weights = source_weights or {}
        self.reranker_model = reranker_model
        self.use_hybrid_score = use_hybrid_score
        self.cross_encoder = None
    
    async def _load_reranker(self) -> bool:
        """
        Load the cross-encoder reranker model.
        
        Returns:
            True if successful
        """
        if not self.reranker_model:
            return False
        
        if self.cross_encoder:
            return True
        
        try:
            from sentence_transformers import CrossEncoder
            
            self.cross_encoder = await asyncio.to_thread(
                CrossEncoder, self.reranker_model
            )
            return True
        except Exception as e:
            logger.error(f"Error loading reranker model: {e}")
            return False
    
    async def _expand_query(self, query: str) -> List[str]:
        """
        Generate query variations using an LLM.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations
        """
        try:
            # TODO: Replace with actual LLM integration
            # For now, use some heuristics to generate variations
            
            # Lowercase variation
            variations = [query.lower()]
            
            # Remove question marks
            if "?" in query:
                variations.append(query.replace("?", ""))
            
            # Convert question to statement
            question_starters = ["what is", "how do", "can you", "where is", "who is", "when is"]
            lower_query = query.lower()
            
            for starter in question_starters:
                if lower_query.startswith(starter):
                    statement = lower_query.replace(starter, "").strip()
                    variations.append(statement)
                    break
            
            # Add keyword-only version
            # Get nouns and main content words with simple regex
            words = re.findall(r'\b[a-zA-Z]{3,}\b', query)
            if len(words) >= 2:
                variations.append(" ".join(words))
            
            # Return unique variations
            return list(set(variations))
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return [query]
    
    def _compute_keyword_scores(self, query: str, chunks: List[DocumentChunk]) -> Dict[str, float]:
        """
        Compute keyword-based relevance scores for chunks.
        
        Args:
            query: Search query
            chunks: List of document chunks
            
        Returns:
            Dictionary mapping chunk IDs to keyword scores
        """
        # Extract important keywords from query
        # Simple approach: take words with at least 4 characters
        keywords = [word.lower() for word in re.findall(r'\b[a-zA-Z]{4,}\b', query)]
        
        if not keywords:
            # If no keywords, use all words
            keywords = [word.lower() for word in query.split()]
        
        # Compute TF-IDF inspired scores
        keyword_scores = {}
        
        # Count document frequency of each keyword
        df = defaultdict(int)
        for chunk in chunks:
            chunk_text = chunk.content.lower()
            for keyword in keywords:
                if keyword in chunk_text:
                    df[keyword] += 1
        
        # Compute inverse document frequency
        num_docs = len(chunks)
        idf = {keyword: np.log((num_docs + 1) / (count + 1)) + 1 for keyword, count in df.items()}
        
        # Compute scores for each chunk
        for chunk in chunks:
            chunk_text = chunk.content.lower()
            score = 0
            
            for keyword in keywords:
                # Term frequency in this chunk
                tf = chunk_text.count(keyword) / len(chunk_text.split())
                
                # TF-IDF score
                score += tf * idf[keyword]
            
            keyword_scores[chunk.chunk_id] = score
        
        # Normalize scores
        if keyword_scores:
            max_score = max(keyword_scores.values())
            if max_score > 0:
                keyword_scores = {k: v / max_score for k, v in keyword_scores.items()}
        
        return keyword_scores
    
    async def _rerank_chunks(self, query: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Rerank chunks using a cross-encoder model.
        
        Args:
            query: Search query
            chunks: List of document chunks
            
        Returns:
            Reranked list of chunks
        """
        if not await self._load_reranker():
            return chunks
        
        try:
            # Prepare sentence pairs
            sentence_pairs = [(query, chunk.content) for chunk in chunks]
            
            # Compute cross-encoder scores
            scores = await asyncio.to_thread(self.cross_encoder.predict, sentence_pairs)
            
            # Assign scores to chunks
            for i, score in enumerate(scores):
                chunks[i].score = float(score)
            
            # Sort by score (descending)
            reranked_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
            
            return reranked_chunks
        except Exception as e:
            logger.error(f"Error reranking chunks: {e}")
            return chunks
    
    async def retrieve(self, 
                     query: str, 
                     sources: Optional[List[str]] = None,
                     filters: Optional[Dict[str, Any]] = None,
                     limit: Optional[int] = None,
                     use_reranking: Optional[bool] = None,
                     use_multi_query: Optional[bool] = None) -> List[DocumentChunk]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: Search query
            sources: Optional list of source types to filter by
            filters: Optional additional filters
            limit: Maximum number of chunks to return
            use_reranking: Whether to use reranking
            use_multi_query: Whether to use query expansion
            
        Returns:
            List of relevant document chunks
        """
        limit = limit or self.max_chunks
        
        # Apply source type filter
        search_filters = filters or {}
        if sources:
            search_filters["source_type"] = sources
        
        # Determine whether to use query expansion
        should_expand = use_multi_query if use_multi_query is not None else (sources is None)
        
        try:
            # If using query expansion, run searches for different query variations
            all_chunks = []
            seen_chunk_ids = set()
            
            if should_expand:
                # Generate query variations
                query_variations = await self._expand_query(query)
                
                # Run search for each variation
                for variation in query_variations:
                    chunks = await self.indexer.search(
                        query=variation,
                        limit=limit,
                        filters=search_filters
                    )
                    
                    # Add new chunks
                    for chunk in chunks:
                        if chunk.chunk_id not in seen_chunk_ids:
                            all_chunks.append(chunk)
                            seen_chunk_ids.add(chunk.chunk_id)
            else:
                # Single query mode
                all_chunks = await self.indexer.search(
                    query=query,
                    limit=limit,
                    filters=search_filters
                )
            
            # Apply hybrid scoring if enabled
            if self.use_hybrid_score and all_chunks:
                # Compute keyword-based scores
                keyword_scores = self._compute_keyword_scores(query, all_chunks)
                
                # Apply source weighting and combine scores
                for chunk in all_chunks:
                    # Get source weight (default to 1.0)
                    source_weight = self.source_weights.get(chunk.source_type, 1.0)
                    
                    # Get keyword score (default to 0.0)
                    keyword_score = keyword_scores.get(chunk.chunk_id, 0.0)
                    
                    # Combine scores (vector score * 0.7 + keyword score * 0.3) * source weight
                    chunk.score = ((chunk.score * 0.7) + (keyword_score * 0.3)) * source_weight
                
                # Re-sort by combined score
                all_chunks = sorted(all_chunks, key=lambda x: x.score, reverse=True)
            
            # Apply reranking if enabled
            should_rerank = use_reranking if use_reranking is not None else bool(self.reranker_model)
            
            if should_rerank and len(all_chunks) > 1:
                all_chunks = await self._rerank_chunks(query, all_chunks)
            
            # Apply score threshold
            filtered_chunks = [
                chunk for chunk in all_chunks 
                if chunk.score >= self.similarity_threshold
            ]
            
            # Limit number of chunks
            return filtered_chunks[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []