"""
Embedding model integration for vector representations.
"""

import logging
import os
import json
import asyncio
from typing import Any, Dict, List, Optional, Union
import numpy as np
import time
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embedding models and generation.
    
    Features:
    - Multiple embedding providers (SentenceTransformers, OpenAI, Cohere, etc.)
    - Caching for efficient processing
    - Batched embedding for performance
    - Model pooling strategies
    """
    
    def __init__(self,
                provider: str = "sentence_transformers",
                model_name: str = "BAAI/bge-large-en-v1.5",
                dimension: int = 1024,
                api_key: Optional[str] = None,
                cache_dir: Optional[str] = None,
                batch_size: int = 32,
                normalize: bool = True,
                pooling_strategy: str = "mean",
                use_cache: bool = True):
        """
        Initialize the embedding manager.
        
        Args:
            provider: Embedding provider (sentence_transformers, openai, cohere, huggingface)
            model_name: Name or path of the embedding model
            dimension: Dimension of the embeddings
            api_key: API key for cloud providers
            cache_dir: Directory for embedding cache
            batch_size: Batch size for embedding generation
            normalize: Whether to normalize embeddings to unit length
            pooling_strategy: Strategy for pooling token embeddings (mean, max, cls)
            use_cache: Whether to use embedding caching
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.dimension = dimension
        self.api_key = api_key
        self.batch_size = batch_size
        self.normalize = normalize
        self.pooling_strategy = pooling_strategy
        self.use_cache = use_cache
        
        # Set up cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path("./cache/embeddings")
        
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        self.model = None
        self.client = None
        self.tokenizer = None
    
    async def initialize(self) -> bool:
        """
        Initialize the embedding model.
        
        Returns:
            True if successful
        """
        try:
            if self.provider == "sentence_transformers":
                # Import here to avoid dependency if not using this provider
                from sentence_transformers import SentenceTransformer
                
                self.model = await asyncio.to_thread(
                    SentenceTransformer, self.model_name
                )
                
                # Verify embedding dimension
                test_embedding = await asyncio.to_thread(
                    self.model.encode, ["Test sentence"], convert_to_numpy=True
                )
                actual_dim = test_embedding.shape[1]
                
                if actual_dim != self.dimension:
                    logger.warning(
                        f"Embedding dimension mismatch: expected {self.dimension}, got {actual_dim}"
                    )
                    self.dimension = actual_dim
                
                return True
                
            elif self.provider == "openai":
                # Check for API key
                if not self.api_key:
                    self.api_key = os.environ.get("OPENAI_API_KEY")
                
                if not self.api_key:
                    logger.error("OpenAI API key not provided")
                    return False
                
                # Import and initialize client
                from openai import AsyncOpenAI
                
                self.client = AsyncOpenAI(api_key=self.api_key)
                
                # Verify embedding dimension with mapping
                embedding_dimensions = {
                    "text-embedding-ada-002": 1536,
                    "text-embedding-3-small": 1536,
                    "text-embedding-3-large": 3072
                }
                
                if self.model_name in embedding_dimensions:
                    self.dimension = embedding_dimensions[self.model_name]
                
                return True
                
            elif self.provider == "cohere":
                # Check for API key
                if not self.api_key:
                    self.api_key = os.environ.get("COHERE_API_KEY")
                
                if not self.api_key:
                    logger.error("Cohere API key not provided")
                    return False
                
                # Import and initialize client
                import cohere
                
                self.client = cohere.Client(api_key=self.api_key)
                
                # Cohere embedding dimensions vary by model
                embedding_dimensions = {
                    "embed-english-v3.0": 1024,
                    "embed-english-light-v3.0": 384,
                    "embed-multilingual-v3.0": 1024,
                    "embed-multilingual-light-v3.0": 384
                }
                
                if self.model_name in embedding_dimensions:
                    self.dimension = embedding_dimensions[self.model_name]
                
                return True
                
            elif self.provider == "huggingface":
                # Import here to avoid dependency if not using this provider
                from transformers import AutoModel, AutoTokenizer
                import torch
                
                # Load model and tokenizer
                self.tokenizer = await asyncio.to_thread(
                    AutoTokenizer.from_pretrained, self.model_name
                )
                
                self.model = await asyncio.to_thread(
                    AutoModel.from_pretrained, self.model_name
                )
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                
                # Test embedding to verify dimension
                test_tokens = await asyncio.to_thread(
                    self.tokenizer, ["Test sentence"], padding=True, truncation=True,
                    return_tensors="pt"
                )
                
                if torch.cuda.is_available():
                    test_tokens = {key: val.to("cuda") for key, val in test_tokens.items()}
                
                with torch.no_grad():
                    test_output = await asyncio.to_thread(
                        self.model, **test_tokens
                    )
                
                # Use last hidden state
                test_embedding = test_output.last_hidden_state
                
                # Apply pooling strategy
                if self.pooling_strategy == "mean":
                    test_embedding = torch.mean(test_embedding, dim=1)
                elif self.pooling_strategy == "max":
                    test_embedding = torch.max(test_embedding, dim=1)[0]
                elif self.pooling_strategy == "cls":
                    test_embedding = test_embedding[:, 0]
                
                test_embedding = test_embedding.cpu().numpy()
                actual_dim = test_embedding.shape[1]
                
                if actual_dim != self.dimension:
                    logger.warning(
                        f"Embedding dimension mismatch: expected {self.dimension}, got {actual_dim}"
                    )
                    self.dimension = actual_dim
                
                return True
                
            else:
                logger.error(f"Unsupported embedding provider: {self.provider}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            return False
    
    def _compute_hash(self, text: str) -> str:
        """
        Compute hash for text for caching.
        
        Args:
            text: Text to hash
            
        Returns:
            Hash string
        """
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_cache_path(self, text_hash: str) -> Path:
        """
        Get path for cached embedding.
        
        Args:
            text_hash: Hash of the text
            
        Returns:
            Path to cached embedding
        """
        provider_dir = self.cache_dir / self.provider / self.model_name.replace("/", "_")
        os.makedirs(provider_dir, exist_ok=True)
        return provider_dir / f"{text_hash}.npy"
    
    async def _check_cache(self, text: str) -> Optional[np.ndarray]:
        """
        Check if embedding is cached.
        
        Args:
            text: Text to check
            
        Returns:
            Cached embedding if available, None otherwise
        """
        if not self.use_cache:
            return None
        
        text_hash = self._compute_hash(text)
        cache_path = self._get_cache_path(text_hash)
        
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception as e:
                logger.warning(f"Error loading cached embedding: {e}")
                return None
        
        return None
    
    async def _save_to_cache(self, text: str, embedding: np.ndarray) -> None:
        """
        Save embedding to cache.
        
        Args:
            text: Original text
            embedding: Generated embedding
        """
        if not self.use_cache:
            return
        
        text_hash = self._compute_hash(text)
        cache_path = self._get_cache_path(text_hash)
        
        try:
            np.save(cache_path, embedding)
        except Exception as e:
            logger.warning(f"Error saving embedding to cache: {e}")
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Check cache first
        cached_embedding = await self._check_cache(text)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate new embedding
        embedding = await self._generate_single_embedding(text)
        
        # Save to cache
        await self._save_to_cache(text, embedding)
        
        return embedding
    
    async def _generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text without caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not self.model and not self.client:
            if not await self.initialize():
                raise RuntimeError("Failed to initialize embedding model")
        
        try:
            if self.provider == "sentence_transformers":
                embedding = await asyncio.to_thread(
                    self.model.encode, text, convert_to_numpy=True
                )
                
                # Ensure it's a 1D array
                if embedding.ndim > 1:
                    embedding = embedding.flatten()
                
                # Normalize if needed
                if self.normalize:
                    embedding = embedding / np.linalg.norm(embedding)
                
                return embedding
                
            elif self.provider == "openai":
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=text
                )
                
                embedding = np.array(response.data[0].embedding)
                
                # Normalize if needed
                if self.normalize:
                    embedding = embedding / np.linalg.norm(embedding)
                
                return embedding
                
            elif self.provider == "cohere":
                response = await asyncio.to_thread(
                    self.client.embed,
                    texts=[text],
                    model=self.model_name
                )
                
                embedding = np.array(response.embeddings[0])
                
                # Normalize if needed
                if self.normalize:
                    embedding = embedding / np.linalg.norm(embedding)
                
                return embedding
                
            elif self.provider == "huggingface":
                import torch
                
                # Tokenize
                tokens = await asyncio.to_thread(
                    self.tokenizer, [text], padding=True, truncation=True,
                    return_tensors="pt"
                )
                
                if torch.cuda.is_available():
                    tokens = {key: val.to("cuda") for key, val in tokens.items()}
                
                # Generate embedding
                with torch.no_grad():
                    output = await asyncio.to_thread(
                        self.model, **tokens
                    )
                
                # Use last hidden state
                embeddings = output.last_hidden_state
                
                # Apply pooling strategy
                if self.pooling_strategy == "mean":
                    embedding = torch.mean(embeddings, dim=1)
                elif self.pooling_strategy == "max":
                    embedding = torch.max(embeddings, dim=1)[0]
                elif self.pooling_strategy == "cls":
                    embedding = embeddings[:, 0]
                
                # Convert to numpy array
                embedding = embedding.cpu().numpy().flatten()
                
                # Normalize if needed
                if self.normalize:
                    embedding = embedding / np.linalg.norm(embedding)
                
                return embedding
                
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        all_embeddings = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_embeddings = []
            
            # Check cache for each text
            cache_hits = 0
            for text in batch:
                cached_embedding = await self._check_cache(text)
                if cached_embedding is not None:
                    batch_embeddings.append(cached_embedding)
                    cache_hits += 1
                else:
                    batch_embeddings.append(None)
            
            # Generate embeddings for cache misses
            if cache_hits < len(batch):
                # Collect texts that need embedding
                missing_indices = [i for i, emb in enumerate(batch_embeddings) if emb is None]
                missing_texts = [batch[i] for i in missing_indices]
                
                # Generate missing embeddings
                new_embeddings = await self._generate_batch_embeddings(missing_texts)
                
                # Save to cache and update batch_embeddings
                for i, text_idx in enumerate(missing_indices):
                    original_text = batch[text_idx]
                    new_embedding = new_embeddings[i]
                    
                    batch_embeddings[text_idx] = new_embedding
                    await self._save_to_cache(original_text, new_embedding)
            
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts without caching.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
        
        if not self.model and not self.client:
            if not await self.initialize():
                raise RuntimeError("Failed to initialize embedding model")
        
        try:
            if self.provider == "sentence_transformers":
                batch_embeddings = await asyncio.to_thread(
                    self.model.encode, texts, convert_to_numpy=True
                )
                
                # Normalize if needed
                if self.normalize:
                    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    batch_embeddings = batch_embeddings / norms
                
                return batch_embeddings
                
            elif self.provider == "openai":
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=texts
                )
                
                batch_embeddings = np.array([item.embedding for item in response.data])
                
                # Normalize if needed
                if self.normalize:
                    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    batch_embeddings = batch_embeddings / norms
                
                return batch_embeddings
                
            elif self.provider == "cohere":
                response = await asyncio.to_thread(
                    self.client.embed,
                    texts=texts,
                    model=self.model_name
                )
                
                batch_embeddings = np.array(response.embeddings)
                
                # Normalize if needed
                if self.normalize:
                    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    batch_embeddings = batch_embeddings / norms
                
                return batch_embeddings
                
            elif self.provider == "huggingface":
                import torch
                
                # Use batching for better performance
                tokens = await asyncio.to_thread(
                    self.tokenizer, texts, padding=True, truncation=True,
                    return_tensors="pt"
                )
                
                if torch.cuda.is_available():
                    tokens = {key: val.to("cuda") for key, val in tokens.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    output = await asyncio.to_thread(
                        self.model, **tokens
                    )
                
                # Use last hidden state
                embeddings = output.last_hidden_state
                
                # Apply pooling strategy
                if self.pooling_strategy == "mean":
                    batch_embeddings = torch.mean(embeddings, dim=1)
                elif self.pooling_strategy == "max":
                    batch_embeddings = torch.max(embeddings, dim=1)[0]
                elif self.pooling_strategy == "cls":
                    batch_embeddings = embeddings[:, 0]
                
                # Convert to numpy array
                batch_embeddings = batch_embeddings.cpu().numpy()
                
                # Normalize if needed
                if self.normalize:
                    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    batch_embeddings = batch_embeddings / norms
                
                return batch_embeddings
                
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    async def close(self) -> None:
        """Close the embedding model."""
        # Clean up resources
        if hasattr(self.model, "close"):
            try:
                if asyncio.iscoroutinefunction(self.model.close):
                    await self.model.close()
                else:
                    await asyncio.to_thread(self.model.close)
            except Exception as e:
                logger.error(f"Error closing embedding model: {e}")
        
        if hasattr(self.client, "close"):
            try:
                if asyncio.iscoroutinefunction(self.client.close):
                    await self.client.close()
                else:
                    await asyncio.to_thread(self.client.close)
            except Exception as e:
                logger.error(f"Error closing client: {e}")
        
        self.model = None
        self.client = None
        self.tokenizer = None