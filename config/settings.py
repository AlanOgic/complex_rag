"""
Configuration settings for the Complex RAG system.
Load this module to access all configuration values.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"

# General settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Vector DB settings
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "qdrant")  # Options: qdrant, pinecone, weaviate, etc.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "documents")
VECTOR_DISTANCE_METRIC = os.getenv("VECTOR_DISTANCE_METRIC", "cosine")  # Options: cosine, dot, euclidean

# SQL Database settings
SQL_DB_TYPE = os.getenv("SQL_DB_TYPE", "postgresql")  # Options: postgresql, mysql, sqlite
SQL_DB_HOST = os.getenv("SQL_DB_HOST", "localhost")
SQL_DB_PORT = int(os.getenv("SQL_DB_PORT", "5432"))
SQL_DB_USER = os.getenv("SQL_DB_USER", "postgres")
SQL_DB_PASSWORD = os.getenv("SQL_DB_PASSWORD", "")
SQL_DB_NAME = os.getenv("SQL_DB_NAME", "complex_rag")
SQL_DB_URI = os.getenv(
    "SQL_DB_URI", 
    f"{SQL_DB_TYPE}://{SQL_DB_USER}:{SQL_DB_PASSWORD}@{SQL_DB_HOST}:{SQL_DB_PORT}/{SQL_DB_NAME}"
)

# Odoo settings
ODOO_HOST = os.getenv("ODOO_HOST", "localhost")
ODOO_PORT = int(os.getenv("ODOO_PORT", "8069"))
ODOO_DB = os.getenv("ODOO_DB", "odoo")
ODOO_USER = os.getenv("ODOO_USER", "admin")
ODOO_PASSWORD = os.getenv("ODOO_PASSWORD", "admin")
ODOO_PROTOCOL = os.getenv("ODOO_PROTOCOL", "jsonrpc")  # Options: jsonrpc, xmlrpc
ODOO_MODULES = os.getenv("ODOO_MODULES", "").split(",")

# Mattermost settings
MATTERMOST_URL = os.getenv("MATTERMOST_URL")
MATTERMOST_TOKEN = os.getenv("MATTERMOST_TOKEN")
MATTERMOST_TEAM = os.getenv("MATTERMOST_TEAM")
MATTERMOST_CHANNELS = os.getenv("MATTERMOST_CHANNELS", "").split(",")

# Email settings
EMAIL_PROVIDER = os.getenv("EMAIL_PROVIDER", "imap")  # Options: imap, gmail, exchange
EMAIL_SERVER = os.getenv("EMAIL_SERVER")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "993"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_USE_SSL = os.getenv("EMAIL_USE_SSL", "True").lower() == "true"
EMAIL_FOLDERS = os.getenv("EMAIL_FOLDERS", "INBOX").split(",")
EMAIL_FILTER = os.getenv("EMAIL_FILTER", "ALL")

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MAX_DOCUMENT_SIZE_MB = float(os.getenv("MAX_DOCUMENT_SIZE_MB", "10"))
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "fixed")  # Options: fixed, sentence, paragraph, semantic
SUPPORTED_FILE_TYPES = [
    "txt", "pdf", "md", "json", "csv", "html", "docx", "pptx", "xlsx", "eml", "xml", "yaml", "yml"
]

# Embedding model settings
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")  # Options: sentence_transformers, openai, cohere, huggingface
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", str(CACHE_DIR / "embeddings"))
EMBEDDING_POOLING_STRATEGY = os.getenv("EMBEDDING_POOLING_STRATEGY", "mean")  # Options: mean, max, cls (for HF models)
EMBEDDING_NORMALIZE = os.getenv("EMBEDDING_NORMALIZE", "True").lower() == "true"
EMBEDDING_USE_CACHE = os.getenv("EMBEDDING_USE_CACHE", "True").lower() == "true"

# Available embedding model mappings
EMBEDDING_MODELS = {
    "sentence_transformers": {
        "BAAI/bge-large-en-v1.5": 1024,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384
    },
    "openai": {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072
    },
    "cohere": {
        "embed-english-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-v3.0": 1024,
        "embed-multilingual-light-v3.0": 384
    },
    "huggingface": {
        "intfloat/e5-large-v2": 1024,
        "intfloat/e5-base-v2": 768,
        "intfloat/e5-small-v2": 384,
        "thenlper/gte-large": 1024,
        "thenlper/gte-base": 768
    }
}

# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # Options: openai, anthropic, cohere
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# Retrieval settings
MAX_RELEVANT_CHUNKS = int(os.getenv("MAX_RELEVANT_CHUNKS", "10"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "True").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
MULTI_QUERY_ENABLED = os.getenv("MULTI_QUERY_ENABLED", "True").lower() == "true"

# Data source weights for hybrid search
SOURCE_WEIGHTS = {
    "email": float(os.getenv("EMAIL_WEIGHT", "1.0")),
    "mattermost": float(os.getenv("MATTERMOST_WEIGHT", "1.0")),
    "odoo": float(os.getenv("ODOO_WEIGHT", "1.0")),
    "pdf": float(os.getenv("PDF_WEIGHT", "1.0")),
    "markdown": float(os.getenv("MARKDOWN_WEIGHT", "1.0")),
    "text": float(os.getenv("TEXT_WEIGHT", "1.0")),
    "json": float(os.getenv("JSON_WEIGHT", "1.0")),
    "database": float(os.getenv("DATABASE_WEIGHT", "1.0"))
}

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "4"))
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "60"))
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "100"))
API_CORS_ORIGINS = os.getenv("API_CORS_ORIGINS", "*").split(",")

# Streaming settings
ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "True").lower() == "true"

# Cache settings
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "True").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # in seconds

# Monitoring and metrics
ENABLE_TELEMETRY = os.getenv("ENABLE_TELEMETRY", "False").lower() == "true"
METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))

# Graph settings
GRAPH_ENABLED = os.getenv("GRAPH_ENABLED", "True").lower() == "true"
GRAPH_RELATIONSHIPS = [
    "mentions", "references", "contains", "relates_to", "precedes", "follows"
]

class Config:
    """Configuration class that can be imported in other modules."""
    
    @classmethod
    def get_database_uri(cls) -> str:
        """Generate database URI based on settings."""
        return SQL_DB_URI
    
    @classmethod
    def get_supported_sources(cls) -> List[str]:
        """Get list of enabled data sources."""
        sources = []
        
        # Add email if configured
        if EMAIL_SERVER and EMAIL_USER and EMAIL_PASSWORD:
            sources.append("email")
            
        # Add Mattermost if configured
        if MATTERMOST_URL and MATTERMOST_TOKEN:
            sources.append("mattermost")
            
        # Add Odoo if configured
        if ODOO_HOST and ODOO_USER and ODOO_PASSWORD:
            sources.append("odoo")
            
        # Add SQL DB if configured
        if SQL_DB_HOST and SQL_DB_USER:
            sources.append("database")
            
        # Always add file-based sources
        sources.extend(["pdf", "text", "markdown", "json"])
        
        return sources
    
    @classmethod
    def get_file_extensions_by_type(cls) -> Dict[str, List[str]]:
        """Get mapping of document types to file extensions."""
        return {
            "text": ["txt", "csv", "tsv"],
            "pdf": ["pdf"],
            "markdown": ["md", "markdown"],
            "json": ["json"],
            "document": ["docx", "doc", "rtf", "odt"],
            "spreadsheet": ["xlsx", "xls", "ods"],
            "presentation": ["pptx", "ppt", "odp"],
            "email": ["eml", "msg"],
            "code": ["py", "js", "java", "cpp", "c", "cs", "php", "rb", "go", "rs", "ts"]
        }
    
    @classmethod
    def get_embedding_model_dimension(cls, provider: str, model: str) -> int:
        """Get embedding dimension for a specific model."""
        provider_models = EMBEDDING_MODELS.get(provider, {})
        return provider_models.get(model, EMBEDDING_DIMENSION)
    
    @classmethod
    def get_embedding_providers(cls) -> List[str]:
        """Get list of available embedding providers."""
        return list(EMBEDDING_MODELS.keys())
    
    @classmethod
    def get_embedding_models(cls, provider: str) -> List[str]:
        """Get list of available embedding models for a specific provider."""
        return list(EMBEDDING_MODELS.get(provider, {}).keys())