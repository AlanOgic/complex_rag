# Complex RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system that integrates multiple data sources for enhanced contextual AI responses.

## Features

- **Multi-Source Integration**: Connect to emails, Odoo, Mattermost, SQL databases, and various document formats (PDFs, Markdown, TXT, JSON)
- **Vector Database**: Qdrant for efficient vector similarity search 
- **Graph-Based Retrieval**: Advanced retrieval with relationship tracking
- **Hybrid Search**: Combines vector similarity, keyword matching, and semantic understanding
- **Advanced Document Processing**: Multi-stage pipeline with quality assessment, OCR, and intelligent chunking
- **Multiple Embedding Options**: Support for various embedding providers (SentenceTransformers, OpenAI, Cohere, HuggingFace)
- **Re-ranking**: Cross-encoder based re-ranking for higher quality results
- **Source Attribution**: Automatic citations and source tracking
- **Multi-Query Expansion**: Generates variations of queries for better recall
- **Document Quality Analysis**: Automatic assessment and enhancement of content quality
- **OCR Integration**: Extract text from images and scanned PDFs
- **Docker Support**: Full containerization for easy deployment

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Data Sources  │────▶│  RAG Pipeline   │────▶│    LLM Model    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  - Emails       │     │  - Indexing     │     │  - Generation   │
│  - Odoo ERP     │     │  - OCR          │     │  - Citations    │
│  - Mattermost   │     │  - Quality Check│     │  - Formatting   │
│  - Databases    │     │  - Chunking     │     │  - Streaming    │
│  - Documents    │     │  - Embedding    │     └─────────────────┘
└─────────────────┘     │  - Retrieval    │
                        │  - Re-ranking   │
                        └─────────────────┘
```

## Document Processing Pipeline

The system includes a sophisticated document processing pipeline:

1. **Document Validation**: Initial check of file format and content
2. **Preprocessing**: Prepare document for extraction
3. **Content Extraction**: Extract text from various formats
4. **OCR Processing**: Apply OCR to images and scanned documents
5. **Quality Assessment**: Evaluate document quality and enhance if needed
6. **Chunking**: Intelligently split document into manageable pieces
7. **Embedding**: Generate vector representations for each chunk
8. **Indexing**: Store in the vector database for retrieval

## Getting Started

### Prerequisites

- Docker and Docker Compose
- API keys for LLM providers (OpenAI, Anthropic, or Cohere)
- Credentials for data sources you want to connect

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/complex_rag.git
   cd complex_rag
   ```

2. Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```

3. Edit the `.env` file to add your API keys and configuration.

4. Start the system using Docker Compose:
   ```bash
   docker-compose up -d
   ```

5. Access the UI at http://localhost:8501

## Configuration

The system can be configured through the `.env` file or by modifying `config/settings.py`. Key configuration options include:

- **Vector Database**: Connection details for Qdrant
- **SQL Database**: Connection details for PostgreSQL 
- **Data Sources**: Credentials for emails, Odoo, Mattermost
- **Embedding Model**: Provider, model name and parameters
- **Document Processing**: OCR, quality thresholds, chunking strategies
- **LLM Settings**: Provider, model, and parameters
- **Retrieval Settings**: Chunk limits, thresholds, reranking

## Data Sources

### Emails

Connect to email servers using IMAP, Gmail API, or Exchange protocols.

### Odoo ERP

Retrieve records from Odoo modules including CRM, Sales, Projects, etc.

### Mattermost

Connect to Mattermost teams and channels for retrieving communication history.

### Databases

Query SQL databases (PostgreSQL, MySQL, SQLite) for structured data.

### File Storage

Process various document formats including:
- PDFs
- Markdown files
- Text files
- JSON data
- Images (via OCR)
- Office documents
- And more...

## System Components

### Connectors

Components that connect to and retrieve data from various sources.

### Processors

Handle document parsing, OCR, quality assessment, chunking, and metadata extraction.

### Indexers

Manage vector embeddings and storage in the vector database.

### Retrievers

Implement retrieval strategies including hybrid search and re-ranking.

### Models

Interface with language models for generating answers and embedding models for vectorization.

### API

FastAPI-based REST API for interacting with the system.

### UI

Streamlit-based user interface for searching and administration.

## Advanced Features

### Document Quality Analysis

The system automatically analyzes document quality, including:
- Content quality scoring
- Noise detection and filtering
- Structure analysis
- Language detection
- Content deduplication
- Entity identification

### OCR Processing

Extract text from images and scanned documents with:
- PDF text extraction with OCR fallback
- Image text extraction
- Table detection and processing
- Layout preservation
- Multi-language support

### Multiple Embedding Options

Support for various embedding providers:
- SentenceTransformers (local)
- OpenAI
- Cohere
- HuggingFace

### Custom Chunking Strategies

Different chunking strategies for different document types:
- Fixed size
- Sentence-based
- Paragraph-based
- Semantic-based

## Docker Deployment

The system uses Docker Compose with the following services:

- **rag-api**: Main RAG application
- **qdrant**: Vector database
- **postgres**: Traditional database
- **rag-ui**: Streamlit UI
- **minio**: Object storage for documents
- **document-processor**: Asynchronous document processing service
- **monitoring**: Prometheus and Grafana for system monitoring

## License

This project is licensed under the MIT License - see the LICENSE file for details.