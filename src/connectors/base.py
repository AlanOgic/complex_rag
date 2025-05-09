"""
Base connector interface for retrieving data from various sources.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Generator, Union
from pydantic import BaseModel


class Document(BaseModel):
    """Representation of a document from any source."""
    
    content: str
    metadata: Dict[str, Any]
    source_type: str
    source_id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    

class BaseConnector(ABC):
    """Base class for all data source connectors."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize connector with configuration."""
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass
    
    @abstractmethod
    async def get_documents(self, query: Optional[str] = None, 
                          limit: Optional[int] = None,
                          **kwargs) -> List[Document]:
        """
        Retrieve documents from the data source.
        
        Args:
            query: Optional query to filter documents
            limit: Maximum number of documents to retrieve
            **kwargs: Additional source-specific parameters
            
        Returns:
            List of Document objects
        """
        pass
    
    @abstractmethod
    async def stream_documents(self, query: Optional[str] = None, 
                             **kwargs) -> Generator[Document, None, None]:
        """
        Stream documents from the data source.
        
        Args:
            query: Optional query to filter documents
            **kwargs: Additional source-specific parameters
            
        Yields:
            Document objects one at a time
        """
        pass
    
    @abstractmethod
    async def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a specific document by ID.
        
        Args:
            doc_id: Unique identifier for the document
            
        Returns:
            Document object if found, None otherwise
        """
        pass
    
    async def validate_connection(self) -> bool:
        """
        Test if connection to the data source is valid.
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            return await self.connect()
        except Exception:
            return False
        finally:
            await self.disconnect()
    
    def get_source_info(self) -> Dict[str, Any]:
        """
        Get information about this data source.
        
        Returns:
            Dictionary with connector metadata
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "config": {k: v for k, v in self.config.items() if k not in ["password", "token", "api_key"]}
        }