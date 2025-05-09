"""
Connectors for retrieving data from various sources.
"""

from typing import Dict, Any, List, Optional, Type
import logging

from .base import BaseConnector, Document
from .email_connector import EmailConnector
from .mattermost_connector import MattermostConnector
from .odoo_connector import OdooConnector
from .sql_connector import SQLConnector
from .file_connector import FileConnector

logger = logging.getLogger(__name__)


class ConnectorFactory:
    """Factory for creating connectors based on configuration."""
    
    _connector_types: Dict[str, Type[BaseConnector]] = {
        "email": EmailConnector,
        "mattermost": MattermostConnector,
        "odoo": OdooConnector,
        "database": SQLConnector,
        "file": FileConnector,
    }
    
    @classmethod
    def register_connector(cls, name: str, connector_class: Type[BaseConnector]) -> None:
        """
        Register a new connector type.
        
        Args:
            name: Name of the connector type
            connector_class: Connector class
        """
        cls._connector_types[name] = connector_class
    
    @classmethod
    def create_connector(cls, connector_type: str, config: Dict[str, Any]) -> Optional[BaseConnector]:
        """
        Create a connector instance.
        
        Args:
            connector_type: Type of connector to create
            config: Connector configuration
            
        Returns:
            Connector instance or None if type not found
        """
        if connector_type not in cls._connector_types:
            logger.error(f"Unknown connector type: {connector_type}")
            return None
        
        try:
            return cls._connector_types[connector_type](config)
        except Exception as e:
            logger.error(f"Error creating connector {connector_type}: {e}")
            return None
    
    @classmethod
    def get_available_connector_types(cls) -> List[str]:
        """
        Get list of available connector types.
        
        Returns:
            List of connector type names
        """
        return list(cls._connector_types.keys())


__all__ = [
    "BaseConnector", 
    "Document", 
    "ConnectorFactory",
    "EmailConnector", 
    "MattermostConnector", 
    "OdooConnector", 
    "SQLConnector", 
    "FileConnector"
]