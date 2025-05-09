"""
Odoo connector for retrieving data from Odoo ERP system.
"""

import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Generator, Union
import asyncio

from ..connectors.base import BaseConnector, Document

logger = logging.getLogger(__name__)


class OdooConnector(BaseConnector):
    """Connector for retrieving data from Odoo ERP system."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Odoo connector.
        
        Args:
            config: Dictionary with the following keys:
                - host: Odoo host address
                - port: Odoo port (default: 8069)
                - database: Odoo database name
                - username: Odoo username
                - password: Odoo password
                - protocol: Connection protocol (jsonrpc or xmlrpc)
                - modules: List of modules to retrieve data from
                - models: List of specific models to retrieve
                - domain: Search domain for filtering records
                - limit: Maximum number of records per model
        """
        super().__init__(config)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 8069)
        self.database = config.get("database")
        self.username = config.get("username")
        self.password = config.get("password")
        self.protocol = config.get("protocol", "jsonrpc").lower()
        self.modules = config.get("modules", [])
        self.models = config.get("models", [])
        self.domain = config.get("domain", [])
        self.record_limit = config.get("limit", 100)
        self.connection = None
        self.uid = None
        
        # Map modules to models if modules are provided but models are not
        if self.modules and not self.models:
            self.models = self._get_default_models_for_modules()
    
    def _get_default_models_for_modules(self):
        """Get default models for specified modules."""
        module_to_models = {
            "sale": ["sale.order", "sale.order.line"],
            "purchase": ["purchase.order", "purchase.order.line"],
            "crm": ["crm.lead", "crm.team"],
            "project": ["project.project", "project.task"],
            "account": ["account.move", "account.payment"],
            "product": ["product.product", "product.template"],
            "hr": ["hr.employee", "hr.department"],
            "stock": ["stock.move", "stock.picking"],
            "mrp": ["mrp.production", "mrp.bom"],
            "pos": ["pos.order", "pos.session"],
        }
        
        models = []
        for module in self.modules:
            if module in module_to_models:
                models.extend(module_to_models[module])
        
        return models
    
    async def connect(self) -> bool:
        """
        Connect to the Odoo server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.protocol == "jsonrpc":
                # Use JSON-RPC
                import requests
                import json
                
                url = f"http://{self.host}:{self.port}/jsonrpc"
                payload = {
                    "jsonrpc": "2.0",
                    "method": "call",
                    "params": {
                        "service": "common",
                        "method": "login",
                        "args": [self.database, self.username, self.password]
                    },
                    "id": 1,
                }
                
                response = await asyncio.to_thread(
                    lambda: requests.post(url, json=payload).json()
                )
                
                if "result" in response:
                    self.uid = response["result"]
                    self.connection = url
                    return True
                else:
                    logger.error(f"Failed to connect to Odoo: {response.get('error')}")
                    return False
                
            elif self.protocol == "xmlrpc":
                # Use XML-RPC
                import xmlrpc.client
                
                common_url = f"http://{self.host}:{self.port}/xmlrpc/2/common"
                common_proxy = xmlrpc.client.ServerProxy(common_url)
                
                self.uid = await asyncio.to_thread(
                    common_proxy.authenticate,
                    self.database, self.username, self.password, {}
                )
                
                if self.uid:
                    object_url = f"http://{self.host}:{self.port}/xmlrpc/2/object"
                    self.connection = xmlrpc.client.ServerProxy(object_url)
                    return True
                else:
                    logger.error("Failed to authenticate with Odoo")
                    return False
            else:
                logger.error(f"Unsupported Odoo protocol: {self.protocol}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to Odoo: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close the connection to the Odoo server."""
        # Odoo doesn't require explicit disconnection for JSON-RPC or XML-RPC
        self.connection = None
        self.uid = None
    
    async def _execute_kw(self, model, method, args=None, kwargs=None):
        """Execute a method on an Odoo model."""
        args = args or []
        kwargs = kwargs or {}
        
        if not self.connection or not self.uid:
            if not await self.connect():
                return None
        
        try:
            if self.protocol == "jsonrpc":
                import requests
                import json
                
                payload = {
                    "jsonrpc": "2.0",
                    "method": "call",
                    "params": {
                        "service": "object",
                        "method": "execute_kw",
                        "args": [
                            self.database, self.uid, self.password,
                            model, method, args, kwargs
                        ]
                    },
                    "id": 1,
                }
                
                response = await asyncio.to_thread(
                    lambda: requests.post(self.connection, json=payload).json()
                )
                
                if "result" in response:
                    return response["result"]
                else:
                    logger.error(f"Odoo API error: {response.get('error')}")
                    return None
                
            elif self.protocol == "xmlrpc":
                result = await asyncio.to_thread(
                    self.connection.execute_kw,
                    self.database, self.uid, self.password,
                    model, method, args, kwargs
                )
                return result
            
        except Exception as e:
            logger.error(f"Error executing Odoo method {model}.{method}: {e}")
            return None
    
    async def _get_fields(self, model):
        """Get field definitions for an Odoo model."""
        fields_info = await self._execute_kw(
            model, 'fields_get', [], {'attributes': ['string', 'type', 'help']}
        )
        return fields_info
    
    async def _search_read(self, model, domain=None, fields=None, limit=None):
        """Search and read records from an Odoo model."""
        domain = domain or self.domain
        limit = limit or self.record_limit
        
        # If fields is None, get all text-like fields
        if fields is None:
            fields_info = await self._get_fields(model)
            fields = [
                field for field, info in fields_info.items()
                if info.get('type') in ['char', 'text', 'html', 'selection', 'many2one']
            ]
        
        records = await self._execute_kw(
            model, 'search_read',
            [domain], 
            {'fields': fields, 'limit': limit}
        )
        
        return records
    
    def _process_record(self, record, model):
        """Process an Odoo record into document content and metadata."""
        # Convert record dictionary to string representation
        record_content = []
        metadata = {"model": model, "id": record["id"]}
        
        for field, value in record.items():
            if field != "id":
                if isinstance(value, (dict, list)):
                    # Handle many2one fields (usually tuples like [id, name])
                    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], int):
                        field_value = value[1]
                    else:
                        field_value = json.dumps(value)
                else:
                    field_value = str(value)
                
                record_content.append(f"{field}: {field_value}")
                metadata[field] = value
        
        content = f"Model: {model}\n"
        content += f"ID: {record['id']}\n"
        content += "\n".join(record_content)
        
        # Extract dates for created_at and updated_at if available
        created_at = None
        updated_at = None
        
        if "create_date" in record:
            try:
                created_at = datetime.fromisoformat(record["create_date"].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        if "write_date" in record:
            try:
                updated_at = datetime.fromisoformat(record["write_date"].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        return Document(
            content=content,
            metadata=metadata,
            source_type="odoo",
            source_id=f"{model}_{record['id']}",
            created_at=created_at,
            updated_at=updated_at
        )
    
    async def get_documents(self, query: Optional[str] = None, 
                          limit: Optional[int] = None,
                          **kwargs) -> List[Document]:
        """
        Retrieve Odoo records as documents.
        
        Args:
            query: Optional query string to search for
            limit: Maximum number of records to retrieve per model
            **kwargs: Additional parameters:
                - models: List of specific models to retrieve
                - domain: Custom search domain
            
        Returns:
            List of Document objects containing Odoo records
        """
        if not await self.connect():
            return []
        
        try:
            all_documents = []
            limit = limit or self.record_limit
            models = kwargs.get("models", self.models)
            domain = kwargs.get("domain", self.domain)
            
            # Add query to domain if provided
            if query:
                # Create a domain filter that searches across text fields
                query_domain = []
                for field in ['name', 'description', 'note', 'comment']:
                    query_domain.append(['|', (field, 'ilike', query)])
                
                # Remove extra leading '|' operator
                if query_domain:
                    query_domain = query_domain[1:]
                
                # Combine with existing domain
                if domain:
                    domain = ['&'] + domain + query_domain
                else:
                    domain = query_domain
            
            per_model_limit = limit // len(models) if models else limit
            
            for model in models:
                records = await self._search_read(model, domain, limit=per_model_limit)
                
                if records:
                    for record in records:
                        document = self._process_record(record, model)
                        all_documents.append(document)
                
                # Stop if we've reached the overall limit
                if len(all_documents) >= limit:
                    break
            
            return all_documents[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving Odoo documents: {e}")
            return []
        finally:
            await self.disconnect()
    
    async def stream_documents(self, query: Optional[str] = None, 
                             **kwargs) -> Generator[Document, None, None]:
        """
        Stream Odoo records as documents.
        
        Args:
            query: Optional query string to search for
            **kwargs: Additional parameters (same as get_documents)
            
        Yields:
            Document objects one at a time
        """
        documents = await self.get_documents(query, **kwargs)
        for doc in documents:
            yield doc
    
    async def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a specific Odoo record by ID.
        
        Args:
            doc_id: Record ID in format "model_id" (e.g., "sale.order_42")
            
        Returns:
            Document object if found, None otherwise
        """
        if not await self.connect():
            return None
        
        try:
            # Parse the doc_id to get model and record ID
            if "_" not in doc_id:
                logger.error(f"Invalid Odoo document ID: {doc_id}")
                return None
            
            model, record_id = doc_id.rsplit("_", 1)
            try:
                record_id = int(record_id)
            except ValueError:
                logger.error(f"Invalid Odoo record ID: {record_id}")
                return None
            
            # Fetch the specific record
            records = await self._search_read(model, [("id", "=", record_id)])
            
            if records and len(records) > 0:
                return self._process_record(records[0], model)
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving Odoo record by ID {doc_id}: {e}")
            return None
        finally:
            await self.disconnect()