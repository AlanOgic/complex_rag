"""
SQL database connector for retrieving data from relational databases.
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Generator, Union, Tuple
import re

from ..connectors.base import BaseConnector, Document

logger = logging.getLogger(__name__)


class SQLConnector(BaseConnector):
    """Connector for retrieving data from SQL databases."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SQL database connector.
        
        Args:
            config: Dictionary with the following keys:
                - connection_string: SQLAlchemy connection string
                - tables: List of tables to retrieve data from
                - excluded_tables: List of tables to exclude
                - schema: Database schema to use
                - query_templates: Custom SQL queries to execute
                - max_rows: Maximum number of rows to retrieve per table
                - text_columns_only: Whether to retrieve only text columns
                - include_metadata: Whether to include table metadata
                - batch_size: Number of rows to process at once
        """
        super().__init__(config)
        self.connection_string = config.get("connection_string")
        self.tables = config.get("tables", [])
        self.excluded_tables = config.get("excluded_tables", [])
        self.schema = config.get("schema")
        self.query_templates = config.get("query_templates", {})
        self.max_rows = config.get("max_rows", 100)
        self.text_columns_only = config.get("text_columns_only", True)
        self.include_metadata = config.get("include_metadata", True)
        self.batch_size = config.get("batch_size", 1000)
        self.engine = None
        self.connection = None
        self.metadata = None
        
        # Validate configuration
        if not self.connection_string:
            raise ValueError("SQL connector requires a connection string")
    
    async def connect(self) -> bool:
        """
        Connect to the SQL database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Import here to avoid dependency if not using this connector
            from sqlalchemy import create_engine, MetaData, inspect
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncConnection
            
            # Check if it's an async connection string
            is_async = any(prefix in self.connection_string for prefix in 
                           ['postgresql+asyncpg', 'mysql+aiomysql', 'sqlite+aiosqlite'])
            
            if is_async:
                self.engine = create_async_engine(self.connection_string)
                self.connection = await self.engine.connect()
            else:
                # Create synchronous engine but use it asynchronously through to_thread
                self.engine = create_engine(self.connection_string)
                self.connection = await asyncio.to_thread(self.engine.connect)
            
            # Get metadata if needed
            if self.include_metadata:
                self.metadata = MetaData(schema=self.schema)
                
                if is_async:
                    await self.metadata.reflect(bind=self.engine)
                else:
                    await asyncio.to_thread(self.metadata.reflect, bind=self.engine)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close the connection to the database."""
        if self.connection:
            try:
                await self.connection.close()
            except Exception as e:
                logger.error(f"Error during database disconnect: {e}")
            finally:
                self.connection = None
                
        if self.engine:
            try:
                if hasattr(self.engine, 'dispose'):
                    if asyncio.iscoroutinefunction(self.engine.dispose):
                        await self.engine.dispose()
                    else:
                        await asyncio.to_thread(self.engine.dispose)
            except Exception as e:
                logger.error(f"Error during engine disposal: {e}")
            finally:
                self.engine = None
    
    async def _get_tables(self) -> List[str]:
        """Get list of tables from database."""
        try:
            from sqlalchemy import inspect
            
            # Determine if we're using async or sync engine
            if hasattr(self.engine, '_run_visitor'):  # Async engine
                inspector = await inspect(self.engine)
                tables = await inspector.get_table_names(schema=self.schema)
            else:  # Sync engine
                inspector = await asyncio.to_thread(inspect, self.engine)
                tables = await asyncio.to_thread(inspector.get_table_names, schema=self.schema)
            
            # Filter tables if needed
            if self.tables:
                # Only include specified tables
                tables = [t for t in tables if t in self.tables]
            elif self.excluded_tables:
                # Exclude specified tables
                tables = [t for t in tables if t not in self.excluded_tables]
            
            return tables
            
        except Exception as e:
            logger.error(f"Error getting tables: {e}")
            return []
    
    async def _get_columns(self, table: str) -> List[Dict[str, Any]]:
        """Get column information for a table."""
        try:
            from sqlalchemy import inspect
            
            # Determine if we're using async or sync engine
            if hasattr(self.engine, '_run_visitor'):  # Async engine
                inspector = await inspect(self.engine)
                columns = await inspector.get_columns(table, schema=self.schema)
            else:  # Sync engine
                inspector = await asyncio.to_thread(inspect, self.engine)
                columns = await asyncio.to_thread(
                    inspector.get_columns, table, schema=self.schema
                )
            
            # Filter text columns if needed
            if self.text_columns_only:
                # Only include columns that are likely to contain text
                text_columns = []
                for column in columns:
                    col_type = str(column['type']).lower()
                    if any(text_type in col_type for text_type in 
                           ['varchar', 'text', 'char', 'string', 'json', 'enum']):
                        text_columns.append(column)
                return text_columns
            else:
                return columns
                
        except Exception as e:
            logger.error(f"Error getting columns for table {table}: {e}")
            return []
    
    async def _execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute SQL query and return results."""
        try:
            if not self.connection:
                if not await self.connect():
                    return []
            
            params = params or {}
            
            # Determine if we're using async or sync connection
            if hasattr(self.connection, 'execute') and asyncio.iscoroutinefunction(self.connection.execute):
                # Async connection
                from sqlalchemy import text
                result = await self.connection.execute(text(query), params)
                # Convert results to dictionaries
                if hasattr(result, 'mappings') and callable(result.mappings):
                    rows = await result.mappings()
                    return [dict(row) for row in rows]
                else:
                    # Fallback for older SQLAlchemy
                    columns = result.keys()
                    rows = await result.fetchall()
                    return [dict(zip(columns, row)) for row in rows]
            else:
                # Sync connection
                result = await asyncio.to_thread(
                    lambda: self.connection.execute(query, params)
                )
                rows = await asyncio.to_thread(result.fetchall)
                
                # Convert to dictionaries
                if hasattr(result, 'keys') and callable(result.keys):
                    columns = await asyncio.to_thread(result.keys)
                    return [dict(zip(columns, row)) for row in rows]
                else:
                    # Fallback if keys() isn't available
                    return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error executing query: {e}\nQuery: {query}")
            return []
    
    async def _get_primary_key(self, table: str) -> List[str]:
        """Get primary key column(s) for a table."""
        try:
            from sqlalchemy import inspect
            
            # Determine if we're using async or sync engine
            if hasattr(self.engine, '_run_visitor'):  # Async engine
                inspector = await inspect(self.engine)
                pk = await inspector.get_pk_constraint(table, schema=self.schema)
            else:  # Sync engine
                inspector = await asyncio.to_thread(inspect, self.engine)
                pk = await asyncio.to_thread(
                    inspector.get_pk_constraint, table, schema=self.schema
                )
            
            return pk.get('constrained_columns', [])
            
        except Exception as e:
            logger.error(f"Error getting primary key for table {table}: {e}")
            return []
    
    def _build_select_query(self, table: str, columns: List[Dict], 
                           query: Optional[str] = None, limit: Optional[int] = None) -> str:
        """Build SQL SELECT query for a table."""
        # Use fully qualified table name if schema is specified
        table_name = f"{self.schema}.{table}" if self.schema else table
        
        # Get column names
        column_names = [column['name'] for column in columns]
        
        # Build basic query
        sql = f"SELECT {', '.join(column_names)} FROM {table_name}"
        
        # Add search condition if query is provided
        if query:
            conditions = []
            for column in columns:
                # Only use text-like columns for searching
                col_type = str(column['type']).lower()
                if any(text_type in col_type for text_type in 
                       ['varchar', 'text', 'char', 'string', 'json']):
                    conditions.append(f"{column['name']} LIKE '%{query}%'")
            
            if conditions:
                sql += f" WHERE {' OR '.join(conditions)}"
        
        # Add limit
        if limit:
            sql += f" LIMIT {limit}"
        
        return sql
    
    def _process_row(self, row: Dict, table: str, primary_keys: List[str]) -> Document:
        """Process a database row into a document."""
        # Convert row to string representation
        row_content = []
        
        for column, value in row.items():
            # Convert value to string
            if value is None:
                str_value = "NULL"
            elif isinstance(value, (dict, list)):
                try:
                    str_value = json.dumps(value)
                except (TypeError, ValueError):
                    str_value = str(value)
            else:
                str_value = str(value)
            
            row_content.append(f"{column}: {str_value}")
        
        # Create content
        content = f"Table: {table}\n"
        for pk in primary_keys:
            if pk in row:
                content += f"{pk}: {row[pk]}\n"
        content += "\n"
        content += "\n".join(row_content)
        
        # Create source ID from primary keys if available
        if primary_keys and all(pk in row for pk in primary_keys):
            source_id = f"{table}_" + "_".join(str(row[pk]) for pk in primary_keys)
        else:
            # Fallback: hash the content
            import hashlib
            content_hash = hashlib.md5(content.encode()).hexdigest()
            source_id = f"{table}_{content_hash}"
        
        # Extract timestamps if available
        created_at = None
        updated_at = None
        
        for date_field in ['created_at', 'creation_date', 'date_created', 'insert_date']:
            if date_field in row and row[date_field]:
                try:
                    if isinstance(row[date_field], str):
                        created_at = datetime.fromisoformat(row[date_field].replace('Z', '+00:00'))
                    else:
                        created_at = row[date_field]
                    break
                except (ValueError, TypeError):
                    pass
        
        for date_field in ['updated_at', 'modification_date', 'last_update', 'date_modified']:
            if date_field in row and row[date_field]:
                try:
                    if isinstance(row[date_field], str):
                        updated_at = datetime.fromisoformat(row[date_field].replace('Z', '+00:00'))
                    else:
                        updated_at = row[date_field]
                    break
                except (ValueError, TypeError):
                    pass
        
        # Create metadata
        metadata = {
            "table": table,
            "schema": self.schema,
            "primary_keys": {pk: row.get(pk) for pk in primary_keys if pk in row},
            "row_data": row
        }
        
        return Document(
            content=content,
            metadata=metadata,
            source_type="database",
            source_id=source_id,
            created_at=created_at,
            updated_at=updated_at
        )
    
    async def get_documents(self, query: Optional[str] = None, 
                          limit: Optional[int] = None,
                          **kwargs) -> List[Document]:
        """
        Retrieve database rows as documents.
        
        Args:
            query: Optional search term to filter rows
            limit: Maximum number of rows to retrieve per table
            **kwargs: Additional parameters:
                - tables: List of specific tables to retrieve
                - custom_queries: Dictionary of custom SQL queries to execute
            
        Returns:
            List of Document objects containing database rows
        """
        if not await self.connect():
            return []
        
        try:
            all_documents = []
            limit = limit or self.max_rows
            tables = kwargs.get("tables", self.tables) or await self._get_tables()
            custom_queries = kwargs.get("custom_queries", self.query_templates)
            
            per_table_limit = limit // len(tables) if tables else limit
            
            # Process each table
            for table in tables:
                try:
                    # Get column information
                    columns = await self._get_columns(table)
                    
                    # Get primary key information
                    primary_keys = await self._get_primary_key(table)
                    
                    # Check if there's a custom query for this table
                    if table in custom_queries:
                        custom_query = custom_queries[table]
                        # Add WHERE clause for search if query is provided
                        if query:
                            # Simple approach - this might need customization per database
                            search_clause = " OR ".join(
                                f"{column['name']} LIKE '%{query}%'" 
                                for column in columns 
                                if 'name' in column and any(
                                    text_type in str(column.get('type', '')).lower() 
                                    for text_type in ['varchar', 'text', 'char', 'string', 'json']
                                )
                            )
                            
                            if search_clause:
                                if "WHERE" in custom_query.upper():
                                    custom_query += f" AND ({search_clause})"
                                else:
                                    custom_query += f" WHERE {search_clause}"
                        
                        # Add LIMIT if not already present
                        if "LIMIT" not in custom_query.upper():
                            custom_query += f" LIMIT {per_table_limit}"
                            
                        rows = await self._execute_query(custom_query)
                    else:
                        # Build and execute standard query
                        sql = self._build_select_query(table, columns, query, per_table_limit)
                        rows = await self._execute_query(sql)
                    
                    # Process rows into documents
                    for row in rows:
                        document = self._process_row(row, table, primary_keys)
                        all_documents.append(document)
                        
                except Exception as e:
                    logger.error(f"Error processing table {table}: {e}")
                    continue
                
                # Stop if we've reached the overall limit
                if len(all_documents) >= limit:
                    break
            
            # Process custom queries that aren't tied to specific tables
            for query_name, custom_query in custom_queries.items():
                if query_name not in tables and isinstance(custom_query, str):
                    try:
                        # Execute the custom query
                        rows = await self._execute_query(custom_query)
                        
                        # Process rows into documents
                        for row in rows:
                            # Use a simpler process for custom queries
                            content = f"Query: {query_name}\n\n"
                            content += "\n".join(f"{k}: {v}" for k, v in row.items())
                            
                            # Generate a unique ID for this row
                            import hashlib
                            content_hash = hashlib.md5(content.encode()).hexdigest()
                            source_id = f"query_{query_name}_{content_hash}"
                            
                            document = Document(
                                content=content,
                                metadata={"query": query_name, "row_data": row},
                                source_type="database_query",
                                source_id=source_id
                            )
                            all_documents.append(document)
                            
                            # Stop if we've reached the overall limit
                            if len(all_documents) >= limit:
                                break
                        
                    except Exception as e:
                        logger.error(f"Error executing custom query {query_name}: {e}")
                        continue
            
            return all_documents[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving database documents: {e}")
            return []
        finally:
            await self.disconnect()
    
    async def stream_documents(self, query: Optional[str] = None, 
                             **kwargs) -> Generator[Document, None, None]:
        """
        Stream database rows as documents.
        
        Args:
            query: Optional search term to filter rows
            **kwargs: Additional parameters (same as get_documents)
            
        Yields:
            Document objects one at a time
        """
        documents = await self.get_documents(query, **kwargs)
        for doc in documents:
            yield doc
    
    async def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a specific database row by ID.
        
        Args:
            doc_id: Document ID in format "table_pk1_pk2_..." or "query_name_hash"
            
        Returns:
            Document object if found, None otherwise
        """
        if not await self.connect():
            return None
        
        try:
            # Parse the doc_id to get table and primary key values
            if doc_id.startswith("query_"):
                # Custom query results can't be retrieved directly by ID
                logger.warning(f"Cannot retrieve custom query results by ID: {doc_id}")
                return None
            
            parts = doc_id.split('_')
            if len(parts) < 2:
                logger.error(f"Invalid database document ID: {doc_id}")
                return None
            
            table = parts[0]
            pk_values = parts[1:]
            
            # Get primary key columns for the table
            primary_keys = await self._get_primary_key(table)
            
            if len(primary_keys) != len(pk_values):
                logger.error(f"Mismatch between primary keys and values: {primary_keys} vs {pk_values}")
                return None
            
            # Build query to fetch the specific row
            conditions = []
            for i, pk in enumerate(primary_keys):
                # Try to handle different data types
                try:
                    # Try as integer
                    int_val = int(pk_values[i])
                    conditions.append(f"{pk} = {int_val}")
                except ValueError:
                    # Use as string
                    conditions.append(f"{pk} = '{pk_values[i]}'")
            
            # Use fully qualified table name if schema is specified
            table_name = f"{self.schema}.{table}" if self.schema else table
            
            # Build and execute query
            sql = f"SELECT * FROM {table_name} WHERE {' AND '.join(conditions)} LIMIT 1"
            rows = await self._execute_query(sql)
            
            if rows and len(rows) > 0:
                return self._process_row(rows[0], table, primary_keys)
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving database row by ID {doc_id}: {e}")
            return None
        finally:
            await self.disconnect()