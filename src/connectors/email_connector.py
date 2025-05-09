"""
Email connector for retrieving emails from IMAP, Gmail, or Exchange.
"""

import email
import imaplib
import asyncio
from datetime import datetime, timedelta
from email.header import decode_header
from typing import Any, Dict, List, Optional, Generator, Union
import re
import logging
from pathlib import Path
import tempfile

from ..connectors.base import BaseConnector, Document

logger = logging.getLogger(__name__)


class EmailConnector(BaseConnector):
    """Connector for retrieving emails from IMAP, Gmail, or Exchange."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the email connector.
        
        Args:
            config: Dictionary with the following keys:
                - provider: 'imap', 'gmail', or 'exchange'
                - server: IMAP server address (for IMAP)
                - port: IMAP server port (for IMAP)
                - username: Email username/address
                - password: Email password or app password
                - folders: List of folders to search
                - use_ssl: Whether to use SSL
                - filter: IMAP search criteria
                - max_emails: Maximum number of emails to retrieve
                - include_attachments: Whether to include attachments
        """
        super().__init__(config)
        self.provider = config.get("provider", "imap").lower()
        self.server = config.get("server")
        self.port = config.get("port", 993)
        self.username = config.get("username")
        self.password = config.get("password")
        self.folders = config.get("folders", ["INBOX"])
        self.use_ssl = config.get("use_ssl", True)
        self.filter = config.get("filter", "ALL")
        self.max_emails = config.get("max_emails", 100)
        self.include_attachments = config.get("include_attachments", True)
        self.connection = None
        self.temp_dir = None
        
        # Validate configuration
        if self.provider == "imap" and not all([self.server, self.username, self.password]):
            raise ValueError("IMAP provider requires server, username, and password")
        elif self.provider == "gmail" and not all([self.username, self.password]):
            self.server = "imap.gmail.com"
            self.port = 993
            self.use_ssl = True
        elif self.provider == "exchange" and not all([self.username, self.password]):
            raise ValueError("Exchange provider requires username and password")
    
    async def connect(self) -> bool:
        """
        Connect to the email server.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.provider in ["imap", "gmail"]:
            try:
                if self.use_ssl:
                    self.connection = imaplib.IMAP4_SSL(self.server, self.port)
                else:
                    self.connection = imaplib.IMAP4(self.server, self.port)
                
                self.connection.login(self.username, self.password)
                return True
            except Exception as e:
                logger.error(f"Failed to connect to email server: {e}")
                return False
        elif self.provider == "exchange":
            try:
                # For Exchange, we would use exchangelib here instead of imaplib
                # This is a placeholder for the exchangelib implementation
                from exchangelib import Credentials, Account
                credentials = Credentials(self.username, self.password)
                self.connection = Account(self.username, credentials=credentials, autodiscover=True)
                return True
            except Exception as e:
                logger.error(f"Failed to connect to Exchange server: {e}")
                return False
        else:
            logger.error(f"Unsupported email provider: {self.provider}")
            return False
    
    async def disconnect(self) -> None:
        """Close the connection to the email server."""
        if self.connection:
            if self.provider in ["imap", "gmail"]:
                try:
                    self.connection.logout()
                except Exception as e:
                    logger.error(f"Error during IMAP logout: {e}")
            self.connection = None
            
        # Clean up temporary directory if it exists
        if self.temp_dir:
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {e}")
    
    def _decode_email_subject(self, subject):
        """Decode email subject."""
        if subject is None:
            return ""
        decoded_parts = []
        for part, encoding in decode_header(subject):
            if isinstance(part, bytes):
                try:
                    if encoding:
                        decoded_parts.append(part.decode(encoding))
                    else:
                        decoded_parts.append(part.decode())
                except (UnicodeDecodeError, LookupError):
                    decoded_parts.append(part.decode('latin1', errors='replace'))
            else:
                decoded_parts.append(part)
        return "".join(decoded_parts)
    
    def _get_email_body(self, msg):
        """Extract email body from message."""
        text_content = ""
        html_content = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                # Skip attachments
                if "attachment" in content_disposition:
                    continue
                
                try:
                    # Get the body
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset()
                        if not charset:
                            charset = 'utf-8'
                            
                        try:
                            decoded_payload = payload.decode(charset)
                        except UnicodeDecodeError:
                            decoded_payload = payload.decode('latin1', errors='replace')
                        
                        if content_type == "text/plain":
                            text_content += decoded_payload
                        elif content_type == "text/html":
                            html_content += decoded_payload
                except Exception as e:
                    logger.error(f"Error extracting email body: {e}")
        else:
            # Not multipart - get the payload directly
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset()
                if not charset:
                    charset = 'utf-8'
                    
                try:
                    decoded_payload = payload.decode(charset)
                except UnicodeDecodeError:
                    decoded_payload = payload.decode('latin1', errors='replace')
                
                content_type = msg.get_content_type()
                if content_type == "text/plain":
                    text_content = decoded_payload
                elif content_type == "text/html":
                    html_content = decoded_payload
        
        # Prefer text content, fall back to HTML with basic HTML tags removed
        if text_content:
            return text_content
        elif html_content:
            # Basic HTML to text conversion
            html_content = re.sub(r'<[^>]+>', ' ', html_content)
            html_content = re.sub(r'\s+', ' ', html_content)
            return html_content
        else:
            return ""
    
    def _save_attachment(self, part, email_id):
        """Save email attachment to temporary directory."""
        if not self.temp_dir:
            self.temp_dir = tempfile.mkdtemp(prefix="email_attachments_")
            
        filename = part.get_filename()
        if filename:
            # Decode filename if needed
            filename_parts = decode_header(filename)
            filename = filename_parts[0][0]
            if isinstance(filename, bytes):
                filename = filename.decode(filename_parts[0][1] or 'utf-8', errors='replace')
            
            # Sanitize filename
            filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
            
            # Create full path
            file_path = Path(self.temp_dir) / f"{email_id}_{filename}"
            
            # Save attachment
            with open(file_path, 'wb') as f:
                f.write(part.get_payload(decode=True))
                
            return str(file_path)
        return None
    
    def _process_imap_email(self, email_data, email_id, folder):
        """Process a raw email from IMAP and convert to Document."""
        email_bytes = email_data[0][1]
        msg = email.message_from_bytes(email_bytes)
        
        # Extract headers
        subject = self._decode_email_subject(msg["Subject"])
        from_address = msg["From"]
        to_address = msg["To"]
        date_str = msg["Date"]
        
        # Parse date
        try:
            date = email.utils.parsedate_to_datetime(date_str)
        except (TypeError, ValueError):
            date = datetime.now()
        
        # Get email body
        body = self._get_email_body(msg)
        
        # Process attachments if required
        attachments = []
        if self.include_attachments:
            for part in msg.walk():
                content_disposition = str(part.get("Content-Disposition"))
                if "attachment" in content_disposition:
                    attachment_path = self._save_attachment(part, email_id)
                    if attachment_path:
                        attachments.append(attachment_path)
        
        # Combine email parts into document content
        content = f"Subject: {subject}\n"
        content += f"From: {from_address}\n"
        content += f"To: {to_address}\n"
        content += f"Date: {date_str}\n"
        content += f"\n{body}\n"
        
        # Metadata
        metadata = {
            "subject": subject,
            "from": from_address,
            "to": to_address,
            "date": date.isoformat(),
            "folder": folder,
            "has_attachments": len(attachments) > 0,
            "attachments": attachments,
            "email_id": email_id
        }
        
        return Document(
            content=content,
            metadata=metadata,
            source_type="email",
            source_id=email_id,
            created_at=date,
            updated_at=date
        )
    
    async def _get_emails_from_folder(self, folder, limit=None):
        """Get emails from a specific IMAP folder."""
        emails = []
        
        if self.provider in ["imap", "gmail"]:
            try:
                # Select the folder
                status, messages = self.connection.select(folder)
                if status != "OK":
                    logger.error(f"Failed to select folder {folder}: {messages}")
                    return emails
                
                # Search for emails
                status, message_ids = self.connection.search(None, self.filter)
                if status != "OK":
                    logger.error(f"Failed to search for emails in folder {folder}: {message_ids}")
                    return emails
                
                # Get email IDs and sort them (newest first)
                email_ids = message_ids[0].split()
                email_ids.reverse()  # Newest first
                
                # Apply limit
                if limit:
                    email_ids = email_ids[:limit]
                
                # Fetch emails
                for email_id in email_ids:
                    status, data = self.connection.fetch(email_id, "(RFC822)")
                    if status != "OK":
                        logger.warning(f"Failed to fetch email {email_id}: {data}")
                        continue
                    
                    # Process the email
                    try:
                        email_doc = self._process_imap_email(data, email_id.decode(), folder)
                        emails.append(email_doc)
                    except Exception as e:
                        logger.error(f"Error processing email {email_id}: {e}")
            except Exception as e:
                logger.error(f"Error retrieving emails from folder {folder}: {e}")
        
        return emails
    
    async def get_documents(self, query: Optional[str] = None, 
                          limit: Optional[int] = None,
                          **kwargs) -> List[Document]:
        """
        Retrieve emails as documents.
        
        Args:
            query: Optional search criteria to override the default filter
            limit: Maximum number of emails to retrieve per folder
            **kwargs: Additional parameters:
                - folders: List of folders to search (overrides instance folders)
                - start_date: Start date for email filtering
                - end_date: End date for email filtering
            
        Returns:
            List of Document objects containing emails
        """
        if not await self.connect():
            return []
        
        try:
            all_emails = []
            limit = limit or self.max_emails
            folders = kwargs.get("folders", self.folders)
            
            # Override filter if query is provided
            original_filter = self.filter
            if query:
                # Construct IMAP query based on query string
                self.filter = f'SUBJECT "{query}" OR BODY "{query}"'
            
            # Apply date filters if provided
            start_date = kwargs.get("start_date")
            end_date = kwargs.get("end_date")
            if start_date or end_date:
                date_filter = ""
                if start_date:
                    date_filter += f' SINCE "{start_date.strftime("%d-%b-%Y")}"'
                if end_date:
                    date_filter += f' BEFORE "{end_date.strftime("%d-%b-%Y")}"'
                self.filter = f'({self.filter}){date_filter}'
            
            # Process each folder
            for folder in folders:
                # Use asyncio to run the synchronous IMAP operations
                folder_emails = await asyncio.to_thread(
                    self._get_emails_from_folder, folder, limit // len(folders)
                )
                all_emails.extend(folder_emails)
                
                # Stop if we've reached the limit
                if len(all_emails) >= limit:
                    break
            
            # Restore original filter
            self.filter = original_filter
            
            return all_emails[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving emails: {e}")
            return []
        finally:
            await self.disconnect()
    
    async def stream_documents(self, query: Optional[str] = None, 
                             **kwargs) -> Generator[Document, None, None]:
        """
        Stream emails as documents.
        
        Args:
            query: Optional search criteria to override the default filter
            **kwargs: Additional parameters (same as get_documents)
            
        Yields:
            Document objects one at a time
        """
        documents = await self.get_documents(query, **kwargs)
        for doc in documents:
            yield doc
    
    async def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a specific email by ID.
        
        Args:
            doc_id: Email ID
            
        Returns:
            Document object if found, None otherwise
        """
        if not await self.connect():
            return None
        
        try:
            for folder in self.folders:
                status, messages = self.connection.select(folder)
                if status != "OK":
                    continue
                
                # Try to fetch the specific email
                status, data = self.connection.fetch(doc_id.encode(), "(RFC822)")
                if status == "OK" and data[0]:
                    return self._process_imap_email(data, doc_id, folder)
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving email by ID {doc_id}: {e}")
            return None
        finally:
            await self.disconnect()