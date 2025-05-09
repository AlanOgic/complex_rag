"""
Mattermost connector for retrieving messages from Mattermost channels.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Generator, Union

from ..connectors.base import BaseConnector, Document

logger = logging.getLogger(__name__)


class MattermostConnector(BaseConnector):
    """Connector for retrieving messages from Mattermost."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Mattermost connector.
        
        Args:
            config: Dictionary with the following keys:
                - url: Mattermost server URL (e.g., https://mattermost.example.com)
                - token: Personal access token
                - team: Team name
                - channels: List of channel names or IDs to fetch
                - max_messages: Maximum number of messages to retrieve per channel
                - include_replies: Whether to include message replies
                - include_files: Whether to include file attachments
                - days_ago: How many days back to fetch messages
        """
        super().__init__(config)
        self.url = config.get("url")
        self.token = config.get("token")
        self.team = config.get("team")
        self.channels = config.get("channels", [])
        self.max_messages = config.get("max_messages", 100)
        self.include_replies = config.get("include_replies", True)
        self.include_files = config.get("include_files", True)
        self.days_ago = config.get("days_ago", 30)
        self.client = None
        self.team_id = None
        self.channel_ids = {}
        
        # Validate configuration
        if not all([self.url, self.token, self.team]):
            raise ValueError("Mattermost connector requires url, token, and team")
    
    async def connect(self) -> bool:
        """
        Connect to the Mattermost server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Import here to avoid dependency if not using this connector
            from mattermostdriver import Driver
            
            # Create the client
            self.client = Driver({
                'url': self.url.rstrip('/'),
                'token': self.token,
                'scheme': 'https' if self.url.startswith('https') else 'http',
                'port': 443 if self.url.startswith('https') else 80,
                'basepath': '/api/v4',
                'verify': True,
                'timeout': 30,
            })
            
            # Test the connection
            await asyncio.to_thread(self.client.login)
            
            # Get the team ID
            teams = await asyncio.to_thread(self.client.teams.get_teams)
            for team in teams:
                if team['name'] == self.team or team['display_name'] == self.team:
                    self.team_id = team['id']
                    break
            
            if not self.team_id:
                logger.error(f"Team '{self.team}' not found")
                return False
            
            # Map channel names to IDs if channels are provided
            if self.channels:
                channels = await asyncio.to_thread(
                    self.client.channels.get_channels_for_user, 'me', self.team_id
                )
                
                for channel in channels:
                    # Store both name and display_name mapping to ID
                    if 'name' in channel:
                        self.channel_ids[channel['name']] = channel['id']
                    if 'display_name' in channel:
                        self.channel_ids[channel['display_name']] = channel['id']
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Mattermost: {e}")
            self.client = None
            return False
    
    async def disconnect(self) -> None:
        """Close the connection to the Mattermost server."""
        if self.client:
            try:
                await asyncio.to_thread(self.client.logout)
            except Exception as e:
                logger.error(f"Error during Mattermost logout: {e}")
            finally:
                self.client = None
    
    async def _get_channel_id(self, channel):
        """Get channel ID from name or return the ID if already an ID."""
        # If it's already in our mapping, return it
        if channel in self.channel_ids:
            return self.channel_ids[channel]
        
        # If it looks like an ID (26 character alphanumeric), use it directly
        if len(channel) == 26 and all(c.isalnum() for c in channel):
            return channel
        
        # Otherwise, try to find the channel
        try:
            channels = await asyncio.to_thread(
                self.client.channels.get_channels_for_user, 'me', self.team_id
            )
            
            for ch in channels:
                if ch.get('name') == channel or ch.get('display_name') == channel:
                    self.channel_ids[channel] = ch['id']
                    return ch['id']
        except Exception as e:
            logger.error(f"Error getting channel ID for {channel}: {e}")
        
        logger.warning(f"Channel '{channel}' not found")
        return None
    
    async def _get_user_info(self, user_id):
        """Get user information for a given user ID."""
        try:
            user = await asyncio.to_thread(self.client.users.get_user, user_id)
            return {
                'id': user['id'],
                'username': user.get('username', ''),
                'first_name': user.get('first_name', ''),
                'last_name': user.get('last_name', ''),
                'nickname': user.get('nickname', '')
            }
        except Exception as e:
            logger.error(f"Error getting user info for {user_id}: {e}")
            return {'id': user_id, 'username': 'unknown'}
    
    async def _get_replies(self, post_id):
        """Get replies to a specific post."""
        try:
            thread = await asyncio.to_thread(self.client.posts.get_thread, post_id)
            # Filter out the original post
            replies = [post for post_id, post in thread['posts'].items() if post_id != thread['order'][0]]
            return replies
        except Exception as e:
            logger.error(f"Error getting replies for post {post_id}: {e}")
            return []
    
    async def _get_file_info(self, file_id):
        """Get information about a file attachment."""
        try:
            file_info = await asyncio.to_thread(self.client.files.get_file_info, file_id)
            return {
                'id': file_id,
                'name': file_info.get('name', ''),
                'extension': file_info.get('extension', ''),
                'size': file_info.get('size', 0),
                'mime_type': file_info.get('mime_type', '')
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_id}: {e}")
            return {'id': file_id, 'name': 'unknown'}
    
    async def _process_message(self, post, channel_id, channel_name=None):
        """Process a Mattermost post and convert to a Document."""
        try:
            # Get user info
            user_info = await self._get_user_info(post['user_id'])
            
            # Convert timestamp to datetime
            created_at = datetime.fromtimestamp(post['create_at'] / 1000)
            updated_at = datetime.fromtimestamp(post['update_at'] / 1000) if post['update_at'] > 0 else None
            
            # Process message content
            content = post.get('message', '')
            
            # Get file attachments if needed
            files = []
            if self.include_files and 'file_ids' in post and post['file_ids']:
                for file_id in post['file_ids']:
                    file_info = await self._get_file_info(file_id)
                    files.append(file_info)
            
            # Get replies if needed
            replies = []
            if self.include_replies and not post.get('root_id'):  # Only for parent posts
                replies = await self._get_replies(post['id'])
            
            # Format document content
            doc_content = f"Channel: {channel_name or channel_id}\n"
            doc_content += f"User: {user_info.get('username', '')}"
            
            if user_info.get('first_name') or user_info.get('last_name'):
                doc_content += f" ({user_info.get('first_name', '')} {user_info.get('last_name', '')})"
            
            doc_content += f"\nDate: {created_at.isoformat()}\n\n"
            doc_content += content + "\n"
            
            # Add file info if available
            if files:
                doc_content += "\nAttachments:\n"
                for file in files:
                    doc_content += f"- {file.get('name', 'unnamed')} ({file.get('mime_type', 'unknown')})\n"
            
            # Add replies if available
            if replies:
                doc_content += "\nReplies:\n"
                for reply in replies:
                    reply_user = await self._get_user_info(reply['user_id'])
                    reply_time = datetime.fromtimestamp(reply['create_at'] / 1000)
                    doc_content += f"- {reply_user.get('username', '')}"
                    doc_content += f" [{reply_time.isoformat()}]: {reply.get('message', '')}\n"
            
            # Create metadata
            metadata = {
                'channel_id': channel_id,
                'channel_name': channel_name,
                'post_id': post['id'],
                'user_id': post['user_id'],
                'username': user_info.get('username', ''),
                'has_files': len(files) > 0,
                'files': files,
                'has_replies': len(replies) > 0,
                'reply_count': len(replies),
                'is_reply': bool(post.get('root_id')),
                'root_id': post.get('root_id', '')
            }
            
            # Create document
            return Document(
                content=doc_content,
                metadata=metadata,
                source_type="mattermost",
                source_id=post['id'],
                created_at=created_at,
                updated_at=updated_at
            )
        except Exception as e:
            logger.error(f"Error processing Mattermost message: {e}")
            return None
    
    async def _get_messages_from_channel(self, channel_id, channel_name, limit=None, query=None):
        """Get messages from a specific Mattermost channel."""
        messages = []
        limit = limit or self.max_messages
        
        try:
            # Calculate timestamp for filtering by date
            since = int((datetime.now() - timedelta(days=self.days_ago)).timestamp() * 1000)
            
            # Get posts from channel
            params = {
                'page': 0,
                'per_page': min(60, limit),  # Mattermost API usually limits to 60 per page
                'since': since
            }
            
            # Add search term if provided
            if query:
                search_params = {
                    'terms': query,
                    'is_or_search': True,
                    'time_zone_offset': 0,
                    'include_deleted_channels': False,
                    'page': 0,
                    'per_page': min(60, limit)
                }
                search_results = await asyncio.to_thread(
                    self.client.posts.search_posts, self.team_id, search_params
                )
                
                # Filter to posts from the specified channel
                posts = [post for _, post in search_results['posts'].items() 
                        if post['channel_id'] == channel_id]
                
                # Sort by create_at (newest first)
                posts.sort(key=lambda x: x['create_at'], reverse=True)
                
                # Apply limit
                posts = posts[:limit]
                
            else:
                # Get posts chronologically
                posts_result = await asyncio.to_thread(
                    self.client.posts.get_posts_for_channel, channel_id, params
                )
                
                posts = [posts_result['posts'][post_id] for post_id in posts_result['order']]
            
            # Process each post
            for post in posts:
                # Skip system messages
                if post.get('type', '') in ['system_join_channel', 'system_leave_channel']:
                    continue
                
                document = await self._process_message(post, channel_id, channel_name)
                if document:
                    messages.append(document)
                
                # Stop if we've reached the limit
                if len(messages) >= limit:
                    break
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting messages from channel {channel_id}: {e}")
            return []
    
    async def get_documents(self, query: Optional[str] = None, 
                          limit: Optional[int] = None,
                          **kwargs) -> List[Document]:
        """
        Retrieve Mattermost messages as documents.
        
        Args:
            query: Optional search term to filter messages
            limit: Maximum number of messages to retrieve per channel
            **kwargs: Additional parameters:
                - channels: List of channels to search (overrides instance channels)
                - days_ago: How many days back to fetch messages
                - include_replies: Whether to include message replies
            
        Returns:
            List of Document objects containing Mattermost messages
        """
        if not await self.connect():
            return []
        
        try:
            all_messages = []
            limit = limit or self.max_messages
            channels = kwargs.get("channels", self.channels)
            days_ago = kwargs.get("days_ago", self.days_ago)
            include_replies = kwargs.get("include_replies", self.include_replies)
            
            # Store original values to restore later
            orig_days_ago = self.days_ago
            orig_include_replies = self.include_replies
            
            # Set temporary values
            self.days_ago = days_ago
            self.include_replies = include_replies
            
            # If no channels specified, get all accessible channels
            if not channels:
                try:
                    channel_list = await asyncio.to_thread(
                        self.client.channels.get_channels_for_user, 'me', self.team_id
                    )
                    channels = [ch.get('name', ch['id']) for ch in channel_list]
                except Exception as e:
                    logger.error(f"Error getting channels: {e}")
                    return []
            
            per_channel_limit = limit // len(channels) if channels else limit
            
            # Process each channel
            for channel in channels:
                channel_id = await self._get_channel_id(channel)
                if not channel_id:
                    continue
                
                channel_messages = await self._get_messages_from_channel(
                    channel_id, channel, per_channel_limit, query
                )
                all_messages.extend(channel_messages)
                
                # Stop if we've reached the overall limit
                if len(all_messages) >= limit:
                    break
            
            # Restore original values
            self.days_ago = orig_days_ago
            self.include_replies = orig_include_replies
            
            return all_messages[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving Mattermost messages: {e}")
            return []
        finally:
            await self.disconnect()
    
    async def stream_documents(self, query: Optional[str] = None, 
                             **kwargs) -> Generator[Document, None, None]:
        """
        Stream Mattermost messages as documents.
        
        Args:
            query: Optional search term to filter messages
            **kwargs: Additional parameters (same as get_documents)
            
        Yields:
            Document objects one at a time
        """
        documents = await self.get_documents(query, **kwargs)
        for doc in documents:
            yield doc
    
    async def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a specific Mattermost message by ID.
        
        Args:
            doc_id: Post ID
            
        Returns:
            Document object if found, None otherwise
        """
        if not await self.connect():
            return None
        
        try:
            # Get the post
            post = await asyncio.to_thread(self.client.posts.get_post, doc_id)
            
            if post:
                # Get channel name
                channel_id = post['channel_id']
                channel_name = None
                
                try:
                    channel = await asyncio.to_thread(self.client.channels.get_channel, channel_id)
                    channel_name = channel.get('name', None)
                except Exception:
                    pass
                
                # Process the post
                return await self._process_message(post, channel_id, channel_name)
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving Mattermost post by ID {doc_id}: {e}")
            return None
        finally:
            await self.disconnect()