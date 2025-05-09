"""
File uploader component for Streamlit UI.
"""

import os
import tempfile
import shutil
import time
from datetime import datetime
import streamlit as st
import requests
from typing import List, Dict, Any, Optional
import json
import mimetypes
import hashlib


def get_file_type(file_path: str) -> str:
    """
    Get file type from file path.
    
    Args:
        file_path: Path to file
        
    Returns:
        File type string
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type:
        main_type = mime_type.split('/')[0]
        if main_type == 'text':
            return 'text'
        elif main_type == 'application':
            if 'pdf' in mime_type:
                return 'pdf'
            elif 'json' in mime_type:
                return 'json'
            elif 'word' in mime_type or 'document' in mime_type:
                return 'document'
            elif 'presentation' in mime_type:
                return 'presentation'
            elif 'spreadsheet' in mime_type or 'excel' in mime_type:
                return 'spreadsheet'
            else:
                return 'other'
        else:
            return main_type
    
    # Fallback to extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower().lstrip('.')
    
    if ext in ['txt', 'text', 'log', 'csv', 'tsv']:
        return 'text'
    elif ext in ['pdf']:
        return 'pdf'
    elif ext in ['md', 'markdown']:
        return 'markdown'
    elif ext in ['json', 'jsonl']:
        return 'json'
    elif ext in ['docx', 'doc', 'rtf', 'odt']:
        return 'document'
    elif ext in ['pptx', 'ppt', 'odp']:
        return 'presentation'
    elif ext in ['xlsx', 'xls', 'ods']:
        return 'spreadsheet'
    elif ext in ['eml', 'msg']:
        return 'email'
    elif ext in ['html', 'htm']:
        return 'html'
    elif ext in ['xml']:
        return 'xml'
    else:
        return 'other'


def generate_file_id(filename: str, size: int) -> str:
    """
    Generate a unique ID for a file.
    
    Args:
        filename: Name of the file
        size: Size of the file in bytes
        
    Returns:
        Unique ID string
    """
    unique_string = f"{filename}_{size}_{time.time()}"
    return hashlib.md5(unique_string.encode()).hexdigest()


def file_uploader_component():
    """File uploader component."""
    st.subheader("Upload Documents")
    
    # Set up upload path
    upload_path = os.environ.get("UPLOAD_DIR", "/app/uploads")
    os.makedirs(upload_path, exist_ok=True)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload", 
        type=["txt", "pdf", "md", "json", "csv", "docx", "pptx", "xlsx", "eml", "html", "xml"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Dictionary to track file processing
        processed_files = {}
        
        # Process each file
        for uploaded_file in uploaded_files:
            file_id = generate_file_id(uploaded_file.name, uploaded_file.size)
            
            # Check if file is already processed
            if file_id in processed_files:
                continue
            
            # Check file size (limit to 10MB by default)
            max_size_mb = float(os.environ.get("MAX_DOCUMENT_SIZE_MB", "10"))
            max_size_bytes = max_size_mb * 1024 * 1024
            
            if uploaded_file.size > max_size_bytes:
                processed_files[file_id] = {
                    "filename": uploaded_file.name,
                    "size": uploaded_file.size,
                    "status": "error",
                    "error": f"File size exceeds the limit of {max_size_mb}MB"
                }
                continue
            
            try:
                # Save file to upload directory
                file_path = os.path.join(upload_path, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Get file type
                file_type = get_file_type(file_path)
                
                # Record file info
                processed_files[file_id] = {
                    "filename": uploaded_file.name,
                    "path": file_path,
                    "size": uploaded_file.size,
                    "type": file_type,
                    "status": "uploaded",
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                processed_files[file_id] = {
                    "filename": uploaded_file.name,
                    "size": uploaded_file.size,
                    "status": "error",
                    "error": str(e)
                }
        
        # Show results
        if processed_files:
            st.success(f"Uploaded {len([f for f in processed_files.values() if f['status'] == 'uploaded'])} files successfully")
            
            # Display uploaded files
            st.subheader("Uploaded Files")
            cols = st.columns(3)
            
            for i, (file_id, file_info) in enumerate(processed_files.items()):
                col = cols[i % 3]
                
                with col:
                    st.markdown(f"**{file_info['filename']}**")
                    st.markdown(f"Type: {file_info.get('type', 'Unknown')}")
                    st.markdown(f"Size: {file_info['size'] / 1024:.1f} KB")
                    
                    if file_info['status'] == 'error':
                        st.error(file_info['error'])
                    else:
                        st.success("Uploaded successfully")


def document_library_component():
    """Document library component."""
    st.subheader("Document Library")
    
    # API URL
    api_url = os.environ.get("API_URL", "http://localhost:8000")
    
    # Fetch document list
    try:
        response = requests.get(f"{api_url}/documents")
        
        if response.status_code == 200:
            documents = response.json()
            
            if not documents:
                st.info("No documents in the library yet. Upload some files to get started.")
                return
            
            # Filter options
            st.markdown("### Filter Documents")
            
            # Get unique file types
            file_types = sorted(set(doc.get("file_type", "unknown") for doc in documents))
            
            # Status options
            status_options = ["All", "Indexed", "Processing", "Pending", "Error"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_type = st.selectbox("File Type", ["All"] + file_types)
            
            with col2:
                selected_status = st.selectbox("Status", status_options)
            
            # Apply filters
            filtered_docs = documents
            
            if selected_type != "All":
                filtered_docs = [doc for doc in filtered_docs if doc.get("file_type") == selected_type]
            
            if selected_status != "All":
                status_filter = selected_status.lower()
                filtered_docs = [doc for doc in filtered_docs if doc.get("status") == status_filter]
            
            # Display documents
            if not filtered_docs:
                st.info("No documents match the selected filters.")
                return
            
            st.markdown(f"### Showing {len(filtered_docs)} Documents")
            
            # Create table
            data = []
            for doc in filtered_docs:
                data.append({
                    "Filename": doc.get("filename", "Unknown"),
                    "Type": doc.get("file_type", "Unknown"),
                    "Status": doc.get("status", "Unknown").capitalize(),
                    "Size (KB)": round(doc.get("file_size", 0) / 1024, 1),
                    "Chunks": doc.get("chunk_count", 0),
                    "Created": doc.get("created_at", "").split("T")[0]
                })
            
            st.dataframe(data, use_container_width=True)
            
        else:
            st.error(f"Error fetching documents: {response.status_code}")
    
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")


def document_management_page():
    """Document management page."""
    st.title("Document Management")
    
    # Tabs
    tabs = st.tabs(["Upload", "Library", "Status"])
    
    with tabs[0]:
        file_uploader_component()
    
    with tabs[1]:
        document_library_component()
    
    with tabs[2]:
        st.subheader("Processing Status")
        
        # API URL
        api_url = os.environ.get("API_URL", "http://localhost:8000")
        
        try:
            response = requests.get(f"{api_url}/documents/stats")
            
            if response.status_code == 200:
                stats = response.json()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Documents", stats.get("total", 0))
                
                with col2:
                    st.metric("Indexed", stats.get("indexed", 0))
                
                with col3:
                    st.metric("Processing", stats.get("processing", 0))
                
                with col4:
                    st.metric("Pending", stats.get("pending", 0))
                
                # Recent activity
                if "recent_activity" in stats:
                    st.subheader("Recent Activity")
                    
                    for activity in stats["recent_activity"]:
                        timestamp = activity.get("timestamp", "").replace("T", " ").split(".")[0]
                        action = activity.get("action", "")
                        details = activity.get("details", "")
                        
                        st.markdown(f"**{timestamp}**: {action} - {details}")
            
            else:
                st.error(f"Error fetching processing status: {response.status_code}")
        
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")


if __name__ == "__main__":
    st.set_page_config(page_title="Document Management", page_icon="📁", layout="wide")
    document_management_page()