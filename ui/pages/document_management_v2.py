"""
Enhanced document management page with advanced features.
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
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


def get_api_url() -> str:
    """Get API URL from environment or use default."""
    return os.environ.get("API_URL", "http://localhost:8000")


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def file_uploader_v2_component():
    """Enhanced file uploader component."""
    st.subheader("Upload Documents")
    
    # API URL
    api_url = get_api_url()
    
    # Get processing configuration
    try:
        config_response = requests.get(f"{api_url}/documents/v2/config")
        if config_response.status_code == 200:
            config = config_response.json()
            
            with st.expander("Processing Configuration"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**General Settings**")
                    st.write(f"OCR Enabled: {'✅' if config.get('enable_ocr', True) else '❌'}")
                    st.write(f"Quality Check: {'✅' if config.get('enable_quality_check', True) else '❌'}")
                    st.write(f"Min Quality Score: {config.get('min_quality_score', 0.5)}")
                    st.write(f"Max Attempts: {config.get('max_processing_attempts', 3)}")
                    
                    st.write("**Chunking Settings**")
                    chunking = config.get("chunking_config", {})
                    st.write(f"Chunk Size: {chunking.get('chunk_size', 'N/A')}")
                    st.write(f"Chunk Overlap: {chunking.get('chunk_overlap', 'N/A')}")
                    st.write(f"Strategy: {chunking.get('strategy', 'N/A')}")
                
                with col2:
                    st.write("**OCR Settings**")
                    ocr = config.get("ocr_config", {})
                    st.write(f"OCR Engine: {ocr.get('ocr_engine', 'N/A')}")
                    st.write(f"Languages: {', '.join(ocr.get('languages', ['eng']))}")
                    st.write(f"Detect Tables: {'✅' if ocr.get('detect_tables', True) else '❌'}")
                    st.write(f"Preserve Layout: {'✅' if ocr.get('preserve_layout', True) else '❌'}")
                    
                    st.write("**Quality Settings**")
                    quality = config.get("quality_config", {})
                    st.write(f"Min Content Length: {quality.get('min_content_length', 'N/A')}")
                    st.write(f"Max Noise Ratio: {quality.get('max_noise_ratio', 'N/A')}")
    except Exception as e:
        st.warning(f"Could not fetch processing configuration: {e}")
    
    # Processing options
    auto_process = st.checkbox("Process documents immediately", value=True, 
                             help="If checked, documents will be processed as soon as they are uploaded")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload", 
        type=["txt", "pdf", "md", "json", "csv", "docx", "pptx", "xlsx", "eml", "html", "xml", "jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Dictionary to track uploaded files
        uploaded_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each file
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Uploading {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
            progress_bar.progress((i) / len(uploaded_files))
            
            try:
                # Upload file to API
                files = {"file": uploaded_file}
                params = {"auto_process": auto_process}
                
                response = requests.post(
                    f"{api_url}/documents/v2/",
                    files=files,
                    params=params
                )
                
                if response.status_code == 200:
                    doc_info = response.json()
                    
                    # Add to results
                    uploaded_results.append({
                        "id": doc_info.get("id", "unknown"),
                        "filename": doc_info.get("filename", uploaded_file.name),
                        "file_type": doc_info.get("file_type", get_file_type(uploaded_file.name)),
                        "file_size": doc_info.get("file_size", uploaded_file.size),
                        "status": doc_info.get("status", "unknown"),
                        "quality_score": doc_info.get("quality_score", None),
                        "success": True
                    })
                else:
                    # Record error
                    uploaded_results.append({
                        "filename": uploaded_file.name,
                        "file_size": uploaded_file.size,
                        "status": "error",
                        "error_message": f"API Error: {response.status_code}",
                        "success": False
                    })
                    
            except Exception as e:
                # Record error
                uploaded_results.append({
                    "filename": uploaded_file.name,
                    "file_size": uploaded_file.size,
                    "status": "error",
                    "error_message": str(e),
                    "success": False
                })
        
        # Complete progress
        progress_bar.progress(1.0)
        status_text.text(f"Uploaded {sum(1 for r in uploaded_results if r['success'])} of {len(uploaded_files)} files")
        
        # Show results
        st.subheader("Upload Results")
        
        # Count successes and failures
        success_count = sum(1 for r in uploaded_results if r["success"])
        failure_count = len(uploaded_results) - success_count
        
        if success_count > 0:
            st.success(f"Successfully uploaded {success_count} file{'s' if success_count != 1 else ''}")
        
        if failure_count > 0:
            st.error(f"Failed to upload {failure_count} file{'s' if failure_count != 1 else ''}")
        
        # Display as table
        result_df = pd.DataFrame(uploaded_results)
        if not result_df.empty:
            # Format for display
            display_df = result_df.copy()
            if "file_size" in display_df.columns:
                display_df["file_size"] = display_df["file_size"].apply(format_size)
            
            # Reorder columns
            display_cols = ["filename", "file_type", "file_size", "status", "quality_score"]
            display_cols = [col for col in display_cols if col in display_df.columns]
            if "error_message" in display_df.columns:
                display_cols.append("error_message")
            
            st.dataframe(display_df[display_cols], use_container_width=True)


def document_library_v2_component():
    """Enhanced document library component."""
    st.subheader("Document Library")
    
    # API URL
    api_url = get_api_url()
    
    # Get document statistics
    try:
        stats_response = requests.get(f"{api_url}/documents/v2/stats")
        
        if stats_response.status_code == 200:
            stats = stats_response.json()
            
            # Show metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total", stats.get("total", 0))
            
            with col2:
                st.metric("Indexed", stats.get("indexed", 0))
            
            with col3:
                st.metric("Processing", stats.get("processing", 0))
            
            with col4:
                st.metric("Pending", stats.get("pending", 0))
            
            with col5:
                st.metric("Error", stats.get("error", 0))
            
            # Document statistics visualizations
            st.subheader("Document Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                # Documents by type
                type_counts = stats.get("by_type", {})
                if type_counts:
                    fig_types = px.pie(
                        names=list(type_counts.keys()),
                        values=list(type_counts.values()),
                        title="Documents by Type"
                    )
                    st.plotly_chart(fig_types, use_container_width=True)
            
            with col2:
                # Documents by language
                lang_counts = stats.get("by_language", {})
                if lang_counts:
                    # Filter out None/unknown
                    filtered_langs = {k: v for k, v in lang_counts.items() if k and k.lower() != "unknown"}
                    if filtered_langs:
                        fig_langs = px.bar(
                            x=list(filtered_langs.keys()),
                            y=list(filtered_langs.values()),
                            title="Documents by Language",
                            labels={"x": "Language", "y": "Count"}
                        )
                        st.plotly_chart(fig_langs, use_container_width=True)
            
            # Quality statistics
            quality_stats = stats.get("quality_stats", {})
            if quality_stats:
                st.subheader("Quality Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create gauge for average quality
                    avg_quality = quality_stats.get("avg_quality", 0)
                    if avg_quality:
                        fig_quality = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=avg_quality,
                            title={"text": "Average Document Quality"},
                            gauge={
                                "axis": {"range": [0, 1]},
                                "bar": {"color": "darkblue"},
                                "steps": [
                                    {"range": [0, 0.5], "color": "lightcoral"},
                                    {"range": [0.5, 0.7], "color": "khaki"},
                                    {"range": [0.7, 1], "color": "lightgreen"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig_quality, use_container_width=True)
                
                with col2:
                    # Create gauge for OCR quality if available
                    avg_ocr = quality_stats.get("avg_ocr_quality", 0)
                    if avg_ocr:
                        fig_ocr = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=avg_ocr,
                            title={"text": "Average OCR Quality"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "darkblue"},
                                "steps": [
                                    {"range": [0, 50], "color": "lightcoral"},
                                    {"range": [50, 75], "color": "khaki"},
                                    {"range": [75, 100], "color": "lightgreen"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig_ocr, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Could not fetch document statistics: {e}")
    
    # Fetch document list
    try:
        response = requests.get(f"{api_url}/documents/v2/")
        
        if response.status_code == 200:
            documents = response.json()
            
            if not documents:
                st.info("No documents in the library yet. Upload some files to get started.")
                return
            
            # Create dataframe
            df = pd.DataFrame(documents)
            
            # Extract metadata columns if they exist
            if "metadata" in df.columns:
                df["source"] = df["metadata"].apply(lambda x: x.get("source", "unknown") if x else "unknown")
            
            # Filter options
            st.subheader("Filter Documents")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Status filter
                statuses = ["All"] + sorted(df["status"].unique().tolist())
                selected_status = st.selectbox("Status", statuses)
            
            with col2:
                # File type filter
                file_types = ["All"] + sorted(df["file_type"].unique().tolist())
                selected_file_type = st.selectbox("File Type", file_types)
            
            with col3:
                # Language filter if available
                if "language" in df.columns:
                    languages = ["All"] + sorted([l for l in df["language"].unique() if l])
                    selected_language = st.selectbox("Language", languages)
                else:
                    selected_language = "All"
            
            # Apply filters
            filtered_df = df.copy()
            
            if selected_status != "All":
                filtered_df = filtered_df[filtered_df["status"] == selected_status]
            
            if selected_file_type != "All":
                filtered_df = filtered_df[filtered_df["file_type"] == selected_file_type]
            
            if selected_language != "All" and "language" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["language"] == selected_language]
            
            # Show filter results
            st.subheader(f"Showing {len(filtered_df)} Documents")
            
            # Format dataframe for display
            if not filtered_df.empty:
                display_df = filtered_df.copy()
                
                # Format timestamps
                if "created_at" in display_df.columns:
                    display_df["created_at"] = pd.to_datetime(display_df["created_at"]).dt.strftime('%Y-%m-%d %H:%M')
                
                if "updated_at" in display_df.columns:
                    display_df["updated_at"] = pd.to_datetime(display_df["updated_at"]).dt.strftime('%Y-%m-%d %H:%M')
                
                # Format file size
                if "file_size" in display_df.columns:
                    display_df["file_size"] = display_df["file_size"].apply(format_size)
                
                # Set columns to display
                display_columns = ["filename", "file_type", "file_size", "status", "chunk_count"]
                if "quality_score" in display_df.columns:
                    display_df["quality_score"] = display_df["quality_score"].apply(lambda x: f"{x:.2f}" if x else "N/A")
                    display_columns.append("quality_score")
                
                if "language" in display_df.columns:
                    display_columns.append("language")
                
                display_columns.append("created_at")
                
                # Add actions column
                display_df["actions"] = "View Details"
                display_columns.append("actions")
                
                # Show table with clickable links
                clicked = st.dataframe(
                    display_df[display_columns],
                    column_config={
                        "actions": st.column_config.LinkColumn("Actions")
                    },
                    use_container_width=True
                )
            
                # Action buttons for bulk operations
                if not filtered_df.empty:
                    st.subheader("Bulk Actions")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Reindex Selected Documents", type="primary"):
                            st.info("Reindex functionality would go here")
                    
                    with col2:
                        if st.button("Delete Selected Documents", type="secondary"):
                            st.warning("Delete functionality would go here")
        
        else:
            st.error(f"Error fetching documents: {response.status_code}")
            
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")


def document_details_component(document_id: str):
    """Document details component."""
    api_url = get_api_url()
    
    try:
        # Fetch document details
        response = requests.get(f"{api_url}/documents/v2/{document_id}?include_history=true")
        
        if response.status_code == 200:
            document = response.json()
            
            # Display header
            st.header(f"Document: {document.get('filename', 'Unknown')}")
            
            # Status indicator
            status = document.get("status", "unknown")
            status_color = {
                "indexed": "green",
                "processing": "blue",
                "pending": "orange",
                "error": "red",
                "uploaded": "purple"
            }.get(status.lower(), "gray")
            
            st.markdown(f"<span style='color: white; background-color: {status_color}; padding: 4px 8px; border-radius: 4px;'>{status.upper()}</span>", unsafe_allow_html=True)
            
            # Basic information
            st.subheader("Basic Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**File Type:** {document.get('file_type', 'Unknown')}")
                st.markdown(f"**File Size:** {format_size(document.get('file_size', 0))}")
                if "language" in document and document["language"]:
                    st.markdown(f"**Language:** {document.get('language', 'Unknown')}")
            
            with col2:
                st.markdown(f"**Created:** {document.get('created_at', 'Unknown')}")
                st.markdown(f"**Updated:** {document.get('updated_at', 'Unknown')}")
                if "chunk_count" in document:
                    st.markdown(f"**Chunks:** {document.get('chunk_count', 0)}")
            
            with col3:
                if "quality_score" in document and document["quality_score"] is not None:
                    st.markdown(f"**Quality Score:** {document.get('quality_score', 0):.2f}")
                if "ocr_applied" in document:
                    st.markdown(f"**OCR Applied:** {'Yes' if document.get('ocr_applied', False) else 'No'}")
                if "ocr_quality" in document and document["ocr_quality"]:
                    st.markdown(f"**OCR Quality:** {document.get('ocr_quality', 0):.2f}")
            
            # Error message if any
            if "error" in document and document["error"]:
                st.error(f"Error: {document['error']}")
            
            # Document actions
            st.subheader("Document Actions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Download Document", type="primary"):
                    # This would trigger a download - not fully implemented in this example
                    st.success(f"Downloading {document.get('filename', 'document')}...")
            
            with col2:
                if st.button("Reindex Document"):
                    # Call reindex endpoint
                    try:
                        reindex_response = requests.post(f"{api_url}/documents/v2/{document_id}/reindex")
                        if reindex_response.status_code == 200:
                            st.success("Document queued for reindexing")
                        else:
                            st.error(f"Error reindexing document: {reindex_response.status_code}")
                    except Exception as e:
                        st.error(f"Error reindexing document: {e}")
            
            with col3:
                if st.button("Delete Document", type="secondary"):
                    # Confirm deletion
                    if st.checkbox("Confirm deletion"):
                        try:
                            delete_response = requests.delete(f"{api_url}/documents/v2/{document_id}")
                            if delete_response.status_code == 200:
                                st.success("Document deleted successfully")
                                st.button("Back to Library")
                                return
                            else:
                                st.error(f"Error deleting document: {delete_response.status_code}")
                        except Exception as e:
                            st.error(f"Error deleting document: {e}")
            
            # Processing information
            if "processing_info" in document and document["processing_info"]:
                st.subheader("Processing Information")
                
                processing_info = document["processing_info"]
                
                # Show stages completed
                if "stages_completed" in processing_info:
                    stages = processing_info["stages_completed"]
                    
                    # Create a progress indicator
                    all_stages = ["validation", "preprocessing", "extraction", "ocr", 
                                "quality_check", "chunking", "embedding", "indexing"]
                    
                    # Calculate progress percentage
                    progress_percentage = len(stages) / len(all_stages)
                    
                    # Show progress bar
                    st.progress(progress_percentage)
                    
                    # Show stages in columns
                    cols = st.columns(len(all_stages))
                    for i, stage in enumerate(all_stages):
                        with cols[i]:
                            completed = stage in stages
                            st.markdown(f"{'✅' if completed else '⬜'} {stage.capitalize()}")
                
                # Show processing metrics
                metrics_to_show = ["processing_time", "chunk_count", "quality_score"]
                metrics_data = {}
                
                for metric in metrics_to_show:
                    if metric in processing_info:
                        metrics_data[metric] = processing_info[metric]
                
                if metrics_data:
                    st.subheader("Processing Metrics")
                    cols = st.columns(len(metrics_data))
                    
                    for i, (metric, value) in enumerate(metrics_data.items()):
                        with cols[i]:
                            formatted_name = " ".join(word.capitalize() for word in metric.split("_"))
                            if metric == "processing_time":
                                st.metric(formatted_name, f"{value:.2f}s")
                            else:
                                st.metric(formatted_name, value)
            
            # Processing history
            if "processing_history" in document and document["processing_history"]:
                st.subheader("Processing History")
                
                history = document["processing_history"]
                
                # Create dataframe for history
                history_df = pd.DataFrame(history)
                
                # Format timestamps
                for col in ["started_at", "completed_at"]:
                    if col in history_df.columns:
                        history_df[col] = pd.to_datetime(history_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Display duration nicely
                if "duration" in history_df.columns:
                    history_df["duration"] = history_df["duration"].apply(lambda x: f"{x:.2f}s" if x else "")
                
                # Select columns to display
                display_cols = ["stage", "status", "started_at", "duration"]
                if "error" in history_df.columns:
                    display_cols.append("error")
                
                # Show as table
                st.dataframe(history_df[display_cols], use_container_width=True)
        
        else:
            st.error(f"Error fetching document details: {response.status_code}")
    
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")


def processing_metrics_component():
    """Processing metrics component."""
    st.subheader("Processing Metrics")
    
    api_url = get_api_url()
    
    # Time period selector
    time_period = st.selectbox(
        "Time Period",
        ["day", "week", "month", "all"],
        format_func=lambda x: {"day": "Last 24 Hours", "week": "Last 7 Days", 
                              "month": "Last 30 Days", "all": "All Time"}[x]
    )
    
    # Fetch metrics
    try:
        response = requests.get(f"{api_url}/documents/v2/metrics/processing_time?time_period={time_period}")
        
        if response.status_code == 200:
            metrics_data = response.json()
            
            if "metrics" in metrics_data and metrics_data["metrics"]:
                metrics = metrics_data["metrics"]
                
                # Processing time metrics
                st.subheader("Processing Times")
                
                # Prepare data for chart
                metric_names = []
                avg_times = []
                min_times = []
                max_times = []
                
                for name, data in metrics.items():
                    metric_names.append(name)
                    avg_times.append(data.get("avg", 0))
                    min_times.append(data.get("min", 0))
                    max_times.append(data.get("max", 0))
                
                # Create figure
                fig = go.Figure()
                
                # Add average time
                fig.add_trace(go.Bar(
                    name="Average Time",
                    x=metric_names,
                    y=avg_times,
                    marker_color='rgb(55, 83, 109)'
                ))
                
                # Add min/max range
                for i, name in enumerate(metric_names):
                    fig.add_trace(go.Scatter(
                        name='Min/Max Range',
                        x=[name, name],
                        y=[min_times[i], max_times[i]],
                        mode='lines',
                        marker=dict(color='rgba(255, 0, 0, 0.5)'),
                        showlegend=False
                    ))
                
                # Update layout
                fig.update_layout(
                    title=f"Processing Times ({metrics_data.get('time_period', 'recent')})",
                    xaxis_title="Processing Stage",
                    yaxis_title="Time (seconds)",
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show metrics in table format for detailed view
                st.subheader("Detailed Metrics")
                
                # Create table data
                table_data = []
                for name, data in metrics.items():
                    table_data.append({
                        "Metric": name,
                        "Average": f"{data.get('avg', 0):.2f}s",
                        "Minimum": f"{data.get('min', 0):.2f}s",
                        "Maximum": f"{data.get('max', 0):.2f}s",
                        "Count": data.get("count", 0)
                    })
                
                # Convert to dataframe and display
                if table_data:
                    metrics_df = pd.DataFrame(table_data)
                    st.dataframe(metrics_df, use_container_width=True)
            else:
                st.info(f"No metrics data available for the selected time period")
        else:
            st.error(f"Error fetching metrics: {response.status_code}")
    
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")


def document_management_v2_page():
    """Enhanced document management page."""
    st.title("Document Management V2")
    
    # Get query parameters for navigation
    query_params = st.experimental_get_query_params()
    
    # Default view is 'upload'
    current_view = query_params.get("view", ["upload"])[0]
    document_id = query_params.get("document_id", [None])[0]
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    selected_view = st.sidebar.radio(
        "Select View",
        ["Upload", "Library", "Processing Metrics"],
        index={"upload": 0, "library": 1, "metrics": 2}.get(current_view, 0)
    )
    
    # Update query parameters based on selection
    if selected_view == "Upload":
        st.experimental_set_query_params(view="upload")
        current_view = "upload"
    elif selected_view == "Library":
        st.experimental_set_query_params(view="library")
        current_view = "library"
    elif selected_view == "Processing Metrics":
        st.experimental_set_query_params(view="metrics")
        current_view = "metrics"
    
    # Special case: if viewing document details
    if current_view == "document" and document_id:
        document_details_component(document_id)
        if st.sidebar.button("Back to Library"):
            st.experimental_set_query_params(view="library")
            st.rerun()
    
    # Show selected view
    elif current_view == "upload":
        file_uploader_v2_component()
    elif current_view == "library":
        document_library_v2_component()
    elif current_view == "metrics":
        processing_metrics_component()


if __name__ == "__main__":
    st.set_page_config(page_title="Document Management V2", page_icon="📁", layout="wide")
    document_management_v2_page()