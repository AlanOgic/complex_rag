"""
Streamlit UI for Complex RAG system.
"""

import streamlit as st
import requests
import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional

# Import components
from components.file_uploader import document_management_page

# Configure page
st.set_page_config(
    page_title="Complex RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        width: 80%;
    }
    .chat-message.user {
        background-color: #e6f3ff;
        border-left: 5px solid #2196F3;
        align-self: flex-end;
        margin-left: 20%;
    }
    .chat-message.bot {
        background-color: #f0f0f0;
        border-left: 5px solid #8bc34a;
        align-self: flex-start;
    }
    .source-box {
        background-color: #f8f9fa;
        border: 1px solid #dfe1e5;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .source-title {
        font-weight: bold;
        margin-bottom: 0.25rem;
    }
    div[data-testid="stExpander"] {
        border: none;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stExpander"] > div:first-child {
        border-radius: 8px;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)


def get_available_sources() -> List[str]:
    """
    Get list of available data sources from the API.
    
    Returns:
        List of source names
    """
    try:
        api_url = os.environ.get("API_URL", "http://localhost:8000")
        response = requests.get(f"{api_url}/sources")
        response.raise_for_status()
        data = response.json()
        return data.get("sources", [])
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []


def get_system_stats() -> Dict[str, Any]:
    """
    Get system statistics from the API.
    
    Returns:
        Dictionary with statistics
    """
    try:
        api_url = os.environ.get("API_URL", "http://localhost:8000")
        response = requests.get(f"{api_url}/stats")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return {}


def send_query(
    query: str,
    sources: Optional[List[str]] = None,
    use_reranking: bool = True,
    use_multi_query: bool = True,
    include_sources: bool = True,
    citation_format: str = "inline",
) -> Dict[str, Any]:
    """
    Send a query to the RAG API.
    
    Args:
        query: User query
        sources: List of sources to query
        use_reranking: Whether to use reranking
        use_multi_query: Whether to use query expansion
        include_sources: Whether to include sources in the answer
        citation_format: Format for citations
        
    Returns:
        API response
    """
    try:
        api_url = os.environ.get("API_URL", "http://localhost:8000")
        data = {
            "query": query,
            "sources": sources,
            "use_reranking": use_reranking,
            "use_multi_query": use_multi_query,
            "include_sources": include_sources,
            "citation_format": citation_format,
        }
        
        response = requests.post(f"{api_url}/query", json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return {"error": str(e)}


def display_sources(sources: List[Dict[str, Any]]):
    """
    Display source information in the UI.
    
    Args:
        sources: List of source dictionaries
    """
    if not sources:
        st.info("No sources were used for this answer.")
        return
    
    with st.expander("📚 Sources", expanded=False):
        for i, source in enumerate(sources):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.write(f"**Source {i+1}**")
                st.write(f"Type: {source.get('type', 'Unknown')}")
            
            with col2:
                if source.get("type") == "file":
                    st.write(f"File: {source.get('name', 'Unknown')}")
                    if "path" in source:
                        st.write(f"Path: {source['path']}")
                
                elif source.get("type") == "email":
                    st.write(f"Subject: {source.get('subject', 'Unknown')}")
                    st.write(f"From: {source.get('from', 'Unknown')}")
                    st.write(f"Date: {source.get('date', 'Unknown')}")
                
                elif source.get("type") == "odoo":
                    st.write(f"Model: {source.get('model', 'Unknown')}")
                    st.write(f"Record ID: {source.get('record_id', 'Unknown')}")
                
                elif source.get("type") == "mattermost":
                    st.write(f"Channel: {source.get('channel', 'Unknown')}")
                    st.write(f"User: {source.get('user', 'Unknown')}")
                
                elif source.get("type") == "database":
                    st.write(f"Table: {source.get('table', 'Unknown')}")
                    if "primary_keys" in source:
                        pk_str = ", ".join([f"{k}: {v}" for k, v in source["primary_keys"].items()])
                        st.write(f"Keys: {pk_str}")
            
            st.markdown("---")


def display_chunks(chunks: List[Dict[str, Any]]):
    """
    Display retrieved chunks in the UI.
    
    Args:
        chunks: List of chunk dictionaries
    """
    if not chunks:
        st.info("No chunks were retrieved for this query.")
        return
    
    with st.expander("🔍 Retrieved Chunks", expanded=False):
        for i, chunk in enumerate(chunks):
            st.markdown(f"### Chunk {i+1} (Score: {chunk['score']:.3f})")
            st.markdown(f"**Source Type:** {chunk['source_type']}")
            
            # Display metadata highlights
            metadata = chunk.get("metadata", {})
            if metadata:
                meta_cols = st.columns(3)
                
                if "file_name" in metadata:
                    meta_cols[0].markdown(f"**File:** {metadata['file_name']}")
                
                if "subject" in metadata:
                    meta_cols[0].markdown(f"**Subject:** {metadata['subject']}")
                
                if "from" in metadata:
                    meta_cols[1].markdown(f"**From:** {metadata['from']}")
                
                if "chunk_index" in metadata:
                    meta_cols[2].markdown(f"**Position:** {metadata['chunk_index'] + 1} of {metadata.get('chunk_count', '?')}")
            
            # Display chunk content
            st.text_area("Content", chunk["content"], height=200)
            st.markdown("---")


def chat_interface():
    """Main chat interface."""
    st.title("Complex RAG System")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "available_sources" not in st.session_state:
        st.session_state.available_sources = get_available_sources()
    
    # Sidebar for settings
    st.sidebar.title("Settings")
    
    # Source selection
    st.sidebar.subheader("Data Sources")
    
    selected_sources = []
    if st.session_state.available_sources:
        for source in st.session_state.available_sources:
            if st.sidebar.checkbox(f"Use {source}", value=True):
                selected_sources.append(source)
    else:
        st.sidebar.warning("No data sources available. Please index some data.")
    
    # Query settings
    st.sidebar.subheader("Query Settings")
    use_reranking = st.sidebar.checkbox("Use Reranking", value=True)
    use_multi_query = st.sidebar.checkbox("Use Query Expansion", value=True)
    include_sources = st.sidebar.checkbox("Include Citations", value=True)
    citation_format = st.sidebar.selectbox(
        "Citation Format",
        options=["inline", "footnote", "endnote"],
        index=0,
    )
    
    # Indexing section
    st.sidebar.subheader("Indexing")
    
    # Index all button
    if st.sidebar.button("Index All Sources"):
        with st.sidebar:
            with st.spinner("Indexing all sources..."):
                api_url = os.environ.get("API_URL", "http://localhost:8000")
                try:
                    response = requests.post(f"{api_url}/index")
                    response.raise_for_status()
                    result = response.json()
                    
                    # Show summary of indexed documents
                    total_indexed = sum(src["indexed_count"] for src in result.values())
                    st.success(f"Indexed {total_indexed} documents from {len(result)} sources")
                    
                    # Update available sources
                    st.session_state.available_sources = get_available_sources()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Individual source indexing
    st.sidebar.subheader("Index Specific Source")
    source_to_index = st.sidebar.selectbox(
        "Select Source",
        options=st.session_state.available_sources if st.session_state.available_sources else ["No sources available"],
    )
    
    index_limit = st.sidebar.slider("Max Documents", min_value=10, max_value=1000, value=100, step=10)
    
    if st.sidebar.button("Index Selected Source"):
        with st.sidebar:
            with st.spinner(f"Indexing {source_to_index}..."):
                api_url = os.environ.get("API_URL", "http://localhost:8000")
                try:
                    response = requests.post(f"{api_url}/index/{source_to_index}", params={"limit": index_limit})
                    response.raise_for_status()
                    result = response.json()
                    
                    st.success(f"Indexed {result['indexed_count']} documents from {source_to_index}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Display statistics
    st.sidebar.subheader("System Stats")
    if st.sidebar.button("Refresh Stats"):
        stats = get_system_stats()
        
        if stats:
            index_stats = stats.get("index_stats", {})
            
            # Display index stats
            st.sidebar.metric("Total Chunks", index_stats.get("total_chunks", 0))
            
            # Display source type counts
            source_type_counts = index_stats.get("source_type_counts", {})
            if source_type_counts:
                st.sidebar.markdown("**Source Distribution:**")
                for source_type, count in source_type_counts.items():
                    st.sidebar.text(f"- {source_type}: {count}")
    
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f'<div class="chat-message user"><div>{content}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot"><div>{content}</div></div>', unsafe_allow_html=True)
            
            # Display sources and chunks if available
            if "details" in message:
                display_sources(message["details"].get("sources", []))
                display_chunks(message["details"].get("chunks", []))
    
    # Chat input
    query = st.chat_input("Ask a question about your data...")
    
    if query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        st.markdown(f'<div class="chat-message user"><div>{query}</div></div>', unsafe_allow_html=True)
        
        # Process query
        with st.spinner("Thinking..."):
            response = send_query(
                query=query,
                sources=selected_sources if selected_sources else None,
                use_reranking=use_reranking,
                use_multi_query=use_multi_query,
                include_sources=include_sources,
                citation_format=citation_format,
            )
            
            if "error" in response:
                answer = f"Error: {response['error']}"
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.markdown(f'<div class="chat-message bot"><div>{answer}</div></div>', unsafe_allow_html=True)
            else:
                answer = response.get("answer", "No answer generated")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "details": {
                        "sources": response.get("sources", []),
                        "chunks": response.get("chunks", [])
                    }
                })
                
                # Display assistant message
                st.markdown(f'<div class="chat-message bot"><div>{answer}</div></div>', unsafe_allow_html=True)
                
                # Display sources and chunks
                display_sources(response.get("sources", []))
                display_chunks(response.get("chunks", []))


def system_status():
    """System status page."""
    st.title("System Status")
    
    # Get system stats
    stats = get_system_stats()
    
    if not stats:
        st.error("Failed to retrieve system statistics.")
        return
    
    # Display general stats
    index_stats = stats.get("index_stats", {})
    config = stats.get("config", {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", index_stats.get("total_chunks", 0))
    
    with col2:
        st.metric("Embedding Dimension", index_stats.get("vector_size", 0))
    
    with col3:
        st.metric("Data Sources", len(stats.get("source_counts", {})))
    
    # Display embedding info
    st.subheader("Embedding Model")
    
    embedding_info = index_stats.get("embedding", {})
    
    if embedding_info:
        st.markdown(f"**Provider:** {embedding_info.get('provider', 'Unknown')}")
        st.markdown(f"**Model:** {embedding_info.get('model', 'Unknown')}")
        st.markdown(f"**Dimension:** {embedding_info.get('dimension', 0)}")
    else:
        st.info("No embedding information available.")
    
    # Display source distribution
    st.subheader("Source Distribution")
    
    source_type_counts = index_stats.get("source_type_counts", {})
    
    if source_type_counts:
        data = pd.DataFrame([
            {"Source Type": source_type, "Count": count}
            for source_type, count in source_type_counts.items()
        ])
        
        st.bar_chart(data, x="Source Type", y="Count")
    else:
        st.info("No indexed documents yet.")
    
    # Display configuration
    st.subheader("System Configuration")
    
    if config:
        configs = [
            {"Setting": "Max Chunks", "Value": config.get("max_chunks", "N/A")},
            {"Setting": "Similarity Threshold", "Value": config.get("similarity_threshold", "N/A")},
            {"Setting": "Reranking Enabled", "Value": "Yes" if config.get("use_reranking") else "No"},
            {"Setting": "Embedding Model", "Value": config.get("embedding_model", "N/A")},
        ]
        
        st.table(pd.DataFrame(configs))
    else:
        st.info("No configuration information available.")


def main():
    """Main application."""
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Chat", "Documents", "System Status"])
    
    # Display selected page
    if page == "Chat":
        chat_interface()
    elif page == "Documents":
        document_management_page()
    elif page == "System Status":
        system_status()


if __name__ == "__main__":
    main()