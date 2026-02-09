"""
================================================================================
kg2sftui: Streamlit UI for Knowledge Graph to SFT Training Data Generator
================================================================================

Purpose:
    Provides a web-based UI for converting Knowledge Graphs (GraphML format) 
    into high-quality training data for Small Language Model (SLM) fine-tuning.

Author: Eason Lai
Date: February 9, 2026
Version: 1.0.0

Features:
    - Upload GraphML files via drag-and-drop
    - Interactive graph network visualization using PyVis
    - Configurable Azure OpenAI settings in sidebar
    - Configurable generation parameters (count, temperature, domain, etc.)
    - Real-time progress tracking during generation
    - Quality metrics and cost reporting
    - Download generated training data in JSONL/JSON formats

Domain Templates:
    - generic:        General-purpose knowledge graph Q&A generation
    - beauty_product: Beauty products domain (brands, products, ingredients, skincare)
    - beauty_makeup:  Makeup consultation domain (skin types, undertones, finishes,
                     coverage, color theory, application techniques)

Usage:
    streamlit run kg2sftui.py

Dependencies:
    pip install streamlit pyvis networkx python-dotenv openai tqdm

Architecture:
    This UI is a separate frontend for the kg2sft.py command-line tool.
    It imports core components from kg2sft and provides a user-friendly
    web interface for the same functionality.

================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import json
import tempfile
import streamlit as st
import networkx as nx

# PyVis is optional - gracefully handle if not installed
try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

# Import core components from kg2sft
from kg2sft import (
    Graph,
    GraphMLLoader,
    KGToTrainingData,
    AutoTuner,
    LLMConfig,
    GenerationConfig,
    ExtractionConfig,
    ValidationConfig,
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# Must be the first Streamlit command after imports
# Sets the browser tab title, favicon, and layout preferences

st.set_page_config(
    page_title="kg2sft - Knowledge Graph to SFT",
    page_icon="🔗",
    layout="wide",                    # Use full width of browser
    initial_sidebar_state="expanded"  # Sidebar open by default
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
# Inject custom CSS to enhance the visual appearance of the UI
# - Custom header styles for branding
# - Metric card styling for statistics display
# - Progress bar color customization

st.markdown("""
<style>
    /* Main title styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    /* Subtitle styling */
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    /* Card-style container for metrics */
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    /* Custom progress bar color */
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
# Streamlit reruns the script on every interaction, so we use session_state
# to persist data across reruns. Initialize all state variables here.

# --- Graph Data State ---
# Stores the loaded knowledge graph and related metadata
if 'graph' not in st.session_state:
    st.session_state.graph = None                    # The loaded Graph object
if 'graph_file_name' not in st.session_state:
    st.session_state.graph_file_name = None          # Name of the uploaded file
if 'dataset' not in st.session_state:
    st.session_state.dataset = None                  # Generated training dataset
if 'generation_complete' not in st.session_state:
    st.session_state.generation_complete = False     # Flag for generation status
if 'auto_tuner' not in st.session_state:
    st.session_state.auto_tuner = None               # AutoTuner instance (for report)

# --- Azure OpenAI Configuration State ---
# Credentials for Azure OpenAI API (empty by default for security)
if 'azure_api_key' not in st.session_state:
    st.session_state.azure_api_key = ""              # API key (kept secure)
if 'azure_endpoint' not in st.session_state:
    st.session_state.azure_endpoint = ""             # e.g., https://xxx.openai.azure.com/
if 'azure_deployment' not in st.session_state:
    st.session_state.azure_deployment = ""           # Model deployment name
if 'azure_api_version' not in st.session_state:
    st.session_state.azure_api_version = "2024-12-01-preview"  # API version

# --- Generation Settings State ---
# Parameters for training data generation (persisted across reruns)
if 'count' not in st.session_state:
    st.session_state.count = 10                      # Number of examples to generate
if 'domain' not in st.session_state:
    st.session_state.domain = "generic"              # Domain template (generic/beauty/beauty_makeup)
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7               # LLM sampling temperature
if 'quality_threshold' not in st.session_state:
    st.session_state.quality_threshold = 0.7         # Minimum quality score
if 'max_depth' not in st.session_state:
    st.session_state.max_depth = 999                 # Max path traversal depth
if 'dedup_threshold' not in st.session_state:
    st.session_state.dedup_threshold = 0.95          # Path similarity threshold
if 'sampling_strategy' not in st.session_state:
    st.session_state.sampling_strategy = "frequency_weighted"  # Sampling approach
if 'auto_mode' not in st.session_state:
    st.session_state.auto_mode = False                         # Auto-tuning mode toggle

# --- File Upload Tracking ---
# Prevents re-processing the same file on every Streamlit rerun
if 'last_uploaded_file_id' not in st.session_state:
    st.session_state.last_uploaded_file_id = None    # Unique ID of last processed file

# =============================================================================
# HELPER FUNCTIONS: GRAPH VISUALIZATION
# =============================================================================

@st.cache_data(show_spinner=False)
def create_graph_html(_graph: Graph, height: str = "600px") -> str:
    """
    Create PyVis network visualization HTML (cached for performance).
    
    Uses Streamlit's cache decorator to avoid regenerating the visualization
    on every rerun. The underscore prefix in _graph tells Streamlit not to
    hash this parameter (Graph objects aren't hashable).
    
    Args:
        _graph: The Graph object to visualize
        height: CSS height string for the visualization
        
    Returns:
        HTML string containing the interactive network visualization
    """
    net = create_pyvis_network_internal(_graph, height)
    
    # Generate HTML content by saving to temp file and reading back
    # PyVis doesn't support direct HTML string generation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        net.save_graph(f.name)
        with open(f.name, 'r', encoding='utf-8') as html_file:
            html_content = html_file.read()
        os.unlink(f.name)  # Clean up temp file
    
    return html_content


def create_pyvis_network_internal(graph: Graph, height: str = "600px") -> Network:
    """
    Create a PyVis Network object from a Graph object.
    
    This function configures the visual appearance and physics simulation
    of the network graph, including node colors, edge styles, and
    interactive features.
    
    Args:
        graph: The Graph object containing nodes and edges
        height: CSS height string for the network canvas
        
    Returns:
        Configured PyVis Network object ready for rendering
    """
    # -------------------------------------------------------------------------
    # Initialize PyVis Network with basic settings
    # -------------------------------------------------------------------------
    net = Network(
        height=height,
        width="100%",
        bgcolor="#ffffff",      # White background
        font_color="#333333",   # Dark gray text
        directed=True           # Show arrow direction on edges
    )
    
    # -------------------------------------------------------------------------
    # Configure Physics and Interaction Options
    # -------------------------------------------------------------------------
    # Using forceAtlas2Based solver for organic-looking layouts
    # Physics settings control how nodes repel/attract each other
    net.set_options("""
    {
        "nodes": {
            "font": {
                "size": 14,
                "face": "Arial"
            },
            "scaling": {
                "min": 10,
                "max": 30
            }
        },
        "edges": {
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            },
            "color": {
                "inherit": false,
                "color": "#848484"
            },
            "font": {
                "size": 10,
                "face": "Arial"
            },
            "smooth": {
                "enabled": true,
                "type": "continuous"
            }
        },
        "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "stabilization": {
                "enabled": true,
                "iterations": 200
            }
        },
        "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)
    
    # -------------------------------------------------------------------------
    # Calculate Node Degrees for Visual Sizing
    # -------------------------------------------------------------------------
    # Nodes with more connections will appear larger
    node_degrees = {}
    for source, target in graph.get_edges():
        node_degrees[source] = node_degrees.get(source, 0) + 1
        node_degrees[target] = node_degrees.get(target, 0) + 1
    
    max_degree = max(node_degrees.values()) if node_degrees else 1
    
    # -------------------------------------------------------------------------
    # Add Nodes with Visual Properties
    # -------------------------------------------------------------------------
    # Color palette for node variety (8 distinct colors)
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", 
              "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"]
    
    for node_id in graph.get_nodes():
        label = graph.get_node_label(node_id)
        degree = node_degrees.get(node_id, 0)
        # Scale node size: base 15px + up to 20px based on relative degree
        size = 15 + (degree / max_degree) * 20
        # Assign color based on hash of node_id for consistency
        color = colors[hash(node_id) % len(colors)]
        
        # Build HTML tooltip with node details
        attrs = graph.node_attributes.get(node_id, {})
        title = f"<b>{label}</b><br>"
        title += f"ID: {node_id}<br>"
        title += f"Connections: {degree}<br>"
        # Add any additional attributes (excluding keys used for label resolution)
        label_keys = {'name', 'label', 'title', 'display_name', 'text', 'value'}
        for key, value in attrs.items():
            if key not in label_keys:
                title += f"{key}: {value}<br>"
        
        net.add_node(
            node_id,
            label=label,
            title=title,      # Tooltip on hover
            size=size,
            color=color,
            font={"size": 12}
        )
    
    # -------------------------------------------------------------------------
    # Add Edges with Relationship Labels
    # -------------------------------------------------------------------------
    for source, target in graph.get_edges():
        relationship = graph.get_edge_relationship(source, target)
        edge_attrs = graph.edge_attributes.get((source, target), {})
        
        # Build HTML tooltip for edge
        title = f"<b>{relationship}</b><br>"
        # Exclude keys used for relationship resolution to avoid redundancy
        rel_keys = {'label', 'relationship_type', 'relationship', 'rel',
                    'type', 'edge_type', 'connection_type', 'relation', 'predicate'}
        for key, value in edge_attrs.items():
            if key not in rel_keys:
                title += f"{key}: {value}<br>"
        
        net.add_edge(
            source,
            target,
            label=relationship,  # Label shown on edge
            title=title,         # Tooltip on hover
            color="#848484"
        )
    
    return net


# =============================================================================
# HELPER FUNCTIONS: FALLBACK VISUALIZATION
# =============================================================================

def visualize_graph_fallback(graph: Graph) -> None:
    """
    Fallback visualization when PyVis is not available.
    
    Displays a simple text-based list of nodes and edges using
    Streamlit's native components. Limited to first 20 items each.
    
    Args:
        graph: The Graph object to display
    """
    st.subheader("📊 Graph Structure")
    
    # Get graph data
    nodes = graph.get_nodes()
    edges = graph.get_edges()
    
    # Display in two columns: nodes on left, edges on right
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Nodes:**")
        # Show first 20 nodes to avoid overwhelming the UI
        for node_id in nodes[:20]:
            label = graph.get_node_label(node_id)
            st.write(f"• {label}")
        if len(nodes) > 20:
            st.write(f"... and {len(nodes) - 20} more nodes")
    
    with col2:
        st.write("**Edges:**")
        # Show first 20 edges with relationship types
        for source, target in edges[:20]:
            source_label = graph.get_node_label(source)
            target_label = graph.get_node_label(target)
            rel = graph.get_edge_relationship(source, target)
            st.write(f"• {source_label} → [{rel}] → {target_label}")
        if len(edges) > 20:
            st.write(f"... and {len(edges) - 20} more edges")


# =============================================================================
# HELPER FUNCTIONS: GRAPH DISPLAY
# =============================================================================

def display_graph_visualization(graph: Graph) -> None:
    """
    Display the interactive graph visualization.
    
    Uses PyVis for rich interactive visualization if available,
    otherwise falls back to simple text display.
    
    Args:
        graph: The Graph object to visualize
    """
    # Check if PyVis is available
    if not HAS_PYVIS:
        st.warning("⚠️ PyVis not installed. Install with: `pip install pyvis`")
        visualize_graph_fallback(graph)
        return
    
    st.subheader("🔗 Interactive Graph Network")
    st.caption("Hover over nodes and edges for details. Use mouse to pan/zoom.")
    
    # Get cached HTML content (avoids regenerating on every rerun)
    html_content = create_graph_html(graph)
    
    # Embed the HTML visualization in an iframe
    st.components.v1.html(html_content, height=620, scrolling=True)


# =============================================================================
# HELPER FUNCTIONS: CONFIGURATION VALIDATION
# =============================================================================

def check_azure_configuration() -> tuple[bool, str]:
    """
    Validate Azure OpenAI configuration from session state.
    
    Checks that all required fields (API Key, Endpoint, Deployment)
    are filled in.
    
    Returns:
        Tuple of (is_valid: bool, message: str)
        - is_valid: True if all required fields are present
        - message: Status message or list of missing fields
    """
    missing = []
    if not st.session_state.azure_api_key:
        missing.append("API Key")
    if not st.session_state.azure_endpoint:
        missing.append("Endpoint")
    if not st.session_state.azure_deployment:
        missing.append("Deployment")
    
    if missing:
        return False, f"Missing: {', '.join(missing)}"
    
    return True, "Configuration OK"


# =============================================================================
# MAIN APPLICATION UI
# =============================================================================

def main() -> None:
    """
    Main application entry point.
    
    This function defines the complete Streamlit UI layout including:
    1. Header section with title and branding
    2. Sidebar with configuration options
    3. Main content area with file upload and graph display
    4. Generation section with progress tracking
    5. Results section with metrics and download options
    6. Footer
    """
    
    # =========================================================================
    # HEADER SECTION
    # =========================================================================
    st.markdown('<p class="main-header">🔗 kg2sft</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Knowledge Graph to SFT Training Data Generator</p>', unsafe_allow_html=True)
    
    # =========================================================================
    # SIDEBAR: CONFIGURATION
    # =========================================================================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # ---------------------------------------------------------------------
        # Azure OpenAI Settings (Collapsible)
        # ---------------------------------------------------------------------
        # Expanded by default if API key is not set (guides user to configure)
        with st.expander("🔑 Azure OpenAI Settings", expanded=not st.session_state.azure_api_key):
            st.session_state.azure_api_key = st.text_input(
                "API Key",
                value=st.session_state.azure_api_key,
                type="password",  # Masks input for security
                help="Your Azure OpenAI API Key"
            )
            
            st.session_state.azure_endpoint = st.text_input(
                "Endpoint",
                value=st.session_state.azure_endpoint,
                placeholder="https://your-resource.openai.azure.com/",
                help="Azure OpenAI endpoint URL (found in Azure Portal)"
            )
            
            st.session_state.azure_deployment = st.text_input(
                "Deployment Name",
                value=st.session_state.azure_deployment,
                placeholder="gpt-4.1-mini",
                help="Your model deployment name (e.g., gpt-4o, gpt-4.1-mini)"
            )
            
            st.session_state.azure_api_version = st.text_input(
                "API Version",
                value=st.session_state.azure_api_version,
                help="Azure OpenAI API version (usually doesn't need changing)"
            )
        
        # Display configuration status indicator
        env_ok, env_message = check_azure_configuration()
        if env_ok:
            st.success("✅ Azure OpenAI configured")
        else:
            st.warning(f"⚠️ {env_message}")
        
        st.divider()
        
        # ---------------------------------------------------------------------
        # Generation Settings
        # ---------------------------------------------------------------------
        st.subheader("Generation Settings")
        
        st.session_state.count = st.slider(
            "Number of examples",
            min_value=1,
            max_value=500,
            value=st.session_state.count,
            step=1,
            help="Number of training examples to generate"
        )
        
        domain_options = ["generic", "beauty_product", "beauty_makeup"]
        st.session_state.domain = st.selectbox(
            "Domain",
            options=domain_options,
            index=domain_options.index(st.session_state.domain),
            help="Domain template: 'generic' for general-purpose KG, 'beauty_product' for beauty products (brands/ingredients/skincare), 'beauty_makeup' for makeup consultation (skin types/finishes/techniques)"
        )
        
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="LLM sampling temperature (0=deterministic, 1=creative)",
            disabled=st.session_state.auto_mode  # Disabled in auto mode
        )
        
        st.divider()
        
        # ---------------------------------------------------------------------
        # Auto Mode Toggle
        # ---------------------------------------------------------------------
        st.session_state.auto_mode = st.toggle(
            "⚡ Auto-Tuning Mode",
            value=st.session_state.auto_mode,
            help="Automatically adjust parameters to reach the target count. "
                 "May trade off quality for quantity. A quality impact report "
                 "will be shown after generation."
        )
        
        if st.session_state.auto_mode:
            st.info(
                "🤖 **Auto mode enabled.** Parameters (quality threshold, "
                "dedup, temperature, sampling) will be automatically tuned "
                "across up to 6 iterations to reach your target count. "
                "Advanced settings below are ignored in this mode."
            )
        
        st.divider()
        
        # ---------------------------------------------------------------------
        # Advanced Settings (Collapsible)
        # ---------------------------------------------------------------------
        # These settings are for power users who want fine-grained control
        with st.expander("🔧 Advanced Settings", expanded=not st.session_state.auto_mode):
            if st.session_state.auto_mode:
                st.caption("⚠️ These settings are ignored in auto-tuning mode.")
            
            # Quality threshold: Examples below this score are rejected
            st.session_state.quality_threshold = st.slider(
                "Quality threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.quality_threshold,
                step=0.05,
                help="Minimum quality score (0-1) to accept examples",
                disabled=st.session_state.auto_mode
            )
            
            # Max path depth: How far to traverse in the graph
            st.session_state.max_depth = st.number_input(
                "Max path depth",
                min_value=1,
                max_value=999,
                value=st.session_state.max_depth,
                help="Maximum number of hops when extracting paths (999=unlimited)"
            )
            
            # Deduplication threshold: Similarity threshold for path deduplication
            st.session_state.dedup_threshold = st.slider(
                "Deduplication threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.dedup_threshold,
                step=0.05,
                help="Paths more similar than this threshold are considered duplicates",
                disabled=st.session_state.auto_mode
            )
            
            # Sampling strategy: How to select starting nodes for path extraction
            sampling_options = ["frequency_weighted", "random"]
            st.session_state.sampling_strategy = st.selectbox(
                "Sampling strategy",
                options=sampling_options,
                index=sampling_options.index(st.session_state.sampling_strategy),
                help="frequency_weighted: Prefer highly-connected nodes; random: Equal probability",
                disabled=st.session_state.auto_mode
            )
    
    # =========================================================================
    # MAIN CONTENT: FILE UPLOAD AND GRAPH DISPLAY
    # =========================================================================
    # Two-column layout: Upload on left, Statistics on right
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # -----------------------------------------------------------------
        # File Upload Section
        # -----------------------------------------------------------------
        st.subheader("📁 Upload Knowledge Graph")
        
        # File uploader widget for GraphML files
        uploaded_file = st.file_uploader(
            "Choose a GraphML file",
            type=["graphml"],
            help="Upload a knowledge graph in GraphML format"
        )
        
        # Quick-load buttons for sample files
        st.caption("Or use a sample file:")
        sample_col1, sample_col2, sample_col3 = st.columns(3)
        
        with sample_col1:
            # Load technology knowledge graph sample (generic domain)
            if st.button("Technology Graph", use_container_width=True):
                if os.path.exists("technology_knowledge.graphml"):
                    st.session_state.graph = GraphMLLoader.load("technology_knowledge.graphml")
                    st.session_state.graph_file_name = "technology_knowledge.graphml"
                    st.session_state.generation_complete = False
                    st.rerun()  # Refresh to show the loaded graph
                else:
                    st.error("Sample file not found")
        
        with sample_col2:
            # Load beauty products knowledge graph sample (beauty_product domain)
            if st.button("Beauty Products", use_container_width=True):
                if os.path.exists("beauty_products.graphml"):
                    st.session_state.graph = GraphMLLoader.load("beauty_products.graphml")
                    st.session_state.graph_file_name = "beauty_products.graphml"
                    st.session_state.generation_complete = False
                    st.rerun()  # Refresh to show the loaded graph
                else:
                    st.error("Sample file not found")
        
        with sample_col3:
            # Load makeup consultation knowledge graph sample (beauty_makeup domain)
            if st.button("Beauty Makeup", use_container_width=True):
                if os.path.exists("makeup_knowledge_graph.graphml"):
                    st.session_state.graph = GraphMLLoader.load("makeup_knowledge_graph.graphml")
                    st.session_state.graph_file_name = "makeup_knowledge_graph.graphml"
                    st.session_state.generation_complete = False
                    st.rerun()  # Refresh to show the loaded graph
                else:
                    st.error("Sample file not found")
        
        # -----------------------------------------------------------------
        # Handle Uploaded File
        # -----------------------------------------------------------------
        # Only process if it's a new file (prevents reprocessing on every rerun)
        if uploaded_file is not None:
            # Create unique file ID from name + size to detect changes
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_id != st.session_state.last_uploaded_file_id:
                # Save uploaded file to temp location for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix=".graphml") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Load the graph using the core GraphMLLoader
                    st.session_state.graph = GraphMLLoader.load(tmp_path)
                    st.session_state.graph_file_name = uploaded_file.name
                    st.session_state.generation_complete = False
                    st.session_state.last_uploaded_file_id = file_id
                    st.success(f"✅ Loaded: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
                finally:
                    # Clean up temp file
                    os.unlink(tmp_path)
            else:
                # File already processed, just show success message
                st.success(f"✅ Loaded: {uploaded_file.name}")
    
    with col2:
        # -----------------------------------------------------------------
        # Graph Statistics Section
        # -----------------------------------------------------------------
        if st.session_state.graph is not None:
            graph = st.session_state.graph
            
            st.subheader("📊 Graph Statistics")
            
            # Display key metrics in three columns
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.metric("Nodes", len(graph.get_nodes()))
            with stat_col2:
                st.metric("Edges", len(graph.get_edges()))
            with stat_col3:
                # Calculate average degree (edges per node)
                if graph.get_nodes():
                    # Multiply by 2 because each edge connects 2 nodes
                    avg_degree = len(graph.get_edges()) * 2 / len(graph.get_nodes())
                    st.metric("Avg Degree", f"{avg_degree:.1f}")
                else:
                    st.metric("Avg Degree", "0")
            
            # Button to clear the loaded graph and start fresh
            if st.button("🗑️ Clear Graph", use_container_width=True):
                # Reset all graph-related state
                st.session_state.graph = None
                st.session_state.graph_file_name = None
                st.session_state.dataset = None
                st.session_state.generation_complete = False
                st.session_state.last_uploaded_file_id = None
                st.rerun()
    
    # =========================================================================
    # GRAPH VISUALIZATION SECTION
    # =========================================================================
    # Display interactive network visualization if graph is loaded
    if st.session_state.graph is not None:
        st.divider()
        display_graph_visualization(st.session_state.graph)
    
    # =========================================================================
    # GENERATION SECTION
    # =========================================================================
    # Show generation controls only when a graph is loaded
    if st.session_state.graph is not None:
        st.divider()
        st.subheader("🚀 Generate Training Data")
        
        # Check if Azure OpenAI is configured
        if not env_ok:
            st.warning("⚠️ Please configure Azure OpenAI credentials before generating.")
        else:
            gen_col1, gen_col2 = st.columns([3, 1])
            
            with gen_col1:
                # Map domain names to human-readable descriptions (aligned with CLI)
                domain_descriptions = {
                    "generic": "generic (general-purpose KG)",
                    "beauty_product": "beauty_product (products, brands, ingredients, skincare)",
                    "beauty_makeup": "beauty_makeup (makeup consultation: skin types, finishes, techniques)"
                }
                domain_display = domain_descriptions.get(st.session_state.domain, st.session_state.domain)
                # Show summary of generation settings
                mode_label = "⚡ AUTO-TUNING" if st.session_state.auto_mode else "MANUAL"
                st.info(f"Ready to generate **{st.session_state.count}** training examples from **{st.session_state.graph_file_name}** using **{domain_display}** domain template. Mode: **{mode_label}**")
            
            with gen_col2:
                # Primary action button to start generation
                generate_button = st.button("🎯 Generate", type="primary", use_container_width=True)
            
            if generate_button:
                # -----------------------------------------------------------------
                # Execute Generation Pipeline
                # -----------------------------------------------------------------
                # Export in-memory graph to temp file (KGToTrainingData expects file path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".graphml") as tmp_file:
                    # Reconstruct NetworkX graph from our Graph object
                    nx_graph = nx.DiGraph()
                    for node_id in st.session_state.graph.get_nodes():
                        nx_graph.add_node(node_id, **st.session_state.graph.node_attributes.get(node_id, {}))
                    for source, target in st.session_state.graph.get_edges():
                        nx_graph.add_edge(source, target, **st.session_state.graph.edge_attributes.get((source, target), {}))
                    nx.write_graphml(nx_graph, tmp_file.name)
                    tmp_path = tmp_file.name
                
                try:
                    # Setup progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    if st.session_state.auto_mode:
                        # =============================================================
                        # AUTO-TUNING MODE
                        # =============================================================
                        status_text.text("⚡ Auto-tuning mode: analysing graph...")
                        progress_bar.progress(5)
                        
                        # Override Azure OpenAI env vars so AutoTuner picks them up
                        os.environ["AZURE_OPENAI_API_KEY"] = st.session_state.azure_api_key
                        os.environ["AZURE_OPENAI_ENDPOINT"] = st.session_state.azure_endpoint
                        os.environ["AZURE_OPENAI_DEPLOYMENT"] = st.session_state.azure_deployment
                        os.environ["AZURE_OPENAI_MODEL"] = st.session_state.azure_deployment
                        
                        tuner = AutoTuner(
                            graph_path=tmp_path,
                            domain=st.session_state.domain,
                            target_count=st.session_state.count,
                            max_depth=st.session_state.max_depth,
                        )
                        
                        status_text.text("🤖 Auto-tuning: generating training examples (this may take multiple iterations)...")
                        progress_bar.progress(10)
                        
                        dataset = tuner.run()
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Auto-tuning generation complete!")
                        
                        # Store tuner for quality impact report
                        st.session_state.dataset = dataset
                        st.session_state.auto_tuner = tuner
                        st.session_state.generation_complete = True
                    
                    else:
                        # =============================================================
                        # MANUAL MODE (default)
                        # =============================================================
                        status_text.text("🔧 Initializing pipeline...")
                        
                        # Create LLM configuration from user-provided settings
                        llm_config = LLMConfig(
                            api_key=st.session_state.azure_api_key,
                            api_endpoint=st.session_state.azure_endpoint,
                            deployment_name=st.session_state.azure_deployment,
                            api_version=st.session_state.azure_api_version,
                            model_name=st.session_state.azure_deployment
                        )
                        
                        # Initialize the main training data generator pipeline
                        trainer = KGToTrainingData(
                            graph_path=tmp_path,
                            llm_config=llm_config,
                            generation_config=GenerationConfig(
                                count=st.session_state.count,
                                temperature=st.session_state.temperature,
                                max_tokens=500
                            ),
                            extraction_config=ExtractionConfig(
                                max_hop_depth=st.session_state.max_depth,
                                sampling_strategy=st.session_state.sampling_strategy,
                                dedup_threshold=st.session_state.dedup_threshold
                            ),
                            validation_config=ValidationConfig(
                                quality_threshold=st.session_state.quality_threshold,
                                min_length=20,
                                max_length=500
                            ),
                            domain=st.session_state.domain
                        )
                        
                        status_text.text("📊 Extracting paths...")
                        progress_bar.progress(10)
                        
                        status_text.text("🤖 Generating training examples...")
                        dataset = trainer.generate(count=st.session_state.count)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Generation complete!")
                        
                        # Store results in session state for display
                        st.session_state.dataset = dataset
                        st.session_state.auto_tuner = None
                        st.session_state.generation_complete = True
                    
                except Exception as e:
                    st.error(f"❌ Generation error: {e}")
                finally:
                    # Always clean up temp file
                    os.unlink(tmp_path)
    
    # =========================================================================
    # RESULTS SECTION
    # =========================================================================
    # Display results after generation is complete
    if st.session_state.generation_complete and st.session_state.dataset is not None:
        st.divider()
        st.subheader("📋 Generation Results")
        
        dataset = st.session_state.dataset
        metadata = dataset.metadata
        
        # -----------------------------------------------------------------
        # Key Metrics Display
        # -----------------------------------------------------------------
        # Extract metrics from dataset metadata
        gen_cfg = metadata.get("generation_config", {})
        cost_info = metadata.get("cost", {})
        quality_info = metadata.get("quality", {})
        
        # Display in 4-column layout
        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
        
        with result_col1:
            # Number of successfully generated examples
            st.metric(
                "Generated",
                gen_cfg.get("count_generated", 0),
                delta=f"{gen_cfg.get('acceptance_rate', 0)*100:.0f}% acceptance"
            )
        
        with result_col2:
            # Number of rejected examples (failed quality threshold)
            st.metric(
                "Rejected",
                gen_cfg.get("count_rejected", 0)
            )
        
        with result_col3:
            # Average quality score of accepted examples
            st.metric(
                "Avg Quality",
                f"{quality_info.get('avg_quality', 0):.2f}"
            )
        
        with result_col4:
            # Total API cost in USD
            st.metric(
                "Total Cost",
                f"${cost_info.get('total_cost', 0):.4f}"
            )
        
        # -----------------------------------------------------------------
        # Auto-Tuning Quality Impact Assessment (only in auto mode)
        # -----------------------------------------------------------------
        auto_tuner = getattr(st.session_state, 'auto_tuner', None)
        if auto_tuner is not None and hasattr(auto_tuner, 'tier_results') and auto_tuner.tier_results:
            gen_cfg_mode = metadata.get("generation_config", {}).get("mode", "")
            if gen_cfg_mode == "auto":
                with st.expander("⚠️ Quality Impact Assessment (Auto-Tuning)", expanded=True):
                    # Determine quality level
                    max_tier_used = -1
                    for r in auto_tuner.tier_results:
                        if r.generated_count > 0:
                            max_tier_used = max(max_tier_used, r.iteration)
                    
                    if max_tier_used <= 0:
                        st.success(
                            "✅ **Quality Level: HIGH** — All examples generated with "
                            "default parameters. No trade-offs were needed."
                        )
                    elif max_tier_used <= 2:
                        st.warning(
                            f"🟡 **Quality Level: MODERATE** — Some parameters were relaxed. "
                            f"Highest tuning tier used: {max_tier_used} "
                            f"({auto_tuner.TUNING_TIERS[max_tier_used]['description']})"
                        )
                    else:
                        st.error(
                            f"🔴 **Quality Level: REDUCED** — Aggressive parameters were used "
                            f"to meet the target count. "
                            f"Highest tuning tier used: {max_tier_used} "
                            f"({auto_tuner.TUNING_TIERS[max_tier_used]['description']})"
                        )
                    
                    # Iteration breakdown table
                    st.markdown("**Iteration Breakdown:**")
                    iter_data = []
                    for r in auto_tuner.tier_results:
                        tier_desc = auto_tuner.TUNING_TIERS[r.iteration]["description"]
                        iter_data.append({
                            "Iter": r.iteration,
                            "Description": tier_desc,
                            "Generated": r.generated_count,
                            "Rejected": r.rejected_count,
                            "Avg Quality": f"{r.avg_quality:.3f}",
                            "Quality Threshold": r.quality_threshold,
                            "Dedup Threshold": r.dedup_threshold,
                            "Temperature": r.temperature,
                            "Sampling": r.sampling_strategy,
                        })
                    st.dataframe(iter_data, use_container_width=True, hide_index=True)
                    
                    # Collect unique warnings
                    all_warnings = []
                    seen_warnings = set()
                    for r in auto_tuner.tier_results:
                        for w in r.quality_warnings:
                            if w not in seen_warnings:
                                seen_warnings.add(w)
                                all_warnings.append(w)
                    
                    if all_warnings:
                        st.markdown("**Trade-offs applied:**")
                        for w in all_warnings:
                            st.markdown(f"- {w}")
                    
                    # Recommendations
                    st.markdown("**Recommendations:**")
                    if max_tier_used <= 0:
                        st.markdown("- Data is high quality and ready for fine-tuning.")
                    else:
                        st.markdown("- Review generated examples for accuracy before fine-tuning.")
                        if max_tier_used >= 3:
                            st.markdown("- Consider manually filtering low-quality examples.")
                            st.markdown(
                                "- Consider enriching your knowledge graph with more "
                                "nodes/edges for better results."
                            )
                        avg_q = quality_info.get('avg_quality', 1.0)
                        if avg_q < 0.6:
                            st.markdown(
                                f"- Average quality ({avg_q:.3f}) is below 0.6 — "
                                f"manual review is strongly recommended."
                            )
        
        # -----------------------------------------------------------------
        # Detailed Cost Breakdown (Collapsible)
        # -----------------------------------------------------------------
        with st.expander("💰 Cost Breakdown"):
            cost_col1, cost_col2, cost_col3 = st.columns(3)
            with cost_col1:
                st.write(f"**Input Tokens:** {cost_info.get('input_tokens', 0):,}")
                st.write(f"**Input Cost:** ${cost_info.get('input_cost', 0):.4f}")
            with cost_col2:
                st.write(f"**Output Tokens:** {cost_info.get('output_tokens', 0):,}")
                st.write(f"**Output Cost:** ${cost_info.get('output_cost', 0):.4f}")
            with cost_col3:
                st.write(f"**API Calls:** {cost_info.get('api_calls', 0)}")
                st.write(f"**Cost/Example:** ${cost_info.get('avg_cost_per_example', 0):.6f}")
        
        # -----------------------------------------------------------------
        # Example Preview (Collapsible, Expanded by Default)
        # -----------------------------------------------------------------
        if len(dataset.examples) > 0:
            with st.expander("👁️ Preview Examples", expanded=True):
                # Show first 5 examples
                preview_count = min(5, len(dataset.examples))
                for i, example in enumerate(dataset.examples[:preview_count]):
                    with st.container():
                        st.markdown(f"**Example {i+1}** (Quality: {example.get('quality_score', 0):.2f})")
                        # Display Q&A messages
                        messages = example.get("messages", [])
                        for msg in messages:
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if role == "user":
                                st.markdown(f"❓ **Q:** {content}")
                            elif role == "assistant":
                                st.markdown(f"💬 **A:** {content}")
                        st.divider()
        
        # -----------------------------------------------------------------
        # Download Section
        # -----------------------------------------------------------------
        st.subheader("📥 Download Results")
        
        download_col1, download_col2 = st.columns(2)
        
        # Prepare JSONL content (one JSON object per line, no quality_score)
        # This format is standard for SFT training pipelines
        jsonl_content = ""
        for ex in dataset.examples:
            # Remove internal quality_score field from output
            output_ex = {k: v for k, v in ex.items() if k != 'quality_score'}
            jsonl_content += json.dumps(output_ex) + '\n'
        
        # Prepare JSON content (pretty-printed array for human review)
        json_content = json.dumps(dataset.examples, indent=2, ensure_ascii=False)
        
        with download_col1:
            # JSONL format: Standard for fine-tuning pipelines
            st.download_button(
                label="📄 Download JSONL (SFT Format)",
                data=jsonl_content,
                file_name="training_data.jsonl",
                mime="application/jsonl",
                use_container_width=True
            )
        
        with download_col2:
            # JSON format: Human-readable for review/debugging
            st.download_button(
                label="📋 Download JSON (Human Readable)",
                data=json_content,
                file_name="training_data.json",
                mime="application/json",
                use_container_width=True
            )
    
    # =========================================================================
    # FOOTER
    # =========================================================================
    st.divider()
    st.caption("kg2sft v1.0.0 | Knowledge Graph to SFT Training Data Generator | Built with Streamlit")


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
