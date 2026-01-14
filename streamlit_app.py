"""
Luminous Fingerprint Analysis System - Streamlit UI
Clean, modern interface inspired by professional design
"""

import streamlit as st
import requests
import base64
from pathlib import Path
from PIL import Image
import io
import time
import subprocess
import os
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Luminous - Fingerprint Analysis",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API endpoint configuration
# Can be set via Streamlit secrets or environment variable or defaults to localhost
# For deployment: Set API_URL in Streamlit Cloud secrets
# Example: API_URL="https://luminousproject.onrender.com"

# Try Streamlit secrets first (for Streamlit Cloud), then environment variables
try:
    API_URL = st.secrets.get("API_URL", None)
except (AttributeError, FileNotFoundError):
    API_URL = None

# Fallback to environment variable
if not API_URL:
    API_URL = os.getenv("API_URL")

# Final fallback to legacy host:port configuration for local development
if not API_URL:
    API_HOST = os.getenv("API_HOST", "localhost")
    API_PORT = os.getenv("API_PORT", "8000")
    API_URL = f"http://{API_HOST}:{API_PORT}"

print(f"🔗 Using API endpoint: {API_URL}")

# Load custom CSS
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts - Poppins (exact Figma font) */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Clean white background - Responsive padding */
    .main {
        background: #FFFFFF;
        padding: 0 2rem;
    }
    
    @media (max-width: 1200px) {
        .main {
            padding: 0 1rem;
        }
    }
    
    /* Remove top spacing and ensure viewport fit */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem;
        max-width: 100% !important;
        overflow-x: hidden;
    }
    
    /* Ensure body fits viewport */
    body {
        overflow-x: hidden;
        max-width: 100vw;
    }
    
    /* Constrain images to viewport - but allow login page image to be responsive */
    .stImage img {
        max-width: 100% !important;
        height: auto !important;
        object-fit: contain !important;
        display: block !important;
        margin: 0 auto !important;
    }
    
    /* Regular page images - constrained */
    .main .stImage img:not([data-login-image]) {
        max-height: 50vh !important;
    }
    
    /* Login page image - fully responsive, follows zoom */
    [data-testid="stImage"] img {
        max-height: none !important;
    }
    
    /* Specific fix for login page right column image - fully responsive */
    div[data-testid="column"] img {
        max-height: none !important;
    }
    
    /* Login page specific - make illustration fully responsive */
    .main > div:has(img[alt*="Luminous-Start"]) img,
    .main > div:has(img[src*="Luminous-Start"]) img {
        max-height: none !important;
        max-width: 100% !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
    }
    
    /* Ensure login page columns allow full image display */
    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* Remove modal backdrop completely - only show popup on original page */
    [data-baseweb="modal"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        max-width: none !important;
        max-height: none !important;
        margin: 0 !important;
        padding: 0 !important;
        background: rgba(0, 0, 0, 0.5) !important;
        background-color: rgba(0, 0, 0, 0.5) !important;
        pointer-events: auto !important;
        z-index: 1000 !important;
    }
    
    /* Remove all backdrop layers */
    [data-baseweb="modal"]::before,
    [data-baseweb="modal"]::after {
        display: none !important;
        background: transparent !important;
    }
    
    /* Make backdrop layer transparent */
    [role="presentation"] {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Modal content container - properly sized and centered with spacing */
    [data-baseweb="modal"] > div {
        position: fixed !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        width: auto !important;
        min-width: 600px !important;
        max-width: 65vw !important;
        max-height: 65vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        box-sizing: border-box !important;
        padding: 1rem 1.5rem !important;
        margin: 0 !important;
        background: white !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3) !important;
        pointer-events: auto !important;
    }
    
    /* Minimal spacing in dialog - everything compact */
    [data-baseweb="modal"] [data-testid="stVerticalBlock"] {
        max-height: none !important;
        overflow: visible !important;
        padding: 0.3rem !important;
        margin-bottom: 0.1rem !important;
    }
    
    /* Reduce spacing in markdown divs */
    [data-baseweb="modal"] div[style*="text-align: center"] {
        margin-bottom: 0.1rem !important;
        padding: 0 !important;
    }
    
    /* Compact all text elements */
    [data-baseweb="modal"] h3 {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    [data-baseweb="modal"] p {
        margin: 0.05rem 0 !important;
        padding: 0 !important;
    }
    
    /* Larger preview images in modal */
    [data-baseweb="modal"] .stImage {
        margin: 0.3rem 0 !important;
    }
    
    [data-baseweb="modal"] .stImage img {
        max-height: 50vh !important;
        max-width: 100% !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
    }
    
    /* Overlay images in expander - even smaller */
    [data-baseweb="modal"] [data-testid="stExpander"] .stImage img {
        max-height: 25vh !important;
    }
    
    /* Very compact buttons - minimal spacing */
    [data-baseweb="modal"] .stButton {
        margin: 0.05rem 0 !important;
    }
    
    [data-baseweb="modal"] .stButton > button {
        padding: 0.35rem 0.7rem !important;
        font-size: 12px !important;
        min-height: auto !important;
    }
    
    /* Compact metrics */
    [data-baseweb="modal"] [data-testid="stMetric"] {
        margin: 0.1rem 0 !important;
        padding: 0.3rem !important;
    }
    
    /* Compact expander */
    [data-baseweb="modal"] [data-testid="stExpander"] {
        margin: 0.25rem 0 !important;
    }
    
    [data-baseweb="modal"] [data-testid="stExpander"] [data-testid="stVerticalBlock"] {
        padding: 0.5rem !important;
    }
    
    /* Smaller headings with reduced top margin */
    [data-baseweb="modal"] h3 {
        margin: 0.1rem 0 !important;
        font-size: 0.9rem !important;
        line-height: 1.1 !important;
    }
    
    /* Reduce top spacing for first element in modal */
    [data-baseweb="modal"] > div > div:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Compact messages */
    [data-baseweb="modal"] [data-baseweb="alert"],
    [data-baseweb="modal"] [data-testid="stAlert"] {
        margin: 0.15rem 0 !important;
        padding: 0.4rem !important;
        font-size: 11px !important;
    }
    
    /* Minimal column spacing */
    [data-baseweb="modal"] [data-testid="column"] {
        padding: 0.05rem !important;
        gap: 0.15rem !important;
    }
    
    /* Reduce all paragraph margins */
    [data-baseweb="modal"] p {
        margin: 0.05rem 0 !important;
        line-height: 1.2 !important;
    }
    
    /* Compact markdown separators */
    [data-baseweb="modal"] hr {
        margin: 0.3rem 0 !important;
    }
    
    /* Prevent scrolling when modal is open */
    html, body {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }
    
    /* Responsive font sizes */
    @media (max-width: 768px) {
        .page-heading {
            font-size: 24px !important;
        }
        h1, h2, h3 {
            font-size: 1.2em !important;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Heading - Purple/Blue gradient color */
    .page-heading {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 32px;
        font-weight: 600;
        line-height: 1.3em;
        margin-bottom: 1.5rem;
        margin-top: 0;
    }
    
    /* Style the bordered container (form frame) - Reduced padding for better fit */
    [data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] > div[style*="border"] {
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        margin-top: 1rem !important;
        background: #FFFFFF !important;
        transition: all 0.3s ease !important;
        max-width: 100% !important;
    }
    
    [data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] > div[style*="border"]:hover {
        border-color: #CBD5E0 !important;
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.08), 0 4px 6px rgba(0, 0, 0, 0.05) !important;
        transform: translateY(-2px);
    }
    
    /* Input Labels - Better spacing and alignment */
    .stTextInput > label {
        color: #1A202C;
        font-size: 14px;
        font-weight: 600;
        line-height: 1.5em;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    /* Input Fields - Improved visual organization */
    .stTextInput {
        max-width: 100%;
        margin-bottom: 1rem;
    }
    
    .stTextInput > div {
        margin-top: 0.5rem;
    }
    
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1.5px solid #E2E8F0;
        padding: 12px 16px;
        font-size: 14px;
        font-weight: 500;
        background: #FFFFFF;
        transition: all 0.2s ease;
        width: 100%;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #10B981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1), 0 1px 2px rgba(0, 0, 0, 0.05);
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #A0AEC0;
        font-size: 14px;
        font-weight: 400;
    }
    
    /* Primary Button (Sign In) - Bright younger green */
    .stButton {
        margin-top: 0.5rem;
    }
    
    .stButton > button[kind="primary"],
    .stButton > button:first-child {
        width: 100%;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 24px;
        font-size: 15px;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .stButton > button[kind="primary"]:hover,
    .stButton > button:first-child:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }
    
    
    /* Dashboard Styles - Responsive */
    .dashboard-container {
        background: #F7FAFC;
        min-height: 100vh;
        padding: 1rem;
    }
    
    @media (max-width: 1200px) {
        .dashboard-container {
            padding: 0.5rem;
        }
    }
    
    .dashboard-header {
        background: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .user-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1A202C;
    }
    
    .user-subtitle {
        color: #718096;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    
    /* Logout Button */
    .stButton > button[kind="secondary"] {
        background: #FFFFFF;
        color: #E53E3E;
        border: 1.5px solid #E53E3E;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: #E53E3E;
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        font-weight: 600;
        color: #4A5568;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        border-bottom: 3px solid #667eea;
    }
    
    /* Card Styles */
    .info-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1A202C;
        margin-bottom: 1rem;
    }
    
    .card-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Result Display */
    .result-box {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-top: 1.5rem;
    }
    
    .result-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1A202C;
        margin-bottom: 1rem;
    }
    
    /* File Uploader */
    .stFileUploader {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed #E2E8F0;
    }
    
    .stFileUploader:hover {
        border-color: #4C51BF;
    }
    
    /* Action Buttons - Bright green to match sign in */
    .action-btn {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }
    
    .action-btn:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
    }
    
    /* Metrics */
    .stMetric {
        background: #F7FAFC;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Success/Error Messages */
    .stAlert {
        border-radius: 8px;
        border-left-width: 4px;
        margin-top: 1rem;
    }
    
    /* Go to Summary Button */
    button[key="summary_btn"] {
        background: #379F85 !important;
        color: white !important;
        font-size: 32px !important;
        font-weight: 400 !important;
        padding: 2rem !important;
        border-radius: 8px !important;
    }
    
    /* Fingerprint Container Borders - Blue like Figma */
    .main [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] > div[style*="border"] {
        border: 1px solid #0F3CD2 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Show API configuration helper
def show_api_config():
    """Display API configuration information"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ API Configuration")
        st.write(f"**Endpoint:** `{API_URL}`")
        
        # Detect source of API_URL
        try:
            if st.secrets.get("API_URL"):
                st.caption("✓ Using Streamlit Cloud secrets")
        except (AttributeError, FileNotFoundError):
            pass
        
        if os.getenv("API_URL"):
            st.caption("✓ Using environment variable")
        elif "localhost" in API_URL or "127.0.0.1" in API_URL:
            st.caption(f"⚠ Using localhost (local development)")
            st.caption(f"Host: {os.getenv('API_HOST', 'localhost')}")
            st.caption(f"Port: {os.getenv('API_PORT', '8000')}")
        
        # Show how to change for deployment
        with st.expander("📝 Deployment Settings"):
            st.markdown("""
            **For Streamlit Cloud deployment:**
            
            1. Go to your Streamlit Cloud app settings
            2. Navigate to **Secrets** section
            3. Add environment variable:
               ```toml
               API_URL = "https://your-backend.onrender.com"
               ```
            4. Restart your app
            
            **For local network deployment:**
            
            1. Find your server IP address:
               ```bash
               # On Windows:
               ipconfig
               
               # On Linux/Mac:
               ifconfig
               ```
            
            2. Set environment variable before starting Streamlit:
               ```bash
               # Windows PowerShell:
               $env:API_HOST="192.168.1.100"
               streamlit run streamlit_app.py
               
               # Linux/Mac:
               export API_HOST="192.168.1.100"
               streamlit run streamlit_app.py
               ```
            
            3. Both API (port 8000) and Streamlit (port 8501) must run on server
            
            4. Access from other laptop using server IP:
               ```
               http://192.168.1.100:8501
               ```
            """)

# Initialize session state
def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'scan_result' not in st.session_state:
        st.session_state.scan_result = None
    if 'fingerprints' not in st.session_state:
        st.session_state.fingerprints = {}
    if 'show_summary' not in st.session_state:
        st.session_state.show_summary = False
    if 'input_mode' not in st.session_state:
        st.session_state.input_mode = None

# Display fingerprint section for each hand
def display_fingerprint_section(hand, num_fingers):
    """Display fingerprint input slots for a hand"""
    fingers = ["Thumb", "Index", "Middle", "Ring", "Little"]
    finger_codes = ["1", "2", "3", "4", "5"]
    
    for i in range(num_fingers):
        finger_name = fingers[i]
        finger_code = f"{'R' if hand == 'right' else 'L'}{finger_codes[i]}"
        
        # Create a container for each fingerprint row
        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 2, 1.5])
            
            with col1:
                # Fingerprint thumbnail - show captured image if available
                # Use consistent key format: R1, R2, L1, L2, etc.
                status_key = f"{'R' if hand == 'right' else 'L'}{i+1}"
                if status_key in st.session_state.fingerprints:
                    # Show captured fingerprint image
                    try:
                        img_base64 = st.session_state.fingerprints[status_key]["image_base64"]
                        img_data = base64.b64decode(img_base64)
                        img = Image.open(io.BytesIO(img_data))
                        st.image(img, width=80)
                    except Exception as e:
                        st.markdown(f"""
                        <div style='
                            width: 80px;
                            height: 100px;
                            border: 1px solid #DE8F0F;
                            border-radius: 4px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            background: #F5F5F5;
                            font-size: 12px;
                            color: #666;
                        '>
                            {finger_code}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Show placeholder
                    st.markdown(f"""
                    <div style='
                        width: 80px;
                        height: 100px;
                        border: 1px solid #DE8F0F;
                        border-radius: 4px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background: #F5F5F5;
                        font-size: 12px;
                        color: #666;
                    '>
                        {finger_code}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <p style='font-size: 20px; font-weight: 400; margin-bottom: 0.5rem;'>
                    <strong>Finger:</strong> {finger_code} ({finger_name})
                </p>
                """, unsafe_allow_html=True)
                # Check if already scanned
                if status_key in st.session_state.fingerprints:
                    # Already scanned - show rescan button
                    if st.button(f"Rescan {finger_name}", key=f"scan_{hand}_{i}", use_container_width=True):
                        # Set flag to open dialog
                        st.session_state.active_scan_finger = status_key
                        st.session_state.active_scan_name = f"{finger_code} ({finger_name})"
                        st.session_state.active_scan_is_rescan = True
                        st.rerun()
                else:
                    # Not scanned yet - show scan button
                    if st.button(f"Scan {finger_name}", key=f"scan_{hand}_{i}", use_container_width=True, type="primary"):
                        # Set flag to open dialog
                        st.session_state.active_scan_finger = status_key
                        st.session_state.active_scan_name = f"{finger_code} ({finger_name})"
                        st.session_state.active_scan_is_rescan = False
                        st.rerun()
            
            with col3:
                # Status indicator with validation check
                # Use consistent key format: R1, R2, L1, L2, etc.
                status_key = f"{'R' if hand == 'right' else 'L'}{i+1}"
                if status_key in st.session_state.fingerprints:
                    # Check if there are validation warnings/errors
                    fp_data = st.session_state.fingerprints[status_key]
                    analysis = fp_data.get("analysis", {})
                    has_warnings = False
                    
                    if analysis:
                        # Check for validation failures
                        warnings = analysis.get("warnings", [])
                        violations = analysis.get("violations", [])
                        quality_status = analysis.get("quality", {}).get("status", "OK")
                        success = analysis.get("success", False)
                        
                        # Determine if there are issues
                        has_warnings = (
                            not success or 
                            quality_status in ["FAIL", "WARN"] or 
                            len(warnings) > 0 or 
                            len(violations) > 0
                        )
                    
                    if has_warnings:
                        # Warning status (yellow)
                        st.markdown("""
                        <div style='
                            display: flex;
                            align-items: center;
                            gap: 0.5rem;
                            padding: 0.5rem;
                            min-height: 50px;
                        '>
                            <div style='
                                width: 40px;
                                height: 40px;
                                background: #F59E0B;
                                border-radius: 50%;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                color: white;
                                font-size: 20px;
                                flex-shrink: 0;
                            '>⚠</div>
                            <p style='
                                font-size: 16px; 
                                margin: 0;
                                line-height: 1.5;
                                padding: 2px 0;
                                white-space: nowrap;
                                overflow: visible;
                                color: #D97706;
                            '><strong>Warning</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show warning details in expander
                        with st.expander("⚠️ View Issues", expanded=False):
                            if not success:
                                st.error("❌ Analysis failed")
                            if quality_status in ["FAIL", "WARN"]:
                                st.warning(f"🔍 Quality: {quality_status}")
                            if warnings:
                                st.warning("**Warnings:**")
                                for w in warnings:
                                    st.write(f"- {w.get('message', 'Unknown warning')}")
                            if violations:
                                st.error("**Validation Errors:**")
                                for v in violations:
                                    st.write(f"- {v.get('message', 'Unknown violation')}")
                            st.info("💡 Consider rescanning this finger for better quality")
                    else:
                        # Success status (green)
                        st.markdown("""
                        <div style='
                            display: flex;
                            align-items: center;
                            gap: 0.5rem;
                            padding: 0.5rem;
                            min-height: 50px;
                        '>
                            <div style='
                                width: 40px;
                                height: 40px;
                                background: #10B981;
                                border-radius: 50%;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                color: white;
                                font-size: 20px;
                                flex-shrink: 0;
                            '>✓</div>
                            <p style='
                                font-size: 16px; 
                                margin: 0;
                                line-height: 1.5;
                                padding: 2px 0;
                                white-space: nowrap;
                                overflow: visible;
                            '><strong>Captured</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Pending status (gray)
                    st.markdown("""
                    <div style='
                        display: flex;
                        align-items: center;
                        gap: 0.5rem;
                        padding: 0.5rem;
                        min-height: 50px;
                    '>
                        <div style='
                            width: 40px;
                            height: 40px;
                            background: #E2E8F0;
                            border-radius: 50%;
                            flex-shrink: 0;
                        '></div>
                        <p style='
                            font-size: 16px; 
                            margin: 0; 
                            color: #718096;
                            line-height: 1.5;
                            padding: 2px 0;
                            white-space: nowrap;
                            overflow: visible;
                        '><strong>Pending</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

# Login Page
def login_page():
    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.markdown("<div style='max-width: 404px; margin: 0 auto;'>", unsafe_allow_html=True)
        logo_path = Path("WebImages/luminous-logo-withname.png")
        if logo_path.exists():
            st.image(Image.open(logo_path), width=200)
        st.markdown("<div style='margin-top: 2.5rem; margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("<h1 class='page-heading'>Welcome Back</h1>", unsafe_allow_html=True)
        
        with st.container(border=True):
            username = st.text_input("Name", placeholder="Enter your name", key="username_input")
            
            # [修改] 移除了这里的 Radio Button 选择代码
            
            if st.button("Sign In", use_container_width=True, type="primary"):
                if username and username.strip():
                    st.session_state.logged_in = True
                    st.session_state.username = username.strip()
                    # 注意：这里不设置 input_mode，跳转到选择页去设置
                    st.rerun()
                else:
                    st.error("⚠️ Please enter your name")
        
        st.markdown("<div style='margin-top: 2rem; text-align: center;'><p style='color: #A0AEC0; font-size: 14px;'>Secure • Fast • Accurate Fingerprint Analysis</p></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with right_col:
        st.markdown("<div style='display: flex; align-items: flex-start; justify-content: center;'>", unsafe_allow_html=True)
        illustration_path = Path("WebImages/Luminous-Start.png")
        if illustration_path.exists():
            st.image(Image.open(illustration_path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

def mode_selection_page():
    """
    New intermediate page to select input mode
    """
    # 顶部简单的 Header
    st.markdown(f"<div style='text-align: center; padding-top: 2rem; margin-bottom: 3rem;'><h2 style='font-family:Inter;font-size:36px;'>Hi, {st.session_state.username}!</h2><p style='color:#666; font-size:18px;'>Choose your capture method to continue</p></div>", unsafe_allow_html=True)

    # 创建两列布局
    col1, col2, col3 = st.columns([1, 4, 1]) # 让中间宽一点，或者直接用下面这种居中布局
    
    # 更好的布局：两个大卡片居中
    with col2:
        c1, c2 = st.columns(2, gap="large")
        
        # === Touchbased Card ===
        with c1:
            with st.container(border=True):
                st.markdown("<div style='text-align:center; height:150px; display:flex; flex-direction:column; justify-content:center; align-items:center;'>", unsafe_allow_html=True)
                st.markdown("<div style='font-size: 60px; margin-bottom: 10px;'>👆</div>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin:0;'>Touchbased</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color:#888; font-size:14px;'>Hardware Scanner</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                if st.button("Select Scanner", use_container_width=True, type="primary", key="btn_touchbased"):
                    st.session_state.input_mode = "Touchbased"
                    st.rerun()

        # === Touchless Card ===
        with c2:
            with st.container(border=True):
                st.markdown("<div style='text-align:center; height:150px; display:flex; flex-direction:column; justify-content:center; align-items:center;'>", unsafe_allow_html=True)
                st.markdown("<div style='font-size: 60px; margin-bottom: 10px;'>📸</div>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin:0;'>Touchless</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color:#888; font-size:14px;'>Camera Capture</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                if st.button("Select Camera", use_container_width=True, type="primary", key="btn_touchless"):
                    st.session_state.input_mode = "Touchless"
                    st.rerun()

    # 底部返回按钮
    st.markdown("<div style='margin-top: 3rem; text-align: center;'>", unsafe_allow_html=True)
    if st.button("← Logout", type="secondary"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.input_mode = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
        
# Helper function to save fingerprint to its own folder
def save_fingerprint_to_folder(finger_key, finger_name, image_base64):
    """
    Save fingerprint image and metadata to its own folder.
    Creates: results/captures/{username}/{finger_key}/
    Returns: folder path
    """
    try:
        # Create folder structure
        username = st.session_state.get("username", "unknown").replace(" ", "_")
        base_dir = Path("results/captures") / username / finger_key
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Save fingerprint image
        img_bytes = base64.b64decode(image_base64)
        img_path = base_dir / "fingerprint.bmp"
        with open(img_path, 'wb') as f:
            f.write(img_bytes)
        
        # Save metadata
        metadata = {
            "finger_id": finger_key,
            "finger_name": finger_name,
            "username": st.session_state.get("username", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "image_path": str(img_path)
        }
        metadata_path = base_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return base_dir
    except Exception as e:
        print(f"Error saving fingerprint to folder: {e}")
        return None

# Helper function to run fingerprint analysis
def analyze_fingerprint(finger_key, image_base64, folder_path):
    """
    Run classification and Poincaré index detection on fingerprint.
    Saves results to folder and returns analysis data.
    """
    API_BASE = API_URL  # Use configured API endpoint
    
    try:
        # Convert base64 to bytes
        img_bytes = base64.b64decode(image_base64)
        
        # Send to API for analysis
        files = {"file": ("fingerprint.bmp", io.BytesIO(img_bytes), "image/bmp")}
        
        print(f"🔬 Analyzing {finger_key}...")
        print(f"   Image size: {len(img_bytes)} bytes")
        print(f"   Calling API: {API_BASE}/detect")
        
        response = requests.post(f"{API_BASE}/detect", files=files, timeout=60)
        
        print(f"   API Response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Debug: Print key results
            print(f"   Success: {result.get('success')}")
            print(f"   Classification: {result.get('classification', {}).get('predicted_class', 'N/A')}")
            print(f"   Confidence: {result.get('classification', {}).get('confidence', 0):.2f}")
            print(f"   Cores: {result.get('num_cores', 0)}")
            print(f"   Deltas: {result.get('num_deltas', 0)}")
            print(f"   Ridge Counts: {result.get('ridge_counts', [])}")
            print(f"   Has Overlay: {bool(result.get('overlay_base64'))}")
            
            # Save analysis results to folder
            if folder_path:
                results_path = Path(folder_path) / "analysis_results.json"
                with open(results_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"   Saved results to: {results_path}")
                
                # Save overlay image if available
                if result.get("overlay_base64"):
                    overlay_data = result["overlay_base64"]
                    if overlay_data.startswith("data:image"):
                        overlay_data = overlay_data.split(",")[1]
                    overlay_bytes = base64.b64decode(overlay_data)
                    overlay_path = Path(folder_path) / "detection_overlay.png"
                    with open(overlay_path, 'wb') as f:
                        f.write(overlay_bytes)
                    print(f"   Saved overlay to: {overlay_path}")
            
            return result
        else:
            error_msg = f"Analysis failed for {finger_key}: HTTP {response.status_code}"
            print(f"   ❌ {error_msg}")
            try:
                error_detail = response.json()
                print(f"   Error detail: {error_detail}")
            except:
                print(f"   Response text: {response.text[:200]}")
            return None
            
    except requests.exceptions.ConnectionError as e:
        error_msg = f"❌ Cannot connect to API at {API_BASE}. Is the backend running?"
        print(error_msg)
        return {"error": error_msg, "success": False}
    except Exception as e:
        error_msg = f"Error analyzing fingerprint {finger_key}: {str(e)}"
        print(f"   ❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {"error": error_msg, "success": False}

# Helper function to analyze all fingerprints
def analyze_all_fingerprints(progress_callback=None):
    """
    Analyze all captured fingerprints with classification and Poincaré index.
    Updates session state with results.
    Returns: (success_count, total_count)
    """
    success_count = 0
    total_count = len(st.session_state.fingerprints)
    
    for idx, (finger_key, fp_data) in enumerate(st.session_state.fingerprints.items()):
        if progress_callback:
            progress_callback(idx + 1, total_count, fp_data.get("finger_full_name", finger_key))
        
        # Skip if already analyzed
        if fp_data.get("analysis") and fp_data["analysis"].get("success"):
            success_count += 1
            continue
        
        # Get folder path (should already exist)
        username = st.session_state.get("username", "unknown").replace(" ", "_")
        folder_path = Path("results/captures") / username / finger_key
        
        # Run analysis
        image_base64 = fp_data.get("image_base64")
        if image_base64:
            result = analyze_fingerprint(finger_key, image_base64, folder_path)
            if result:
                # Store analysis in session state
                st.session_state.fingerprints[finger_key]["analysis"] = result
                st.session_state.fingerprints[finger_key]["folder_path"] = str(folder_path)
                
                if result.get("success"):
                    success_count += 1
    
    return success_count, total_count

# Helper function to validate fingerprint data
def validate_fingerprint_data(image_base64):
    """
    Validate that fingerprint data is valid and not corrupted.
    Returns True if valid, False otherwise.
    """
    if not image_base64:
        return False
    
    try:
        # Try to decode base64
        img_bytes = base64.b64decode(image_base64)
        
        # Check minimum size (fingerprint images should be reasonably sized)
        if len(img_bytes) < 1000:  # Less than 1KB is suspicious
            return False
        
        # Try to open as image
        img = Image.open(io.BytesIO(img_bytes))
        
        # Check dimensions (ZKTeco scanner typically 300x400 or similar)
        width, height = img.size
        if width < 100 or height < 100:  # Too small
            return False
        
        return True
    except Exception as e:
        print(f"Fingerprint validation error: {e}")
        return False

# Helper function to get fingerprint list for validation
def get_all_finger_keys():
    """Returns list of all 10 finger keys in order (new format: R1-R5, L1-L5)"""
    return [
        "R1", "R2", "R3", "R4", "R5",  # Right hand (Left Brain)
        "L1", "L2", "L3", "L4", "L5"   # Left hand (Right Brain)
    ]

# Helper function to check if all fingerprints are captured
def all_fingerprints_captured():
    """Check if all 10 fingerprints are captured and valid"""
    required_keys = get_all_finger_keys()
    
    for key in required_keys:
        if key not in st.session_state.fingerprints:
            return False
        
        # Validate the fingerprint data
        fp_data = st.session_state.fingerprints[key]
        if not fp_data.get("image_base64"):
            return False
    
    return True

# Helper function to capture fingerprint using CaptureOnce (same as test_java_capture.py)
def capture_fingerprint_once():
    """
    Directly call CaptureOnce.java to capture fingerprint.
    This will turn on the green light and wait for finger placement.
    Returns (base64_data, error_message) tuple.
    """
    java_capture_dir = Path("java_ capture")
    zkfp_jar = Path("ZKFinger Standard SDK 5.3.0.33/Java/lib/ZKFingerReader.jar")
    
    try:
        # Build classpath (absolute paths for Windows) - same as test
        zkfp_jar_abs = zkfp_jar.absolute()
        cp = f".;{zkfp_jar_abs}"
        
        # Set environment to include native library path
        env = os.environ.copy()
        java_capture_abs = java_capture_dir.absolute()
        env["PATH"] = f"{java_capture_abs};{env.get('PATH', '')}"
        
        # Run CaptureOnce synchronously (will turn on green light and wait)
        result = subprocess.run(
            ["java", "-cp", cp, "CaptureOnce"],
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
            cwd=str(java_capture_abs),
            env=env
        )
        
        # Check result
        if result.returncode != 0:
            return None, f"Capture failed (exit code {result.returncode})"
        
        # Parse output - same format as test
        output = result.stdout.strip()
        if not output.startswith("OK:"):
            return None, f"Unexpected output: {output}"
        
        # Extract base64 data (remove "OK:" prefix)
        b64_data = output[3:].strip()
        
        # Validate it's actually base64
        try:
            test_decode = base64.b64decode(b64_data)
            if len(test_decode) < 100:  # Sanity check
                return None, "Image data too small"
            return b64_data, None
        except Exception as e:
            return None, f"Invalid base64 data: {str(e)}"
            
    except subprocess.TimeoutExpired:
        return None, "Timeout: No fingerprint detected after 60 seconds"
    except Exception as e:
        return None, f"Error: {str(e)}"

# Scanning Dialog Function
@st.dialog("Scan Fingerprint", width="medium")
def scan_fingerprint_dialog(finger_key, finger_name, is_rescan=False):
    """
    Modal dialog for scanning a fingerprint.
    Supports both 'Touchbased' (Java Scanner) and 'Touchless' (Camera).
    """
    API_BASE = API_URL  # Use configured API endpoint
    
    # [新增] 获取当前模式
    current_mode = st.session_state.get("input_mode", "Touchbased")
    
    # Compact header
    st.markdown(f"""
    <div style='text-align: center; margin: 0; padding: 0;'>
        <h3 style='color: #10B981; margin: 0; padding: 0; font-size: 0.85rem; line-height: 1.1; font-weight: 600;'>
            {current_mode} - {'Rescanning' if is_rescan else 'Scanning'}: {finger_name}
        </h3>
        <p style='color: #666; font-size: 10px; margin: 0.1rem 0 0 0; padding: 0; line-height: 1.1;'>
            Click "Start Scan" or use Camera • Record ID: {finger_key}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize scanning state - UNIQUE PER FINGER to prevent mixing
    captured_key = f"captured_image_{finger_key}"
    saved_key = f"scan_saved_{finger_key}"
    analysis_key = f"analysis_{finger_key}"
    
    if captured_key not in st.session_state:
        st.session_state[captured_key] = None
    if saved_key not in st.session_state:
        st.session_state[saved_key] = False
    
    # =========================================================
    # 状态 A: 尚未采集图像 (显示 扫描按钮 或 摄像头)
    # =========================================================
    if not st.session_state[captured_key]:
        
        # --- 分支 1: Touchbased (原有 Java 逻辑) ---
        if current_mode == "Touchbased":
            col1, col2 = st.columns([1, 1], gap="small")
            with col1:
                if st.button("🟢 Start Scan", use_container_width=True, type="primary", key=f"start_{finger_key}"):
                    # Show scanning status
                    with st.status("🔄 Scanning...", expanded=True) as status:
                        st.write("🟢 Place finger and hold still")
                        
                        # Call CaptureOnce - this blocks until capture completes
                        b64_data, error = capture_fingerprint_once()
                        
                        if b64_data:
                            st.session_state[captured_key] = b64_data
                            status.update(label="✓ Captured!", state="complete", expanded=False)
                            # Rerun to show the preview and action buttons
                            time.sleep(0.3)
                            st.rerun()
                        else:
                            status.update(label="❌ Capture failed", state="error", expanded=True)
                            st.error(f"Error: {error}. Check scanner connection and try again.")
            
            with col2:
                if st.button("❌ Cancel", use_container_width=True, key=f"cancel_{finger_key}"):
                    if 'active_scan_finger' in st.session_state:
                        del st.session_state.active_scan_finger
                    st.rerun()

        # --- 分支 2: Touchless (摄像头逻辑) ---
        else:
            st.info("📸 Please center your finger clearly in the camera frame.")
            
            # 摄像头组件
            # 当拍照后，camera_file 会变成一个 UploadedFile 对象
            camera_file = st.camera_input(
                label="Fingerprint Camera", 
                key=f"cam_{finger_key}", 
                label_visibility="collapsed"
            )
            
            if camera_file:
                # 1. 读取二进制数据
                bytes_data = camera_file.getvalue()
                # 2. 转换为 Base64 字符串 (保持与 Scanner 数据格式一致)
                b64_data = base64.b64encode(bytes_data).decode('utf-8')
                # 3. 存入 Session
                st.session_state[captured_key] = b64_data
                # 4. 刷新页面，进入预览流程
                st.rerun()
            
            # 摄像头模式下的 Cancel 按钮
            if st.button("❌ Cancel", use_container_width=True, key=f"cancel_cam_{finger_key}"):
                if 'active_scan_finger' in st.session_state:
                    del st.session_state.active_scan_finger
                st.rerun()

    # =========================================================
    # 状态 B: 已采集图像 (显示预览 + 保存/重拍) - 逻辑通用
    # =========================================================
    else:
        # After capture: Show preview with Save, Recapture, Analyze buttons
        # Very compact button layout - minimal gap
        col1, col2, col3 = st.columns([1, 1, 1], gap="small")
        
        with col1:
            if st.button("💾 Save", use_container_width=True, type="primary", 
                        disabled=st.session_state[saved_key], key=f"save_{finger_key}"):
                # Validate fingerprint data before saving
                if validate_fingerprint_data(st.session_state[captured_key]):
                    with st.spinner("💾 Saving fingerprint to folder..."):
                        # Save to folder structure
                        folder_path = save_fingerprint_to_folder(
                            finger_key, 
                            finger_name, 
                            st.session_state[captured_key]
                        )
                        
                        if folder_path:
                            # Save to session state with unique key
                            st.session_state.fingerprints[finger_key] = {
                                "finger_id": finger_key,
                                "finger_code": finger_name.split()[0],
                                "finger_full_name": finger_name,
                                "image_base64": st.session_state[captured_key],
                                "timestamp": time.time(),
                                "scan_number": 1 if finger_key not in st.session_state.fingerprints else st.session_state.fingerprints[finger_key].get("scan_number", 0) + 1,
                                "folder_path": str(folder_path),
                                "source": current_mode  # [新增] 记录数据来源
                            }
                            st.session_state[saved_key] = True
                            
                            # AUTO-ANALYZE after saving
                            # 注意：Touchless 模式暂时也调用同一个 analyze 函数
                            # 虽然 API 可能不适配，但这是"页面一致"的要求
                            with st.spinner("🔬 Auto-analyzing..."):
                                analysis_result = analyze_fingerprint(
                                    finger_key, 
                                    st.session_state[captured_key], 
                                    folder_path
                                )
                                if analysis_result:
                                    st.session_state.fingerprints[finger_key]["analysis"] = analysis_result
                            
                            # Show compact success message
                            captured_count = len(st.session_state.fingerprints)
                            st.success(f"✅ Saved & Analyzed! ({captured_count}/10)")
                            time.sleep(1)  # Brief pause to show success message
                            
                            # Cleanup and close dialog
                            st.session_state[captured_key] = None
                            st.session_state[saved_key] = False
                            if f"analysis_{finger_key}" in st.session_state:
                                del st.session_state[f"analysis_{finger_key}"]
                            if 'active_scan_finger' in st.session_state:
                                del st.session_state.active_scan_finger
                            st.rerun()
                        else:
                            st.error("❌ Save failed. Try again.")
                else:
                    st.error("❌ Invalid data. Rescan.")

        with col2:
            if st.button("🔄 Recapture", use_container_width=True, key=f"recapture_{finger_key}"):
                st.session_state[captured_key] = None
                st.session_state[saved_key] = False
                # Clear analysis result
                if f"analysis_{finger_key}" in st.session_state:
                    del st.session_state[f"analysis_{finger_key}"]
                st.rerun()
        
        with col3:
            if st.button("🔍 Analyze", use_container_width=True, key=f"analyze_{finger_key}"):
                # Store analysis result in session state for this finger
                st.session_state[f"analysis_{finger_key}"] = "running"
                st.rerun()

    # Show preview - centered initially, side-by-side after analysis
    if st.session_state[captured_key]:
        # Check if analysis should run
        if st.session_state.get(analysis_key) == "running":
            with st.spinner("Analyzing..."):
                try:
                    img_bytes = base64.b64decode(st.session_state[captured_key])
                    files = {"file": ("fingerprint.bmp", io.BytesIO(img_bytes), "image/bmp")}
                    response = requests.post(f"{API_BASE}/detect", files=files, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state[analysis_key] = result
                        # Update fingerprint with analysis if already saved
                        if st.session_state[saved_key] and finger_key in st.session_state.fingerprints:
                            st.session_state.fingerprints[finger_key]["analysis"] = result
                    else:
                        st.session_state[analysis_key] = {"error": "Analysis failed"}
                except Exception as e:
                    st.session_state[analysis_key] = {"error": str(e)}
            st.rerun()
    
        # Check if analysis has been done
        has_analysis = analysis_key in st.session_state and st.session_state[analysis_key] not in ["running", None]
        
        if has_analysis:
            # Side-by-side: Image on left, analysis on right
            img_col, analysis_col = st.columns([1, 1], gap="small")
            
            with img_col:
                try:
                    img_data = base64.b64decode(st.session_state[captured_key])
                    img = Image.open(io.BytesIO(img_data))
                    st.image(img, caption="📸 Captured", use_container_width=True, clamp=True)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            with analysis_col:
                # Show analysis results
                result = st.session_state[analysis_key]
                if result.get("success"):
                    classification = result.get("classification", {})
                    st.markdown("**📊 Analysis**")
                    st.metric("Pattern", classification.get("predicted_class", "?").upper())
                    st.metric("Confidence", f"{classification.get('confidence', 0) * 100:.0f}%")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Cores", result.get("num_cores", 0))
                    with col_b:
                        st.metric("Deltas", result.get("num_deltas", 0))
                elif result.get("error"):
                    st.error(f"❌ {result.get('error')}")
        else:
            # No analysis yet: Show centered image only
            try:
                img_data = base64.b64decode(st.session_state[captured_key])
                img = Image.open(io.BytesIO(img_data))
                st.image(img, caption="📸 Captured Preview", use_container_width=True, clamp=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        # Initial state (Only when Touchbased, as camera mode shows camera input instead)
        if current_mode == "Touchbased":
            st.markdown("""
            <div style='
                width: 100%;
                height: 300px;
                max-height: 40vh;
                background: #f5f5f5;
                border: 2px dashed #ccc;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #999;
                font-size: 14px;
                margin-top: 0.5rem;
            '>
                <div style='text-align: center;'>
                    <div style='font-size: 48px; margin-bottom: 0.5rem;'>👆</div>
                    <div style='font-size: 14px;'>Preview will appear after capture</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
# Dashboard Page - Fingerprint Input
def dashboard_page():
mode = st.session_state.get("input_mode", "Touchbased")

    h1, h2, h3 = st.columns([2, 3, 2])
    with h1: 
        st.markdown(f"<div style='padding-top:2rem;'><h2 style='font-family:Inter;font-size:32px;color:black;'>{st.session_state.username.upper()}</h2></div>", unsafe_allow_html=True)
        
        # [新增] 切换模式按钮
        st.caption(f"Mode: **{mode}**")
        if st.button("🔄 Switch Mode", key="switch_mode_dboard", type="secondary"):
            st.session_state.input_mode = None
            st.rerun()

    with h2: 
        # [新增] 标题根据模式变化
        title = "Fingerprint Input" if mode == "Touchbased" else "Touchless Capture"
        st.markdown(f"<div style='padding-top:2rem;text-align:center;'><h1 style='font-family:Inter;font-size:48px;color:black;'>{title}</h1></div>", unsafe_allow_html=True)
   
    with h3:
        if Path("WebImages/luminous-logo-withname.png").exists():
            st.markdown("<div style='padding-top:1.5rem;'></div>", unsafe_allow_html=True)
            st.image(Image.open("WebImages/luminous-logo-withname.png"), width=150)
    
    # First purple line - after header
    st.markdown("""
    <hr style='
        border: none;
        border-top: 3px solid #664ED0;
        margin: 2rem 0 1.5rem 0;
    '>
    """, unsafe_allow_html=True)
    
    # Section titles row
    title_left, title_right = st.columns(2)
    
    with title_left:
        st.markdown("""
        <h3 style='
            font-family: Inter, sans-serif;
            font-size: 32px;
            font-weight: 400;
            color: #000000;
            text-align: center;
            margin: 0.5rem 0 1rem 0;
        '>Left Brain / Right Hand</h3>
        """, unsafe_allow_html=True)
    
    with title_right:
        st.markdown("""
        <h3 style='
            font-family: Inter, sans-serif;
            font-size: 32px;
            font-weight: 400;
            color: #000000;
            text-align: center;
            margin: 0.5rem 0 1rem 0;
        '>Right Brain / Left Hand</h3>
        """, unsafe_allow_html=True)
    
    # Purple horizontal divider line below titles (ONLY this line at top)
    st.markdown("""
    <hr style='
        border: none;
        border-top: 3px solid #664ED0;
        margin: 0 0 2rem 0;
    '>
    """, unsafe_allow_html=True)
    
    # Main content area with vertical separator
    left_section, separator_col, right_section = st.columns([49, 2, 49])
    
    with left_section:
        # Fingerprint slots for right hand (5 fingers)
        display_fingerprint_section("right", 5)
    
    with separator_col:
        # Vertical purple divider line in the middle
        st.markdown("""
        <div style='
            width: 3px;
            height: 100%;
            min-height: 700px;
            background: #664ED0;
            margin: 0 auto;
            position: relative;
            top: 0;
        '></div>
        """, unsafe_allow_html=True)
    
    with right_section:
        # Fingerprint slots for left hand (5 fingers)
        display_fingerprint_section("left", 5)
    
    # Open dialog if there's an active scan (called ONCE outside the loop)
    if hasattr(st.session_state, 'active_scan_finger') and st.session_state.active_scan_finger:
        scan_fingerprint_dialog(
            st.session_state.active_scan_finger,
            st.session_state.active_scan_name,
            st.session_state.active_scan_is_rescan
        )
    
    # Purple line before "Go to summary" button
    st.markdown("""
    <hr style='
        border: none;
        border-top: 3px solid #664ED0;
        margin: 3rem 0 2rem 0;
    '>
    """, unsafe_allow_html=True)
    
    # Check if all 10 fingerprints are captured
    total_fingers = 10
    captured_count = len(st.session_state.fingerprints)
    all_captured = captured_count == total_fingers
    
    # Progress indicator
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 1.5rem;'>
        <p style='font-size: 18px; font-weight: 600; color: {'#10B981' if all_captured else '#718096'};'>
            Fingerprints Captured: {captured_count} / {total_fingers}
        </p>
        <div style='
            width: 60%;
            height: 10px;
            background: #E2E8F0;
            border-radius: 5px;
            margin: 0.5rem auto;
            overflow: hidden;
        '>
            <div style='
                width: {(captured_count/total_fingers)*100}%;
                height: 100%;
                background: linear-gradient(135deg, #10B981 0%, #059669 100%);
                transition: width 0.3s ease;
            '></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # "Go to summary" button - INSIDE the line area
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not all_captured:
            # Show disabled button with message
            st.button("Go to summary", use_container_width=True, key="summary_btn", disabled=True)
            st.warning(f"⚠️ Please capture all {total_fingers} fingerprints before viewing summary. ({captured_count}/{total_fingers} completed)")
        else:
            if st.button("Go to summary", use_container_width=True, key="summary_btn"):
                # Run analysis on all fingerprints before showing summary
                with st.spinner("🔬 Analyzing all fingerprints..."):
                    # Create progress container
                    progress_container = st.empty()
                    
                    def update_progress(current, total, finger_name):
                        progress_container.info(f"🔬 Analyzing {finger_name}... ({current}/{total})")
                    
                    success_count, total_count = analyze_all_fingerprints(progress_callback=update_progress)
                    progress_container.empty()
                    
                    if success_count == total_count:
                        st.success(f"✅ All {total_count} fingerprints analyzed successfully!")
                        time.sleep(1)
                    else:
                        st.warning(f"⚠️ {success_count}/{total_count} fingerprints analyzed successfully. Continuing to summary...")
                        time.sleep(1.5)
                    
                    st.session_state.show_summary = True
                    st.rerun()
    
    # Purple line after "Go to summary" button to close the area
    st.markdown("""
    <hr style='
        border: none;
        border-top: 3px solid #664ED0;
        margin: 2rem 0 2rem 0;
    '>
    """, unsafe_allow_html=True)
    
    # Diagnostic information (expandable)
    with st.expander("📊 Capture Status & Diagnostics", expanded=False):
        st.write("**Stored Fingerprint Keys:**")
        if st.session_state.fingerprints:
            for key in st.session_state.fingerprints.keys():
                st.write(f"- {key}")
        else:
            st.write("No fingerprints captured yet.")
        st.markdown("### Fingerprint Capture Status")
        
        # Create a table showing all fingers
        finger_names = {
            "right_0": "R1 (Thumb)", "right_1": "R2 (Index)", "right_2": "R3 (Middle)", 
            "right_3": "R4 (Ring)", "right_4": "R5 (Little)",
            "left_0": "L1 (Thumb)", "left_1": "L2 (Index)", "left_2": "L3 (Middle)", 
            "left_3": "L4 (Ring)", "left_4": "L5 (Little)"
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Right Hand (Left Brain)**")
            for key in ["right_0", "right_1", "right_2", "right_3", "right_4"]:
                status = "✅ Captured" if key in st.session_state.fingerprints else "⏳ Pending"
                color = "green" if key in st.session_state.fingerprints else "orange"
                st.markdown(f"- {finger_names[key]}: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Left Hand (Right Brain)**")
            for key in ["left_0", "left_1", "left_2", "left_3", "left_4"]:
                status = "✅ Captured" if key in st.session_state.fingerprints else "⏳ Pending"
                color = "green" if key in st.session_state.fingerprints else "orange"
                st.markdown(f"- {finger_names[key]}: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(f"**Total Captured:** {len(st.session_state.fingerprints)} / 10")
        st.markdown(f"**All Complete:** {'✅ Yes' if all_fingerprints_captured() else '❌ No'}")
        
        # Show detailed info
        if st.session_state.fingerprints:
            st.markdown("### Captured Records Details")
            for key, data in st.session_state.fingerprints.items():
                with st.expander(f"{data.get('finger_full_name', key)}"):
                    st.json({
                        "Finger ID": data.get("finger_id", "N/A"),
                        "Finger Code": data.get("finger_code", "N/A"),
                        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data.get("timestamp", 0))),
                        "Scan Number": data.get("scan_number", 1),
                        "Has Image": "Yes" if data.get("image_base64") else "No",
                        "Has Analysis": "Yes" if data.get("analysis") else "No"
                    })

# Upload Scanner
def upload_scanner():
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("### Upload Fingerprint Image")
    st.markdown("Upload a fingerprint image (BMP, PNG, JPG, JPEG) for analysis")
    st.markdown("</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a fingerprint image",
        type=['bmp', 'png', 'jpg', 'jpeg'],
        help="Supported formats: BMP, PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='info-card'>", unsafe_allow_html=True)
            st.markdown("#### 📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, clamp=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='info-card'>", unsafe_allow_html=True)
            st.markdown("#### ⚙️ Analysis")
            
            if st.button("🔍 Analyze Fingerprint", use_container_width=True):
                with st.spinner("🔄 Analyzing fingerprint..."):
                    # Send to API
                    files = {'file': uploaded_file.getvalue()}
                    try:
                        response = requests.post(f"{API_URL}/detect", files=files)
                        result = response.json()
                        st.session_state.scan_result = result
                        st.success("✅ Analysis complete!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display results if available
        if st.session_state.scan_result:
            display_results(st.session_state.scan_result)

# Live Scanner
def live_scanner():
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("### 📷 Live Fingerprint Scanner")
    st.markdown("Connect to your fingerprint scanner for real-time capture")
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🟢 Start Scanner", use_container_width=True):
            try:
                response = requests.post(f"{API_URL}/live_scan/start")
                if response.json().get("success"):
                    st.success("✅ Scanner started!")
                else:
                    st.error(f"❌ {response.json().get('error')}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col2:
        if st.button("📸 Capture", use_container_width=True):
            try:
                with st.spinner("📸 Capturing..."):
                    response = requests.post(f"{API_URL}/live_scan/capture")
                    result = response.json()
                    st.session_state.scan_result = result
                    if result.get("success"):
                        st.success("✅ Capture successful!")
                    else:
                        st.error(f"❌ {result.get('error')}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col3:
        if st.button("🔴 Stop Scanner", use_container_width=True):
            try:
                response = requests.post(f"{API_URL}/live_scan/stop")
                st.info("ℹ️ Scanner stopped")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Display results if available
    if st.session_state.scan_result:
        display_results(st.session_state.scan_result)

# Display Results
def display_results(result):
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("## 📊 Analysis Results")
    
    if not result.get("success"):
        # Show error
        st.markdown("<div class='result-box' style='border-left: 4px solid #E53E3E;'>", unsafe_allow_html=True)
        st.error(f"❌ {result.get('error', 'Analysis failed')}")
        
        if 'structure' in result:
            with st.expander("📋 View Details"):
                st.json(result)
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # Success - show results
    col1, col2 = st.columns(2)
    
    with col1:
        # Classification Result
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Classification")
        
        final_class = result.get('final', {}).get('class', 'N/A').upper()
        confidence = result.get('final', {}).get('confidence', 0) * 100
        
        st.markdown(f"<div class='card-value'>{final_class}</div>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        
        # Pattern info
        pattern_names = {
            "wpe": "Whorl Peacock Eye",
            "ws": "Whorl Spiral",
            "wd": "Whorl Double Loop",
            "we": "Whorl Elongated",
            "lu": "Loop Ulnar",
            "au": "Loop Arch/Radial",
            "at": "Tented Arch",
            "as": "Simple Arch"
        }
        pattern_name = pattern_names.get(final_class.lower(), final_class)
        st.info(f"📌 {pattern_name}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Poincaré Detection
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown("### 🎪 Core & Delta Detection")
        
        num_cores = result.get('num_cores', 0)
        num_deltas = result.get('num_deltas', 0)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Cores", num_cores, help="Number of core points detected")
        with col_b:
            st.metric("Deltas", num_deltas, help="Number of delta points detected")
        
        # Ridge count
        if 'ridge_counts' in result and result['ridge_counts']:
            max_ridge = max(result['ridge_counts'])
            st.metric("Max Ridge Count", f"{max_ridge}", help="Maximum ridge count between core-delta pairs")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Quality Metrics
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown("### 📈 Quality Metrics")
        
        quality = result.get('quality', {})
        metrics = quality.get('metrics', {})
        
        mean_quality = metrics.get('mean_quality', 0)
        mean_coherence = metrics.get('mean_coherence', 0)
        
        st.metric("Ridge Quality", f"{mean_quality:.3f}", help="Mean ridge quality score")
        st.metric("Ridge Coherence", f"{mean_coherence:.3f}", help="Mean ridge coherence score")
        
        status = quality.get('status', 'N/A')
        if status == 'OK':
            st.success(f"✅ Quality: {status}")
        elif status == 'WARN':
            st.warning(f"⚠️ Quality: {status}")
        else:
            st.error(f"❌ Quality: {status}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Visualization
        if 'overlay_base64' in result and result['overlay_base64']:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown("### 🖼️ Detection Overlay")
            
            # Decode and display overlay
            try:
                if result['overlay_base64'].startswith('data:image'):
                    img_data = result['overlay_base64'].split(',')[1]
                else:
                    img_data = result['overlay_base64']
                
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes))
                st.image(img, use_container_width=True, clamp=True)
            except Exception as e:
                st.error(f"Failed to display overlay: {e}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Detailed Information (Expandable)
    with st.expander("📋 View Detailed Analysis"):
        st.json(result)

# Summary Report Page
def calculate_ridge_count_rankings():
    """
    Calculate rankings for all 10 fingerprints based on ridge count.
    Rank 1 = Highest ridge count, Rank 10 = Lowest ridge count
    Returns: dict mapping finger_code to rank (1-10)
    """
    # Collect ridge counts for all fingers
    finger_ridge_counts = []
    
    for finger_code in ["R1", "R2", "R3", "R4", "R5", "L1", "L2", "L3", "L4", "L5"]:
        fp_data = st.session_state.fingerprints.get(finger_code, {})
        analysis = fp_data.get("analysis", {})
        
        if analysis and analysis.get("success"):
            ridge_counts = analysis.get("ridge_counts", [])
            if ridge_counts:
                max_rc = max(ridge_counts)  # Use MAXIMUM instead of average
            else:
                max_rc = 0
        else:
            max_rc = 0
        
        finger_ridge_counts.append((finger_code, max_rc))
    
    # Sort by ridge count (descending - highest first)
    finger_ridge_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Assign ranks (1 for highest, 10 for lowest)
    rankings = {}
    for rank, (finger_code, rc) in enumerate(finger_ridge_counts, start=1):
        rankings[finger_code] = rank
    
    return rankings

def summary_report_page():
    """Display summary report of all captured fingerprints"""
    
    # 1. 获取当前模式
    mode = st.session_state.get("input_mode", "Touchbased")
    
    # 2. 校验数据
    if not all_fingerprints_captured():
        st.error("⚠️ Error: Not all fingerprints are captured. Please return to input page.")
        if st.button("← Back to Fingerprint Input"):
            st.session_state.show_summary = False
            st.rerun()
        return

    # 3. 准备数据：只有 Touchbased 需要计算排名
    ridge_count_rankings = {}
    if mode == "Touchbased":
        ridge_count_rankings = calculate_ridge_count_rankings()

    # 4. Header Section
    header_col1, header_col2, header_col3 = st.columns([2, 3, 2])
    
    with header_col1:
        st.markdown(f"<div style='padding-top: 2rem;'><h2 style='font-family: Inter, sans-serif; font-size: 32px; font-weight: 400; color: #000000; margin: 0;'>{st.session_state.username.upper()}</h2></div>", unsafe_allow_html=True)
    
    with header_col2:
        title_suffix = "(Scanner)" if mode == "Touchbased" else "(Touchless)"
        st.markdown(f"<div style='padding-top: 2rem; text-align: center;'><h1 style='font-family: Inter, sans-serif; font-size: 48px; font-weight: 400; color: #000000; margin: 0;'>Summary Report <br><span style='font-size:20px;color:#666;'>{title_suffix}</span></h1></div>", unsafe_allow_html=True)
    
    with header_col3:
        if Path("WebImages/luminous-logo-withname.png").exists():
            st.markdown("<div style='padding-top: 1.5rem;'></div>", unsafe_allow_html=True)
            st.image(Image.open("WebImages/luminous-logo-withname.png"), width=150)

    # Touchless 专属提示
    if mode == "Touchless":
        st.info("💡 Note: Touchless mode focuses on Pattern Classification. Ridge Counts are excluded.")

    # 5. Main Content (左右手布局)
    st.markdown("<hr style='border: none; border-top: 3px solid #664ED0; margin: 2rem 0 1.5rem 0;'>", unsafe_allow_html=True)
    
    t1, t2 = st.columns(2)
    with t1: st.markdown("<h3 style='text-align:center;font-family:Inter;font-size:32px;color:black;'>Left Brain / Right Hand</h3>", unsafe_allow_html=True)
    with t2: st.markdown("<h3 style='text-align:center;font-family:Inter;font-size:32px;color:black;'>Right Brain / Left Hand</h3>", unsafe_allow_html=True)
    
    st.markdown("<hr style='border: none; border-top: 3px solid #664ED0; margin: 0 0 2rem 0;'>", unsafe_allow_html=True)
    
    l_col, sep, r_col = st.columns([49, 2, 49])
    
    # 关键修改：传入 mode 和 rankings
    with l_col: 
        display_summary_section("right", 5, mode, ridge_count_rankings)
    
    with sep: 
        st.markdown("<div style='width: 3px; height: 100%; min-height: 500px; background: #664ED0; margin: 0 auto; position: relative; top: 0;'></div>", unsafe_allow_html=True)
    
    with r_col: 
        display_summary_section("left", 5, mode, ridge_count_rankings)
    
    # 6. Footer & TRC (逻辑分流)
    st.markdown("<hr style='border: none; border-top: 3px solid #664ED0; margin: 3rem 0 2rem 0;'>", unsafe_allow_html=True)
    
    # [Touchbased] 显示 TRC 总分
    if mode == "Touchbased":
        trc1, trc2 = st.columns(2)
        with trc1: st.markdown(f"<h3 style='text-align:center;'>Left Brain TRC = {calculate_hand_trc('right')}</h3>", unsafe_allow_html=True)
        with trc2: st.markdown(f"<h3 style='text-align:center;'>Right Brain TRC = {calculate_hand_trc('left')}</h3>", unsafe_allow_html=True)
        st.markdown("<hr style='border: none; border-top: 3px solid #664ED0; margin: 2rem 0 2rem 0;'>", unsafe_allow_html=True)
    
    # Back Button
    _, b_col, _ = st.columns([1, 2, 1])
    with b_col:
        if st.button("← Back to Input", use_container_width=True, key="back_btn"):
            st.session_state.show_summary = False
            st.rerun()


def display_summary_section(hand, num_fingers, mode, rankings):
    """
    Display summary section.
    - Touchbased: Shows 5 columns (Image, Finger, Pattern, RC, Rank)
    - Touchless: Shows 2 columns (Image, Info)
    """
    fingers = ["Thumb", "Index", "Middle", "Ring", "Little"]
    finger_codes = ["1", "2", "3", "4", "5"]
    
    pattern_map = {
        "wpe": "Whorl Peacock Eye", "ws": "Whorl Spiral", "wd": "Whorl Double Loop", "we": "Whorl Elongated",
        "lu": "Loop Ulnar", "au": "Loop Radial", "at": "Tented Arch", "as": "Simple Arch"
    }

    for i in range(num_fingers):
        finger_name = fingers[i]
        finger_code = f"{'R' if hand == 'right' else 'L'}{finger_codes[i]}"
        
        # Get data
        fp_data = st.session_state.fingerprints.get(finger_code, {})
        analysis = fp_data.get("analysis", {})
        
        # Basic Info extraction
        pattern_code = "N/A"
        pattern_display = "Pending"
        confidence = 0
        
        if analysis and analysis.get("success"):
            cls = analysis.get("classification", {})
            pattern_code = cls.get("predicted_class", "N/A")
            pattern_display = pattern_map.get(pattern_code, pattern_code.upper())
            confidence = cls.get("confidence", 0) * 100

        with st.container(border=True):
            
            # ====== 分支 A: Touchbased (保持完全原样) ======
            if mode == "Touchbased":
                col1, col2, col3, col4, col5 = st.columns([1, 1.5, 1.5, 1, 1])
                
                # 1. Image
                with col1:
                    if finger_code in st.session_state.fingerprints:
                        try:
                            img_data = base64.b64decode(fp_data["image_base64"])
                            img = Image.open(io.BytesIO(img_data))
                            if st.button("🔍", key=f"v_{finger_code}"): 
                                st.session_state[f"show_{finger_code}"] = not st.session_state.get(f"show_{finger_code}", False)
                            st.image(img, width=60)
                        except: st.write("Error")
                
                # 2. Finger Code
                with col2: 
                    st.markdown(f"<p style='padding-top:10px;'><strong>Finger: {finger_code}</strong></p>", unsafe_allow_html=True)
                
                # 3. Pattern
                with col3: 
                    st.markdown(f"<p style='padding-top:10px;'><strong>{pattern_display}</strong><br><span style='font-size:12px;color:#888'>{confidence:.0f}%</span></p>", unsafe_allow_html=True)
                
                # 4. RC (Touchbased 专属)
                rc = max(analysis.get("ridge_counts", []) or [0]) if analysis else 0
                with col4: 
                    st.markdown(f"<p style='padding-top:10px;'><strong>RC</strong><br>{rc}</p>", unsafe_allow_html=True)
                
                # 5. Rank (Touchbased 专属)
                rank = rankings.get(finger_code, "-")
                with col5: 
                    st.markdown(f"<p style='padding-top:10px;'><strong>Rank</strong><br>{rank}</p>", unsafe_allow_html=True)

            # ====== 分支 B: Touchless (简化版) ======
            else:
                col1, col2 = st.columns([1, 3])
                
                # 1. Image
                with col1:
                    if finger_code in st.session_state.fingerprints:
                        try:
                            img_data = base64.b64decode(fp_data["image_base64"])
                            img = Image.open(io.BytesIO(img_data))
                            if st.button("🔍", key=f"v_{finger_code}"): 
                                st.session_state[f"show_{finger_code}"] = not st.session_state.get(f"show_{finger_code}", False)
                            st.image(img, width=80)
                        except: st.write("Err")
                
                # 2. Info (Combined Finger + Pattern)
                with col2:
                    color = "#10B981" if confidence > 80 else "#3B82F6"
                    st.markdown(f"""
                    <div style='margin-top:5px;'>
                        <span style='font-weight:600; font-size:16px;'>{finger_code} - {finger_name}</span><br>
                        <span style='font-size:22px; font-weight:700; color:{color};'>{pattern_display}</span>
                        <span style='font-size:12px; color:#999; margin-left:8px;'>({confidence:.0f}%)</span>
                    </div>
                    """, unsafe_allow_html=True)

            # === Overlay Expand (通用) ===
            if st.session_state.get(f"show_{finger_code}", False):
                if analysis.get("overlay_base64"):
                    try:
                        d = analysis["overlay_base64"]
                        if "data:image" in d: d = d.split(",")[1]
                        st.image(Image.open(io.BytesIO(base64.b64decode(d))), caption=f"{finger_code} Overlay", use_container_width=True)
                    except: st.error("Overlay error")
# Helper function to calculate total ridge count for a hand
def calculate_hand_trc(hand):
    """Calculate total ridge count for a hand (5 fingers)"""
    total_rc = 0
    finger_codes = ["1", "2", "3", "4", "5"]
    
    for i in range(5):
        # Use proper finger code (R1-R5 for right, L1-L5 for left)
        finger_code = f"{'R' if hand == 'right' else 'L'}{finger_codes[i]}"
        fp_data = st.session_state.fingerprints.get(finger_code, {})
        analysis = fp_data.get("analysis", {})
        
        if analysis and analysis.get("success"):
            ridge_counts = analysis.get("ridge_counts", [])
            if ridge_counts:
                # Use MAXIMUM ridge count for this finger
                max_rc = max(ridge_counts)
                total_rc += max_rc
    
    # Return formatted to 1 decimal place
    return f"{total_rc:.1f}"

# Main App
def main():
    load_css()
    init_session_state()
    
    if st.session_state.logged_in:
        show_api_config()
    
    # 页面跳转逻辑
    if not st.session_state.logged_in:
        login_page()
    elif st.session_state.input_mode is None:
        # [新增] 如果已登录但未选择模式，显示选择页
        mode_selection_page()
    elif st.session_state.show_summary:
        summary_report_page()
    else:
        dashboard_page()

if __name__ == "__main__":
    main()
