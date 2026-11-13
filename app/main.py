"""
BreastGuard AI — Streamlit Application Orchestrator
================================================================================
Entry point configuring multi-page navigation, dark theme styling,
global caching, and error boundaries.
================================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is available in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.sidebar import render_sidebar
from app.pages import (
    about,
    dataset_analysis,
    explainable_ai,
    model_performance,
    overview,
    prediction_page,
    responsible_ai,
)
from config.config import PROJECT_NAME, THEME_DARK

# Set page configuration
st.set_page_config(
    page_title=f"{PROJECT_NAME} — Trustworthy AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global Modern Dark Theme Styling
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background-color: {THEME_DARK['bg']};
        color: {THEME_DARK['text_main']};
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {THEME_DARK['surface']};
        border-right: 1px solid {THEME_DARK['border']};
    }}

    /* Headers */
    h1, h2, h3, h4 {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600;
        color: #F8FAFC;
    }}

    /* Card Containers */
    [data-testid="stMetric"] {{
        background-color: {THEME_DARK['surface']};
        border: 1px solid {THEME_DARK['border']};
        border-radius: 8px;
        padding: 0.8rem 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background-color: transparent;
        margin-bottom: 1rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px;
        padding: 8px 18px;
        background-color: {THEME_DARK['surface']};
        border: 1px solid {THEME_DARK['border']};
        color: #94A3B8;
        font-weight: 500;
        font-size: 0.9rem;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        border-color: #3B82F6 !important;
        font-weight: 600;
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        border-color: #3B82F6;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
    }}

    /* Dataframe tables */
    .stDataFrame {{
        border: 1px solid {THEME_DARK['border']};
        border-radius: 8px;
        overflow: hidden;
    }}

    /* Radio button in sidebar */
    [data-testid="stSidebar"] .stRadio > div {{
        gap: 4px;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        padding: 6px 12px;
        border-radius: 6px;
        transition: background 0.15s ease;
    }}
    [data-testid="stSidebar"] .stRadio label:hover {{
        background: rgba(255, 255, 255, 0.05);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Configure modern multi-page navigation using st.navigation
pages = {
    "Clinical Analytics": [
        st.Page(overview.render_page, title="Executive Overview", icon="📊", url_path="overview", default=True),
        st.Page(dataset_analysis.render_page, title="Dataset Analysis", icon="🔬", url_path="dataset_analysis"),
        st.Page(model_performance.render_page, title="Model Performance", icon="🤖", url_path="model_performance"),
        st.Page(prediction_page.render_page, title="Breast Cancer Prediction", icon="🔮", url_path="prediction_page"),
    ],
    "Trustworthy AI & Governance": [
        st.Page(explainable_ai.render_page, title="Explainable AI (SHAP)", icon="🧠", url_path="explainable_ai"),
        st.Page(responsible_ai.render_page, title="Responsible AI & Reliability", icon="🛡️", url_path="responsible_ai"),
        st.Page(about.render_page, title="About & Governance", icon="⚖️", url_path="about"),
    ],
}

pg = st.navigation(pages)

# Render Global Sidebar branding, model specs, and medical safety disclaimer
render_sidebar(include_nav=False)

# Execute the active page safely
try:
    pg.run()
except Exception as e:
    st.error(f"Error rendering active page: {e}")
    st.exception(e)
