"""
BreastGuard AI — Dashboard UI Components
================================================================================
Reusable UI elements: KPI cards, page headers, status badges, and disclaimer banners.
Designed with modern typography and dark theme color hierarchy.
================================================================================
"""
from __future__ import annotations

import streamlit as st
from config.config import THEME_DARK


def render_metric_card(
    title: str,
    value: str | float,
    subtitle: str = "",
    color: str = "#6C63FF",
    icon: str = "",
) -> None:
    """Render a styled dark-mode metric card."""
    icon_html = f"<span style='font-size: 1.4rem; margin-right: 0.5rem;'>{icon}</span>" if icon else ""
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #161B26 0%, #1E2433 100%);
                    border: 1px solid #2D3748; border-radius: 10px;
                    padding: 1.1rem 1.25rem; margin-bottom: 0.8rem;
                    box-shadow: 0 4px 14px rgba(0,0,0,0.25);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.35rem;">
                <span style="font-size: 0.78rem; color: #94A3B8; text-transform: uppercase;
                            letter-spacing: 0.06em; font-weight: 600;">
                    {title}
                </span>
                {icon_html}
            </div>
            <div style="font-size: 1.95rem; font-weight: 700; color: {color}; line-height: 1.2; margin: 0.2rem 0;">
                {value}
            </div>
            <div style="font-size: 0.8rem; color: #A0AEC0; font-weight: 400;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    title: str,
    subtitle: str,
    icon: str = "🩺",
    category: str = "CLINICAL ONCOLOGY AI",
) -> None:
    """Render a standardized modern page header with category pill."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem; padding-bottom: 0.75rem; border-bottom: 1px solid #232B3E;">
            <div style="display: inline-block; padding: 0.25rem 0.65rem; border-radius: 20px;
                        background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.35);
                        font-size: 0.72rem; font-weight: 700; color: #A5B4FC; letter-spacing: 0.08em;
                        text-transform: uppercase; margin-bottom: 0.5rem;">
                {category}
            </div>
            <h1 style="font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem;
                       font-weight: 700; color: #F8FAFC; margin: 0.2rem 0 0.4rem 0;">
                {icon} {title}
            </h1>
            <p style="color: #94A3B8; font-size: 0.98rem; margin: 0; line-height: 1.5;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str = "", icon: str = "") -> None:
    """Render a section header with icon and optional description."""
    icon_str = f"{icon} " if icon else ""
    st.markdown(
        f"""
        <div style="margin: 1.4rem 0 0.8rem 0;">
            <h3 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem;
                       font-weight: 600; color: #E2E8F0; margin: 0 0 0.2rem 0;">
                {icon_str}{title}
            </h3>
            {f'<p style="color: #718096; font-size: 0.85rem; margin: 0;">{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer_banner(compact: bool = False) -> None:
    """Render the standardized medical AI disclaimer banner."""
    if compact:
        st.markdown(
            """
            <div style="background: rgba(239, 68, 68, 0.08); border-left: 3px solid #EF4444;
                        border-radius: 4px; padding: 0.5rem 0.8rem; margin-bottom: 1rem;
                        color: #FCA5A5; font-size: 0.8rem; line-height: 1.4;">
                <strong>⚠️ Research Prototype:</strong> Algorithmic demonstration only. Not for clinical diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background: linear-gradient(90deg, #1E1218 0%, #29151F 100%);
                        border: 1px solid #7F1D1D; border-left: 4px solid #EF4444;
                        border-radius: 8px; padding: 0.75rem 1.2rem; margin-bottom: 1.25rem;
                        color: #FECACA; font-size: 0.85rem; line-height: 1.5;">
                <strong>⚠️ Educational & Academic Research Prototype:</strong>
                This application is developed strictly for competition and academic research evaluation.
                Model outputs represent statistical pattern associations on digitized nuclei measurements
                and must not be used as clinical diagnostic or therapeutic guidance.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_info_callout(text: str, title: str = "Clinical Context", icon: str = "💡") -> None:
    """Render a styled information callout box."""
    st.markdown(
        f"""
        <div style="background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.25);
                    border-radius: 8px; padding: 0.85rem 1.15rem; margin: 0.8rem 0;
                    color: #BFDBFE; font-size: 0.88rem; line-height: 1.5;">
            <div style="font-weight: 600; color: #93C5FD; margin-bottom: 0.25rem;">
                {icon} {title}
            </div>
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )
