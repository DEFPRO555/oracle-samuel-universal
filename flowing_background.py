"""
Flowing Background Module for Oracle Samuel Application

This module provides functions for creating flowing blue lines background
and styled components for the Streamlit application.
"""

import streamlit as st
from typing import Optional


def apply_flowing_background():
    """
    Apply flowing blue lines background to the Streamlit app.
    This function adds CSS for animated background effects.
    """
    flowing_css = """
    <style>
    .flowing-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        overflow: hidden;
    }
    
    .flowing-line {
        position: absolute;
        background: linear-gradient(90deg, transparent, rgba(0, 123, 255, 0.3), transparent);
        animation: flow 8s linear infinite;
    }
    
    .flowing-line:nth-child(1) {
        top: 20%;
        left: -100%;
        width: 200%;
        height: 2px;
        animation-delay: 0s;
    }
    
    .flowing-line:nth-child(2) {
        top: 40%;
        left: -100%;
        width: 150%;
        height: 1px;
        animation-delay: 2s;
    }
    
    .flowing-line:nth-child(3) {
        top: 60%;
        left: -100%;
        width: 180%;
        height: 1.5px;
        animation-delay: 4s;
    }
    
    .flowing-line:nth-child(4) {
        top: 80%;
        left: -100%;
        width: 120%;
        height: 1px;
        animation-delay: 6s;
    }
    
    @keyframes flow {
        0% {
            transform: translateX(0);
        }
        100% {
            transform: translateX(100vw);
        }
    }
    </style>
    
    <div class="flowing-background">
        <div class="flowing-line"></div>
        <div class="flowing-line"></div>
        <div class="flowing-line"></div>
        <div class="flowing-line"></div>
    </div>
    """
    
    st.markdown(flowing_css, unsafe_allow_html=True)


def flowing_header(text: str, level: int = 1) -> None:
    """
    Create a styled header with flowing background effect.
    
    Args:
        text (str): Header text
        level (int): Header level (1-6)
    """
    header_css = f"""
    <style>
    .flowing-header-{level} {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
        margin: 1rem 0;
        text-align: center;
        position: relative;
    }}
    
    .flowing-header-{level}::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(0, 123, 255, 0.1), transparent);
        animation: shimmer 3s ease-in-out infinite;
        z-index: -1;
    }}
    
    @keyframes shimmer {{
        0%, 100% {{ opacity: 0; }}
        50% {{ opacity: 1; }}
    }}
    </style>
    """
    
    st.markdown(header_css, unsafe_allow_html=True)
    st.markdown(f"<h{level} class='flowing-header-{level}'>{text}</h{level}>", unsafe_allow_html=True)


def flowing_card(content: str, title: Optional[str] = None) -> None:
    """
    Create a styled card with flowing background effect.
    
    Args:
        content (str): Card content
        title (Optional[str]): Card title
    """
    card_css = """
    <style>
    .flowing-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .flowing-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 123, 255, 0.1), transparent);
        animation: card-flow 4s ease-in-out infinite;
    }
    
    .flowing-card-title {
        color: #007bff;
        font-weight: bold;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    @keyframes card-flow {
        0%, 100% { left: -100%; }
        50% { left: 100%; }
    }
    </style>
    """
    
    st.markdown(card_css, unsafe_allow_html=True)
    
    if title:
        st.markdown(f"<div class='flowing-card'><div class='flowing-card-title'>{title}</div>{content}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='flowing-card'>{content}</div>", unsafe_allow_html=True)


def flowing_metric(label: str, value: str, delta: Optional[str] = None) -> None:
    """
    Create a styled metric with flowing background effect.
    
    Args:
        label (str): Metric label
        value (str): Metric value
        delta (Optional[str]): Delta value (change indicator)
    """
    metric_css = """
    <style>
    .flowing-metric {
        background: linear-gradient(135deg, rgba(0, 123, 255, 0.1), rgba(0, 123, 255, 0.05));
        border: 1px solid rgba(0, 123, 255, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .flowing-metric::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 123, 255, 0.2), transparent);
        animation: metric-flow 3s ease-in-out infinite;
    }
    
    .flowing-metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }
    
    .flowing-metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #007bff;
        margin-bottom: 0.25rem;
    }
    
    .flowing-metric-delta {
        font-size: 0.8rem;
        color: #28a745;
    }
    
    @keyframes metric-flow {
        0%, 100% { left: -100%; }
        50% { left: 100%; }
    }
    </style>
    """
    
    st.markdown(metric_css, unsafe_allow_html=True)
    
    delta_html = f"<div class='flowing-metric-delta'>{delta}</div>" if delta else ""
    metric_html = f"""
    <div class='flowing-metric'>
        <div class='flowing-metric-label'>{label}</div>
        <div class='flowing-metric-value'>{value}</div>
        {delta_html}
    </div>
    """
    
    st.markdown(metric_html, unsafe_allow_html=True)
