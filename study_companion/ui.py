from __future__ import annotations

from html import escape

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #111827;
            --muted: #5b6472;
            --line: #d8dee8;
            --panel: #ffffff;
            --soft: #f6f8fb;
            --blue: #2563eb;
            --green: #0f8a6b;
            --amber: #b45309;
            --rose: #be123c;
        }

        .stApp {
            background:
                linear-gradient(180deg, #f8fafc 0%, #eef4fb 48%, #f8fafc 100%);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1240px;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            padding: 0.45rem 0.55rem;
            border-radius: 8px;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(37, 99, 235, 0.08);
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        .hero {
            border: 1px solid var(--line);
            background: linear-gradient(135deg, #ffffff 0%, #f7fbff 52%, #f1fbf7 100%);
            border-radius: 8px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1.2rem;
        }

        .hero-title {
            font-size: 2rem;
            line-height: 1.1;
            font-weight: 800;
            color: var(--ink);
            margin: 0 0 0.45rem 0;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 1rem;
            max-width: 780px;
            margin: 0;
        }

        .surface {
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .surface h3 {
            font-size: 1rem;
            margin: 0 0 0.4rem 0;
        }

        .surface p {
            color: var(--muted);
            margin: 0.2rem 0 0 0;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(5, minmax(120px, 1fr));
            gap: 0.75rem;
            margin: 0.8rem 0 1.2rem 0;
        }

        .metric-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            padding: 0.85rem 0.9rem;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.8rem;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            color: var(--ink);
            font-size: 1.55rem;
            line-height: 1.1;
            font-weight: 800;
        }

        .pill {
            display: inline-block;
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--ink);
            background: #f8fafc;
        }

        .pill.high {
            color: var(--rose);
            border-color: #fecdd3;
            background: #fff1f2;
        }

        .pill.medium {
            color: var(--amber);
            border-color: #fed7aa;
            background: #fff7ed;
        }

        .pill.good {
            color: var(--green);
            border-color: #bbf7d0;
            background: #f0fdf4;
        }

        .source-box {
            border-left: 3px solid var(--blue);
            background: #f8fafc;
            padding: 0.75rem 0.9rem;
            margin: 0.55rem 0;
            border-radius: 0 8px 8px 0;
        }

        .empty-state {
            border: 1px dashed #b9c3d4;
            background: rgba(255, 255, 255, 0.7);
            border-radius: 8px;
            padding: 1rem;
            color: var(--muted);
        }

        .sidebar-brand {
            padding: 0.6rem 0 1.1rem 0;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1rem;
        }

        .sidebar-brand h2 {
            color: var(--ink);
            font-size: 1.15rem;
            margin: 0;
        }

        .sidebar-brand p {
            color: var(--muted);
            margin: 0.25rem 0 0 0;
            font-size: 0.82rem;
        }

        .stButton > button {
            border-radius: 8px;
            font-weight: 700;
        }

        .stTextInput input, .stTextArea textarea, .stNumberInput input {
            border-radius: 8px;
        }

        @media (max-width: 900px) {
            .metric-strip {
                grid-template-columns: repeat(2, minmax(120px, 1fr));
            }
            .hero-title {
                font-size: 1.55rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-title">{escape(title)}</div>
            <p class="hero-copy">{escape(copy)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <h2>AI Study Companion</h2>
            <p>Local RAG, quizzes, analytics, and study planning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_strip(items: list[tuple[str, str]]) -> None:
    cards = "".join(
        f'<div class="metric-card"><div class="metric-label">{escape(label)}</div><div class="metric-value">{escape(value)}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="metric-strip">{cards}</div>', unsafe_allow_html=True)


def surface(title: str, body: str, tone: str | None = None) -> None:
    tone_class = f" {tone}" if tone else ""
    st.markdown(
        f"""
        <div class="surface">
            <span class="pill{tone_class}">{escape(title)}</span>
            <p>{escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(message: str) -> None:
    st.markdown(f'<div class="empty-state">{escape(message)}</div>', unsafe_allow_html=True)


def source_box(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="source-box">
            <strong>{escape(title)}</strong>
            <p>{escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(label: str, tone: str = "") -> str:
    tone_class = f" {tone}" if tone else ""
    return f'<span class="pill{tone_class}">{escape(label)}</span>'
