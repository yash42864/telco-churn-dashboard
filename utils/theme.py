"""Theme and styling configuration for Churn Analytics Dashboard.

Implements a dark 'space/orbit' theme with a crisp light-mode variant.
Provides Plotly chart templates, custom CSS injection, and responsive UI components.
"""

from typing import Dict, Any, Optional
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# Consistent semantic color palette
COLOR_CHURN = "#F43F5E"        # Coral / Crimson (churn, risk, loss)
COLOR_SAFE = "#10B981"         # Mint / Emerald (retained, safe, profit)
COLOR_PRIMARY = "#818CF8"      # Periwinkle / Indigo
COLOR_AMBER = "#F59E0B"        # Amber / Warning
COLOR_PURPLE = "#A855F7"       # Electric Purple
COLOR_SKY = "#38BDF8"          # Vivid Sky Blue
COLOR_SLATE_DARK = "#1E293B"   # Dark card fill
COLOR_SLATE_LIGHT = "#F1F5F9"  # Light card fill

FONT_FAMILY = "Sora, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


def get_current_theme() -> str:
    """Return active theme mode ('dark' or 'light')."""
    try:
        if "theme_mode" not in st.session_state:
            st.session_state["theme_mode"] = "dark"
        return st.session_state["theme_mode"]
    except Exception:
        return "dark"


def set_current_theme(mode: str) -> None:
    """Set active theme mode."""
    try:
        st.session_state["theme_mode"] = mode
    except Exception:
        pass


def apply_custom_theme() -> None:
    """Inject Google Fonts and custom CSS for the active theme."""
    theme = get_current_theme()
    is_dark = (theme == "dark")

    bg_main = "#0B0F19" if is_dark else "#F8FAFC"
    bg_card = "rgba(22, 31, 48, 0.75)" if is_dark else "rgba(255, 255, 255, 0.9)"
    border_card = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(0, 0, 0, 0.08)"
    text_primary = "#F8FAFC" if is_dark else "#0F172A"
    text_secondary = "#94A3B8" if is_dark else "#64748B"
    hover_shadow = "0 12px 24px -10px rgba(0, 0, 0, 0.4)" if is_dark else "0 12px 20px -8px rgba(0, 0, 0, 0.1)"
    sidebar_bg = "#070A11" if is_dark else "#F1F5F9"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: {FONT_FAMILY};
    }}

    /* Top padding reduction */
    .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1400px;
    }}

    /* App background */
    .stApp {{
        background-color: {bg_main};
        color: {text_primary};
    }}

    /* Glass Cards */
    .glass-card {{
        background: {bg_card};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {border_card};
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }}

    @media (prefers-reduced-motion: no-preference) {{
        .glass-card:hover {{
            transform: translateY(-2px);
            box-shadow: {hover_shadow};
            border-color: rgba(129, 140, 248, 0.3);
        }}
    }}

    /* Card Titles and Text */
    .card-title {{
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {text_secondary};
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .card-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: {text_primary};
        letter-spacing: -0.02em;
        line-height: 1.2;
    }}

    .card-delta {{
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 0.3rem;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
    }}

    .delta-coral {{ color: {COLOR_CHURN}; }}
    .delta-mint {{ color: {COLOR_SAFE}; }}
    .delta-neutral {{ color: {text_secondary}; }}

    /* Badges and Pills */
    .badge-pill {{
        display: inline-block;
        padding: 0.2rem 0.65rem;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 9999px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }}

    .badge-primary {{
        background: rgba(129, 140, 248, 0.15);
        color: {COLOR_PRIMARY};
        border: 1px solid rgba(129, 140, 248, 0.3);
    }}

    .badge-danger {{
        background: rgba(244, 63, 94, 0.15);
        color: {COLOR_CHURN};
        border: 1px solid rgba(244, 63, 94, 0.3);
    }}

    .badge-success {{
        background: rgba(16, 185, 129, 0.15);
        color: {COLOR_SAFE};
        border: 1px solid rgba(16, 185, 129, 0.3);
    }}

    .badge-amber {{
        background: rgba(245, 158, 11, 0.15);
        color: {COLOR_AMBER};
        border: 1px solid rgba(245, 158, 11, 0.3);
    }}

    /* Page Header Styles */
    .header-container {{
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid {border_card};
    }}

    .page-title {{
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: {text_primary};
        margin: 0.2rem 0 0.4rem 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }}

    .page-subtitle {{
        font-size: 0.95rem;
        color: {text_secondary};
        font-weight: 400;
        margin: 0;
    }}

    /* Streamlit Metric Overrides */
    [data-testid="stMetricValue"] {{
        font-family: {FONT_FAMILY};
        font-weight: 700;
    }}

    /* Sidebar Footer Branding */
    .sidebar-footer {{
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid {border_card};
        font-size: 0.78rem;
        color: {text_secondary};
    }}

    /* Streamlit Buttons */
    .stButton > button {{
        border-radius: 8px;
        font-weight: 600;
        font-family: {FONT_FAMILY};
        transition: all 0.2s ease;
    }}

    .stButton > button:hover {{
        border-color: {COLOR_PRIMARY};
        color: {COLOR_PRIMARY};
    }}

    /* Navigation Card */
    .nav-card {{
        background: {bg_card};
        border: 1px solid {border_card};
        border-radius: 12px;
        padding: 1.1rem;
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
        display: block;
        margin-bottom: 0.75rem;
    }}

    .nav-card:hover {{
        border-color: {COLOR_PRIMARY};
        transform: translateY(-2px);
        box-shadow: {hover_shadow};
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def get_plotly_template(theme_name: Optional[str] = None) -> go.layout.Template:
    """Return a unified Plotly template matching the active theme."""
    theme = theme_name if theme_name is not None else get_current_theme()
    is_dark = (theme == "dark")

    text_color = "#F8FAFC" if is_dark else "#0F172A"
    grid_color = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(0, 0, 0, 0.06)"
    hover_bg = "#1E293B" if is_dark else "#FFFFFF"
    hover_border = "rgba(255, 255, 255, 0.15)" if is_dark else "rgba(0, 0, 0, 0.1)"

    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT_FAMILY, color=text_color, size=12),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        colorway=[COLOR_CHURN, COLOR_SAFE, COLOR_PRIMARY, COLOR_AMBER, COLOR_PURPLE, COLOR_SKY],
        xaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            tickfont=dict(size=11, color=text_color),
            title=dict(font=dict(size=12, color=text_color)),
        ),
        yaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            tickfont=dict(size=11, color=text_color),
            title=dict(font=dict(size=12, color=text_color)),
        ),
        hoverlabel=dict(
            bgcolor=hover_bg,
            bordercolor=hover_border,
            font=dict(family=FONT_FAMILY, color=text_color, size=12),
        ),
        margin=dict(t=40, b=40, l=40, r=30),
        legend=dict(
            font=dict(size=11, color=text_color),
            bgcolor="rgba(0, 0, 0, 0)",
        ),
    )
    return template


def render_page_header(
    title: str,
    subtitle: str,
    badge: str = "PySpark ML",
    icon: str = "📊",
    how_to_read: Optional[str] = None,
) -> None:
    """Render a consistent, executive-style page header with expandable context."""
    badge_html = f'<span class="badge-pill badge-primary">{badge}</span>'
    html = f"""
    <div class="header-container">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">
            <h1 class="page-title">{icon} {title}</h1>
            {badge_html}
        </div>
        <p class="page-subtitle">{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    if how_to_read:
        with st.expander("ℹ️ How to read and interpret this page", expanded=False):
            st.markdown(how_to_read)


def render_kpi_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    delta_type: str = "neutral",  # 'coral', 'mint', 'neutral'
    icon: str = "",
    sublabel: str = "",
) -> None:
    """Render a premium glass KPI card with delta annotation."""
    delta_html = ""
    if delta:
        delta_class = f"delta-{delta_type}"
        delta_html = f'<div class="card-delta {delta_class}">{delta}</div>'

    sublabel_html = f'<div style="font-size: 0.72rem; color: #94A3B8; margin-top: 0.2rem;">{sublabel}</div>' if sublabel else ""

    html = f"""
    <div class="glass-card">
        <div class="card-title">{icon} {title}</div>
        <div class="card-value">{value}</div>
        {delta_html}
        {sublabel_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar_footer() -> None:
    """Render persistent sidebar info, theme switcher, and cache reset."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ App Controls")

        current_theme = get_current_theme()
        new_theme = st.radio(
            "Visual Theme",
            options=["dark", "light"],
            index=0 if current_theme == "dark" else 1,
            format_func=lambda x: "🌙 Dark Space" if x == "dark" else "☀️ Crisp Light",
            horizontal=True,
            key="theme_radio",
        )
        if new_theme != current_theme:
            set_current_theme(new_theme)
            st.rerun()

        if st.button("🔄 Reload Data Cache", use_container_width=True, help="Clears Streamlit cached datasets and reloads from outputs/"):
            st.cache_data.clear()
            st.toast("✅ Data cache cleared successfully!", icon="🧹")
            st.rerun()

        st.markdown(
            """
            <div class="sidebar-footer">
                <div style="font-weight: 600; color: #818CF8; margin-bottom: 0.2rem;">
                    ⚡ PySpark Churn Analytics
                </div>
                <div>Dataset: <b>IBM Telco Customer Churn</b></div>
                <div>Scale: <b>7,043 customers · 21 features</b></div>
                <div style="margin-top: 0.4rem; font-size: 0.72rem; opacity: 0.75;">
                    Engine: Apache Spark (local[*] · 8 cores)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
