import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime

# ==============================================================================
# THEME SYSTEM
# ==============================================================================
THEMES = {
    "dark": {
        "bg":        "#0a0a0f",
        "surface":   "#111118",
        "surface2":  "#1a1a24",
        "border":    "#2a2a3a",
        "text":      "#e8e8f0",
        "muted":     "#888899",
        "primary":   "#7c6af7",
        "secondary": "#f7c26a",
        "accent":    "#6af7c2",
        "danger":    "#db2121",
        "info":      "#6aa8f7",
        "highlight": "#f76af7",
        "success":   "#aaf76a",
    },
    "light": {
        "bg":        "#f8f9fc",
        "surface":   "#ffffff",
        "surface2":  "#f1f3f9",
        "border":    "#e0e3eb",
        "text":      "#1f2937",
        "muted":     "#6b7280",
        "primary":   "#5b4df5",
        "secondary": "#f4b740",
        "accent":    "#22c7a9",
        "danger":    "#cf2929",
        "info":      "#3b82f6",
        "highlight": "#d946ef",
        "success":   "#84cc16",
    },
}

PALETTES = {
    "dark":  ["#7c6af7", "#f7c26a", "#6af7c2", "#f76a6a", "#6aa8f7", "#f76af7", "#aaf76a"],
    "light": ["#5b4df5", "#f4b740", "#22c7a9", "#e54848", "#3b82f6", "#d946ef", "#84cc16"],
}


def get_theme():
    mode = st.session_state.get("theme_mode", "dark")
    return THEMES[mode], PALETTES[mode], mode


def inject_css(t, mode):
    """Inject all CSS using the active theme dict `t`."""
    scrollbar_thumb = t["primary"]
    scrollbar_track = t["surface2"]
    input_shadow    = f"0 0 0 2px {t['primary']}40"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {{
    --bg:        {t["bg"]};
    --surface:   {t["surface"]};
    --surface2:  {t["surface2"]};
    --border:    {t["border"]};
    --text:      {t["text"]};
    --muted:     {t["muted"]};
    --primary:   {t["primary"]};
    --secondary: {t["secondary"]};
    --accent:    {t["accent"]};
    --danger:    {t["danger"]};
    --info:      {t["info"]};
    --highlight: {t["highlight"]};
    --success:   {t["success"]};
}}

/* ── Global ── */
html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}}
.stApp {{ background: var(--bg) !important; }}
.main .block-container {{ padding-top: 1.5rem; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}}
section[data-testid="stSidebar"] * {{ color: var(--text) !important; }}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {{
    color: var(--primary) !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: var(--surface2);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: var(--muted);
    border-radius: 7px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    padding: 8px 20px;
    border: none;
    transition: color 0.15s;
}}
.stTabs [aria-selected="true"] {{
    background: var(--primary) !important;
    color: #ffffff !important;
}}

/* ── Metrics ── */
div[data-testid="metric-container"] {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    transition: border-color 0.2s;
}}
div[data-testid="metric-container"]:hover {{
    border-color: var(--primary);
}}
div[data-testid="metric-container"] label {{
    color: var(--muted) !important;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
    color: var(--secondary) !important;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem !important;
    font-weight: 700;
}}
div[data-testid="stMetricDelta"] {{ color: var(--accent) !important; }}

/* ── Buttons ── */
.stButton > button {{
    background: var(--primary) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.76rem !important;
    letter-spacing: 0.05em !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}}
.stButton > button:hover {{
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px {t["primary"]}44 !important;
}}
.stButton > button:active {{
    transform: translateY(0px) !important;
}}

/* ── Inputs ── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input,
.stTextArea textarea {{
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within,
.stTextInput > div > div:focus-within {{
    box-shadow: {input_shadow} !important;
    border-color: var(--primary) !important;
}}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] [role="slider"] {{
    background-color: var(--primary) !important;
    border-color: var(--primary) !important;
}}
.stSlider [data-baseweb="slider"] div[style*="background"] {{
    background: var(--primary) !important;
}}

/* ── Radio / Checkbox ── */
.stRadio label span, .stCheckbox label span {{
    color: var(--text) !important;
}}

/* ── Dataframe ── */
.stDataFrame {{
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
}}
.stDataFrame thead th {{
    background: var(--surface2) !important;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}

/* ── Expander ── */
.streamlit-expanderHeader {{
    background: var(--surface2) !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    transition: border-color 0.2s;
}}
.streamlit-expanderHeader:hover {{
    border-color: var(--primary) !important;
}}
.streamlit-expanderContent {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}}

/* ── Alerts ── */
.stAlert {{
    border-radius: 10px !important;
    border-width: 1px !important;
}}
div[data-baseweb="notification"][kind="positive"] {{
    background: {t["success"]}18 !important;
    border-color: {t["success"]}60 !important;
    color: var(--text) !important;
}}
div[data-baseweb="notification"][kind="warning"] {{
    background: {t["secondary"]}18 !important;
    border-color: {t["secondary"]}60 !important;
    color: var(--text) !important;
}}
div[data-baseweb="notification"][kind="negative"] {{
    background: {t["danger"]}18 !important;
    border-color: {t["danger"]}60 !important;
    color: var(--text) !important;
}}
div[data-baseweb="notification"][kind="info"] {{
    background: {t["info"]}18 !important;
    border-color: {t["info"]}60 !important;
    color: var(--text) !important;
}}

/* ── Divider ── */
hr {{ border-color: var(--border) !important; opacity: 1; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {scrollbar_track}; border-radius: 3px; }}
::-webkit-scrollbar-thumb {{ background: {scrollbar_thumb}80; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {scrollbar_thumb}; }}

/* ── Custom components ── */
.logo-area {{
    <div style="
        font-family: 'Space Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    ">
        <span style="color:var(--primary)">Data</span>
        <span style="color:{'f7c26a'}">Sci</span>
    </div>
}}
.logo-area span {{ color: var(--secondary); }}
.logo-tagline {{
    font-size: 0.7rem;
    color: var(--muted);
    font-family: 'DM Sans', sans-serif;
    margin-top: 2px;
    letter-spacing: 0.03em;
}}

.sec-header {{
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 20px 0 10px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 7px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.badge {{
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 11px;
    font-family: 'Space Mono', monospace;
    font-size: 0.66rem;
    color: var(--accent);
    margin: 2px;
    letter-spacing: 0.03em;
}}
.badge-num {{ color: var(--secondary); border-color: {t["secondary"]}44; }}
.badge-cat {{ color: var(--primary); border-color: {t["primary"]}44; }}
.badge-info {{ color: var(--info); border-color: {t["info"]}44; }}

.log-panel {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--accent);
    max-height: 180px;
    overflow-y: auto;
    line-height: 1.8;
}}

.theme-toggle-label {{
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
}}

.result-card {{
    border-radius: 12px;
    padding: 18px 22px;
    margin-top: 12px;
    border-width: 1px;
    border-style: solid;
}}
.result-card-label {{
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.result-card-value {{
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.2;
}}
.result-card-sub {{
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 4px;
}}

.model-banner {{
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 12px;
}}
.model-banner-label {{
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 4px;
}}
.model-banner-name {{
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text);
}}

.page-title {{
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
}}
.page-subtitle {{
    color: var(--muted);
    font-size: 0.85rem;
    margin-bottom: 20px;
}}
</style>
""", unsafe_allow_html=True)


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {msg}")
    if len(st.session_state.logs) > 80:
        st.session_state.logs = st.session_state.logs[-80:]


def style_fig(fig):
    """Apply active theme colors to a matplotlib figure."""
    t, pal, _ = get_theme()
    fig.patch.set_facecolor(t["surface"])
    for ax in fig.get_axes():
        ax.set_facecolor(t["bg"])
        ax.tick_params(colors=t["muted"], labelsize=9)
        ax.xaxis.label.set_color(t["muted"])
        ax.yaxis.label.set_color(t["muted"])
        ax.title.set_color(t["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(t["border"])
        # Legend
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor(t["surface2"])
            leg.get_frame().set_edgecolor(t["border"])
            for txt in leg.get_texts():
                txt.set_color(t["text"])
    return fig


def themed_bar_chart(results_df, metrics_to_plot, title="Model Comparison"):
    t, pal, _ = get_theme()
    fig, ax = plt.subplots(figsize=(max(8, len(results_df) * 2), 5))
    x     = np.arange(len(results_df))
    width = 0.8 / max(len(metrics_to_plot), 1)

    for j, metric in enumerate(metrics_to_plot):
        offset = (j - len(metrics_to_plot) / 2) * width + width / 2
        bars = ax.bar(
            x + offset, results_df[metric], width,
            label=metric, color=pal[j % len(pal)], alpha=0.88,
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom",
                fontsize=7, color=t["text"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(results_df.index, rotation=15, ha="right", color=t["text"])
    ax.set_ylim(0, max(results_df[metrics_to_plot].max().max() * 1.25, 1.15))
    ax.set_ylabel("Score", color=t["muted"])
    ax.set_title(title, color=t["text"], fontsize=11, fontfamily="monospace")
    ax.tick_params(colors=t["muted"])
    return style_fig(fig)


def result_card_html(label, value, sub, bg_color, border_color, label_color):
    return (
        f"<div class='result-card' style='background:{bg_color};border-color:{border_color};'>"
        f"<div class='result-card-label' style='color:{label_color};'>{label}</div>"
        f"<div class='result-card-value'>{value}</div>"
        f"<div class='result-card-sub'>{sub}</div>"
        f"</div>"
    )


def model_banner_html(label, name, bg, border, label_color):
    return (
        f"<div class='model-banner' style='background:{bg};border:1px solid {border};'>"
        f"<div class='model-banner-label' style='color:{label_color};'>{label}</div>"
        f"<div class='model-banner-name'>{name}</div>"
        f"</div>"
    )