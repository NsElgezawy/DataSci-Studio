import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
import io
import os
import pickle
import traceback
from datetime import datetime
from EDA import *
from preprocessing import *

# PySpark imports
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, sum, when, mean, trim, lower, to_date, regexp_replace, log1p, variance
from pyspark.sql.types import NumericType
import missingno as msno

# PySpark ML imports
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, MinMaxScaler, ChiSqSelector
from pyspark.sql.functions import min as spark_min

warnings.filterwarnings('ignore')

from ui import get_theme, inject_css, log
from data_loader import get_spark_session, load_csv_spark, load_kaggle_spark, spark
from visualization_tab import visualization_tab
from EDA_tab import eda_tab
from processing_tab import processing_tab
from modeling_tab import modeling_tab

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="DataSci Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# SESSION STATE INIT
# ==============================================================================
defaults = {
    "spark_df": None,
    "spark_df_original": None,
    "logs": [],
    "predictions": None,
    "model": None,
    "train_df": None,
    "test_df": None,
    "theme_mode": "dark",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Resolve theme early so CSS can be injected
theme, palette, mode = get_theme()
inject_css(theme, mode)


# ==============================================================================
# HELPERS
# ==============================================================================
def get_col_type(df, col_name: str) -> str:
    col_type = dict(df.dtypes).get(col_name, "string")
    numeric_types = ['int', 'bigint', 'double', 'float', 'decimal', 'long', 'short', 'tinyint']
    if any(nt in col_type.lower() for nt in numeric_types):
        return "numeric"
    return "categorical"


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    t, pal, mode = get_theme()

    st.markdown(
        '<div class="logo-area">Data<span>Sci</span> Studio</div>'
        '<div class="logo-tagline">End-to-end ML workspace · PySpark</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Theme toggle
    st.markdown('<div class="theme-toggle-label">🎨 Theme</div>', unsafe_allow_html=True)
    theme_choice = st.radio(
        "theme_radio",
        ["🌙 Dark", "☀️ Light"],
        index=0 if mode == "dark" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="theme_radio",
    )
    new_mode = "dark" if "Dark" in theme_choice else "light"
    if new_mode != st.session_state.theme_mode:
        st.session_state.theme_mode = new_mode
        st.rerun()

    st.divider()

    # Data source
    st.markdown("### 📂 Data Source")
    source = st.radio("Input method", ["Upload CSV", "Kaggle Path"],
                      horizontal=True, label_visibility='collapsed')

    if source == "Upload CSV":
        uploaded    = st.file_uploader("Choose CSV file", type=['csv'], label_visibility='collapsed')
        sample_rows = st.number_input("Sample rows (0 = all)", min_value=0, value=100000, step=10000)

        if uploaded and st.button("Load Dataset"):
            with st.spinner("Loading with PySpark..."):
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = tmp.name
                    nrows = sample_rows if sample_rows > 0 else None
                    df = load_csv_spark(tmp_path, nrows=nrows)
                    df = df.cache(); df.count()
                    st.session_state.spark_df          = df
                    st.session_state.spark_df_original = df
                    st.session_state.train_df          = None
                    st.session_state.test_df           = None
                    r, c = get_shape(df)
                    log(f"Loaded CSV: {r:,} rows × {c} cols")
                    st.success(f"✅ Loaded {r:,} rows")
                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Error: {e}"); log(f"ERROR: {e}")
    else:
        kg_path = st.text_input("Dataset ID", placeholder="username/dataset-name")
        kg_file = st.text_input("Filename", placeholder="data.csv")
        kg_rows = st.number_input("Max rows (0 = all)", min_value=0, value=100000, step=10000)

        if st.button("Load from Kaggle"):
            with st.spinner("Downloading from Kaggle..."):
                try:
                    nrows = kg_rows if kg_rows > 0 else None
                    df = load_kaggle_spark(kg_path, kg_file, nrows=nrows)
                    if df:
                        st.session_state.spark_df          = df
                        st.session_state.spark_df_original = df
                        st.session_state.train_df          = None
                        st.session_state.test_df           = None
                        r, c = get_shape(df)
                        log(f"Loaded Kaggle: {r:,} rows × {c} cols")
                        st.success(f"✅ Loaded {r:,} rows")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Column selector
    df_current = st.session_state.spark_df
    if df_current is not None:
        st.divider()
        st.markdown("### 🔍 Column Inspector")
        all_cols     = df_current.columns
        selected_col = st.selectbox("Column", [""] + all_cols, label_visibility='collapsed')
        col_type     = get_col_type(df_current, selected_col) if selected_col else "—"
        if selected_col:
            badge_cls = 'badge-num' if col_type == 'numeric' else 'badge-cat'
            st.markdown(f'<span class="badge {badge_cls}">{col_type}</span>', unsafe_allow_html=True)

        st.divider()
        if st.button("🔄 Reset to Original"):
            if st.session_state.spark_df_original is not None:
                st.session_state.spark_df = st.session_state.spark_df_original
                st.session_state.train_df = None
                st.session_state.test_df  = None
                log("Dataset reset to original.")
                st.success("Reset!")
                st.rerun()
    else:
        selected_col = ""
        col_type     = "—"

    # Operation log
    st.divider()
    st.markdown("### 📋 Operation Log")
    log_html = "<br>".join(st.session_state.logs[-20:][::-1]) if st.session_state.logs else "No operations yet."
    st.markdown(f'<div class="log-panel">{log_html}</div>', unsafe_allow_html=True)


# ==============================================================================
# MAIN AREA
# ==============================================================================
t, pal, mode = get_theme()

if st.session_state.spark_df is None:
    hero_icon  = "🔬"
    hero_color = t["primary"]
    st.markdown(f"""
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
     height:62vh;text-align:center;gap:18px;">
    <div style="font-size:4rem;filter:drop-shadow(0 0 24px {hero_color}66);">{hero_icon}</div>
    <div style="font-family:'Space Mono',monospace;font-size:1.9rem;font-weight:700;
                color:{hero_color};letter-spacing:-0.03em;">DataSci Studio</div>
    <div style="color:{t["muted"]};max-width:440px;line-height:1.75;font-size:0.9rem;">
        Upload a CSV or connect a Kaggle dataset from the sidebar to begin your
        end-to-end data science workflow powered by PySpark.
    </div>
    <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;justify-content:center;">
        <span class="badge">PySpark Backend</span>
        <span class="badge">Scalable EDA</span>
        <span class="badge">Missing Detection</span>
        <span class="badge">Outlier Detection</span>
        <span class="badge">Encoding</span>
        <span class="badge">Feature Scaling</span>
        <span class="badge">Classification</span>
        <span class="badge">Regression</span>
        <span class="badge">Prediction</span>
    </div>
</div>
""", unsafe_allow_html=True)
else:
    df = st.session_state.spark_df
    tab_viz, tab_eda, tab_proc, tab_model = st.tabs([
        "📊  Visualization", "📈  EDA", "🧹  Processing", "🤖  Modeling",
    ])
    with tab_viz:   visualization_tab(df)
    with tab_eda:   eda_tab(df, selected_col, col_type)
    with tab_proc:  processing_tab(df)
    with tab_model: modeling_tab(df)