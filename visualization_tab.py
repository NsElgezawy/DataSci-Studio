import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import functions as F

from EDA import *
from preprocessing import *
from ui import get_theme, style_fig


def visualization_tab(df):
    t, pal, _ = get_theme()

    st.markdown('<div class="sec-header">📊 Visualization Dashboard</div>', unsafe_allow_html=True)

    numeric_cols = [c for c, dtype in df.dtypes if any(nt in dtype.lower()
                   for nt in ['int', 'double', 'float', 'decimal', 'long', 'short'])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    all_cols = df.columns

    viz_subtab = st.tabs([
        "📈 Distribution", "📊 Categorical",
        "🔗 Relationships", "🎻 Advanced", "📉 Missing & Correlation",
    ])

    st.sidebar.markdown("### 🎛️ Visualization Settings")
    sample_size = st.sidebar.slider("Sample Size", 0.01, 1.0, 0.1, 0.01, key="viz_sample")
    st.sidebar.markdown("---")

    # ── Distribution ──
    with viz_subtab[0]:
        st.markdown("#### 📈 Distribution Plots")
        col1, col2 = st.columns(2)

        with col1:
            num_col = st.selectbox("Select Column", numeric_cols if numeric_cols else all_cols, key="dist_col")
            plot_type = st.radio("Plot Type", ["Histogram", "KDE Plot", "Box Plot"], horizontal=True, key="dist_type")

            if st.button("Generate Distribution Plot", key="btn_dist"):
                with st.spinner("Generating plot..."):
                    if plot_type == "Histogram":
                        fig = histogram_plot(df, num_col, sample_size)
                    elif plot_type == "KDE Plot":
                        fig = kde_plot(df, num_col, sample_size)
                    else:
                        fig = box_plot(df, num_col, sample_size)
                    st.pyplot(style_fig(fig)); plt.close()

        with col2:
            st.markdown("##### 📋 Statistical Summary")
            if numeric_cols:
                stats_df = df.select(
                    F.mean(num_col).alias("Mean"),
                    F.stddev(num_col).alias("Std Dev"),
                    F.min(num_col).alias("Min"),
                    F.expr(f"percentile_approx({num_col}, 0.25)").alias("Q1"),
                    F.expr(f"percentile_approx({num_col}, 0.5)").alias("Median"),
                    F.expr(f"percentile_approx({num_col}, 0.75)").alias("Q3"),
                    F.max(num_col).alias("Max"),
                ).toPandas()
                for cn in stats_df.columns:
                    val = stats_df[cn].iloc[0]
                    if val is not None:
                        st.metric(cn, f"{val:.4f}")

    # ── Categorical ──
    with viz_subtab[1]:
        st.markdown("#### 📊 Categorical Analysis")
        if categorical_cols:
            col1, col2 = st.columns(2)
            with col1:
                cat_col = st.selectbox("Select Categorical Column", categorical_cols, key="cat_col")
                top_n = st.slider("Top Categories", 5, 50, 20, 5, key="top_n")
                if st.button("Generate Count Plot", key="btn_cat"):
                    with st.spinner("Generating..."):
                        fig = countplot(df, cat_col, top_n)
                        st.pyplot(style_fig(fig)); plt.close()
            with col2:
                st.markdown("##### 📋 Frequency Table")
                freq_df = (df.groupBy(cat_col).count()
                           .orderBy(F.col("count").desc()).limit(top_n).toPandas())
                st.dataframe(freq_df, use_container_width=True)
        else:
            st.info("No categorical columns found in the dataset.")

    # ── Relationships ──
    with viz_subtab[2]:
        st.markdown("#### 🔗 Relationship Analysis")
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X Axis", numeric_cols, key="scatter_x")
                y_col = st.selectbox("Y Axis", [c for c in numeric_cols if c != x_col], key="scatter_y")
                if st.button("Generate Scatter Plot", key="btn_scatter"):
                    with st.spinner("Generating..."):
                        fig = plot_scatter(df, x_col, y_col, sample_size)
                        st.pyplot(style_fig(fig)); plt.close()
            with col2:
                st.markdown("##### 📊 Correlation")
                corr_val = df.stat.corr(x_col, y_col)
                st.metric(f"Pearson r ({x_col} vs {y_col})", f"{corr_val:.4f}")
                st.caption("±1 → strong | 0 → none")

            st.markdown("---")
            st.markdown("#### Pair Plot Matrix")
            if len(numeric_cols) <= 6:
                if st.button("Generate Pair Plot", key="btn_pair"):
                    with st.spinner("Generating (may take a moment)..."):
                        fig = plot_pair(df, min(sample_size, 0.2))
                        st.pyplot(style_fig(fig)); plt.close()
            else:
                st.info(f"Pair plot skipped: {len(numeric_cols)} numeric columns (max 6).")
        else:
            st.warning("Need at least 2 numeric columns.")

    # ── Advanced ──
    with viz_subtab[3]:
        st.markdown("#### 🎻 Advanced Visualizations")
        adv_type = st.selectbox("Type", ["Violin Plot", "Stacked Bar Chart"], key="adv_type")

        if adv_type == "Violin Plot":
            if categorical_cols and numeric_cols:
                col1, col2 = st.columns(2)
                with col1:
                    x_violin = st.selectbox("Category (X)", categorical_cols, key="violin_x")
                    y_violin = st.selectbox("Value (Y)", numeric_cols, key="violin_y")
                    if st.button("Generate Violin Plot", key="btn_violin"):
                        with st.spinner("Generating..."):
                            fig = plot_violin(df, x_violin, y_violin, sample_size)
                            st.pyplot(style_fig(fig)); plt.close()
                with col2:
                    st.markdown("##### Group Statistics")
                    stats_by_cat = (df.groupBy(x_violin)
                                   .agg(F.mean(y_violin).alias("Mean"),
                                        F.stddev(y_violin).alias("Std Dev"),
                                        F.count(y_violin).alias("Count"))
                                   .orderBy(F.col("Count").desc()).limit(10).toPandas())
                    st.dataframe(stats_by_cat, use_container_width=True)
            else:
                st.warning("Need both categorical and numeric columns.")
        else:
            if categorical_cols and len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    index_col = st.selectbox("Index Column", categorical_cols, key="stacked_index")
                    val_cols  = st.multiselect("Value Columns", numeric_cols,
                                               default=numeric_cols[:min(3, len(numeric_cols))], key="stacked_vals")
                    stacked   = st.checkbox("Stacked", value=True, key="is_stacked")
                    if val_cols and st.button("Generate Stacked Bar Chart", key="btn_stacked"):
                        with st.spinner("Generating..."):
                            fig = plot_stacked_bar(df, index_col, val_cols, stacked, sample_size)
                            st.pyplot(style_fig(fig)); plt.close()
                with col2:
                    if val_cols:
                        preview_df = (df.groupBy(index_col)
                                     .agg(*[F.mean(c).alias(c) for c in val_cols])
                                     .orderBy(F.col(val_cols[0]).desc()).limit(10).toPandas())
                        st.dataframe(preview_df, use_container_width=True)
            else:
                st.warning("Need at least one categorical and 2+ numeric columns.")

    # ── Missing & Correlation ──
    with viz_subtab[4]:
        st.markdown("#### 📉 Data Quality Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 🔍 Missing Values")
            missing_df  = check_missing_values(df).toPandas()
            missing_cols_df = missing_df[missing_df['missing_count'] > 0]
            if not missing_cols_df.empty:
                st.dataframe(missing_cols_df, use_container_width=True)
                if st.button("Show Missing Values Chart", key="btn_missing"):
                    with st.spinner("Generating..."):
                        fig, ax = plt.subplots(figsize=(8, 4))
                        mc_sorted = missing_cols_df.sort_values('missing_count', ascending=False)
                        ax.barh(mc_sorted['column_name'], mc_sorted['missing_count'], color=t["danger"])
                        ax.set_title('Missing Values per Column', color=t["text"])
                        ax.set_xlabel('Count', color=t["muted"])
                        st.pyplot(style_fig(fig)); plt.close()
            else:
                st.success("✅ No missing values found!")

        with col2:
            st.markdown("##### 🔥 Correlation Heatmap")
            if len(numeric_cols) >= 2:
                if st.button("Generate Correlation Heatmap", key="btn_heatmap"):
                    with st.spinner("Generating..."):
                        fig = plot_heatmap(df, min(sample_size, 0.2))
                        st.pyplot(style_fig(fig)); plt.close()
            else:
                st.info("Need at least 2 numeric columns.")