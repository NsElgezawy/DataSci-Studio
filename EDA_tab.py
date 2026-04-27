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


def eda_tab(df, selected_col, col_type):
    t, pal, _ = get_theme()

    st.markdown('<div class="sec-header">Dataset Overview</div>', unsafe_allow_html=True)

    rows, cols_count = get_shape(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{rows:,}")
    c2.metric("Columns", f"{cols_count:,}")

    missing_df   = check_missing_values(df).toPandas()
    total_missing = missing_df['missing_count'].sum() if not missing_df.empty else 0
    c3.metric("Null Values", f"{int(total_missing):,}")

    dup_result     = check_duplicates(df).toPandas()
    duplicate_rows = dup_result['duplicate_rows'].iloc[0] if not dup_result.empty else 0
    c4.metric("Duplicates", f"{int(duplicate_rows):,}")

    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(df.limit(100).toPandas(), use_container_width=True)

    with st.expander("📝 Schema & Info", expanded=False):
        schema_data = [{"Column": cn, "Type": ct} for cn, ct in df.dtypes]
        import pandas as pd
        st.dataframe(pd.DataFrame(schema_data), use_container_width=True)

    st.markdown('<div class="sec-header">Column Analysis</div>', unsafe_allow_html=True)

    if not selected_col:
        st.info("Select a column in the sidebar.")
        return

    badge_cls = "badge-num" if col_type == "numeric" else "badge-cat"
    st.markdown(
        f"**Column:** `{selected_col}` &nbsp;"
        f"<span class='badge {badge_cls}'>{col_type}</span>",
        unsafe_allow_html=True,
    )

    if col_type == "numeric":
        stats = df.select(
            F.mean(selected_col).alias("mean"),
            F.stddev(selected_col).alias("std"),
            F.min(selected_col).alias("min"),
            F.max(selected_col).alias("max"),
        ).collect()[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{stats['mean']:.4f}" if stats['mean'] else "N/A")
        c2.metric("Std",  f"{stats['std']:.4f}"  if stats['std']  else "N/A")
        c3.metric("Min",  f"{stats['min']:.4f}"  if stats['min']  else "N/A")
        c4.metric("Max",  f"{stats['max']:.4f}"  if stats['max']  else "N/A")

        col_a, col_b = st.columns(2)
        with col_a:
            fig = histogram_plot(df, selected_col)
            st.pyplot(style_fig(fig)); plt.close()
        with col_b:
            fig = box_plot(df, selected_col)
            st.pyplot(style_fig(fig)); plt.close()

        with st.expander("🔍 Outlier Detection", expanded=False):
            outlier_method = st.selectbox("Method", ["IQR", "Z-Score", "Percentile"], key="out_meth")
            if outlier_method == "IQR":
                outliers, out_count, lower, upper = iqr_outlier_detection(df, selected_col)
                st.metric("Lower Bound", f"{lower:.4f}")
                st.metric("Upper Bound", f"{upper:.4f}")
                st.metric("Outlier Count", f"{out_count:,}")
                if out_count > 0 and st.button("Show Sample"):
                    st.dataframe(outliers.limit(10).toPandas(), use_container_width=True)

            elif outlier_method == "Z-Score":
                threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.5)
                outliers, out_count, mean_val, std_val = detect_outliers_zscore(df, selected_col, threshold)
                st.metric("Mean", f"{mean_val:.4f}")
                st.metric("Std", f"{std_val:.4f}")
                st.metric("Outlier Count", f"{out_count:,}")
                if out_count > 0 and st.button("Show Sample"):
                    st.dataframe(outliers.limit(10).toPandas(), use_container_width=True)

            else:
                lower_pct = st.slider("Lower Percentile", 0.01, 0.10, 0.01, 0.01)
                upper_pct = st.slider("Upper Percentile", 0.90, 0.99, 0.99, 0.01)
                outliers, out_count, lower, upper = detect_outliers_percentile(df, selected_col, lower_pct, upper_pct)
                st.metric(f"Lower ({lower_pct*100:.0f}%)", f"{lower:.4f}")
                st.metric(f"Upper ({upper_pct*100:.0f}%)", f"{upper:.4f}")
                st.metric("Outlier Count", f"{out_count:,}")
                if out_count > 0 and st.button("Show Sample"):
                    st.dataframe(outliers.limit(10).toPandas(), use_container_width=True)
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = countplot(df, selected_col)
            st.pyplot(style_fig(fig)); plt.close()
        with col_b:
            freq_df = (df.groupBy(selected_col).count()
                      .orderBy(F.col("count").desc()).limit(20).toPandas())
            st.dataframe(freq_df, use_container_width=True)