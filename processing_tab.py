import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import functions as F
from pyspark.sql.functions import col, mean, log1p

from EDA import *
from preprocessing import *
from ui import get_theme, style_fig, log


def processing_tab(df):
    t, pal, _ = get_theme()

    st.markdown('<div class="sec-header">🔄 Data Processing Pipeline</div>', unsafe_allow_html=True)

    proc_subtab = st.tabs([
        "📊 Overview", "🕳️ Missing Values", "🔄 Duplicates",
        "⚠️ Outliers", "📝 Inconsistent Data",
        "🏷️ Encoding", "📏 Scaling", "🎯 Feature Selection", "✂️ Train/Test Split",
    ])

    # ── Overview ──
    with proc_subtab[0]:
        st.markdown("#### 📊 Dataset Summary")
        rows, cols_count = get_shape(df)
        numeric_cols, categorical_cols = detect_columns(df)
        missing_df   = check_missing_values(df).toPandas()
        total_missing = missing_df['missing_count'].sum() if not missing_df.empty else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows",          f"{rows:,}")
        c2.metric("Columns",       f"{cols_count:,}")
        c3.metric("Numeric",       f"{len(numeric_cols)}")
        c4.metric("Categorical",   f"{len(categorical_cols)}")
        c5.metric("Missing Total", f"{int(total_missing):,}")

        st.markdown("#### Column Information")
        col_info = [{"Column": cn, "Type": ct} for cn, ct in df.dtypes]
        st.dataframe(pd.DataFrame(col_info), use_container_width=True)

    # ── Missing Values ──
    with proc_subtab[1]:
        st.markdown("#### 🕳️ Missing Values Analysis")
        missing_df   = check_missing_values(df).toPandas()
        missing_cols_df = missing_df[missing_df['missing_count'] > 0]

        if not missing_cols_df.empty:
            st.dataframe(missing_cols_df, use_container_width=True)

            if st.button("📊 Show Missing Values Chart", key="show_missing_chart"):
                fig, ax = plt.subplots(figsize=(8, 4))
                mc_sorted = missing_cols_df.sort_values('missing_count', ascending=False)
                ax.barh(mc_sorted['column_name'], mc_sorted['missing_count'], color=t["danger"])
                ax.set_title('Missing Values per Column', color=t["text"])
                ax.set_xlabel('Count', color=t["muted"])
                st.pyplot(style_fig(fig)); plt.close()

            st.markdown("#### 🛠️ Handle Missing Values")
            col1, col2 = st.columns(2)
            with col1:
                mv_col      = st.selectbox("Column to treat", missing_cols_df['column_name'].tolist(), key="mv_col_select")
                mv_strategy = st.selectbox("Strategy",
                    ["Drop rows", "Fill with Mean", "Fill with Median", "Fill with Mode", "Fill with Constant"],
                    key="mv_strategy_select")
            with col2:
                mv_const = ""
                if mv_strategy == "Fill with Constant":
                    mv_const = st.text_input("Constant value", "0", key="mv_const")

            if st.button("✅ Apply Treatment", key="apply_missing_btn"):
                try:
                    with st.spinner("Processing..."):
                        if mv_strategy == "Drop rows":
                            new_df = drop_rows_missing_cols(df, [mv_col])
                            log(f"Dropped rows with missing in '{mv_col}'")
                        elif mv_strategy == "Fill with Mean":
                            new_df = impute_mean(df, [mv_col])
                            log(f"Filled '{mv_col}' with mean")
                        elif mv_strategy == "Fill with Median":
                            new_df = impute_median(df, [mv_col])
                            log(f"Filled '{mv_col}' with median")
                        elif mv_strategy == "Fill with Mode":
                            new_df = impute_mode(df, [mv_col])
                            log(f"Filled '{mv_col}' with mode")
                        else:
                            new_df = df.fillna({mv_col: mv_const})
                            log(f"Filled '{mv_col}' with constant '{mv_const}'")

                        st.session_state.spark_df = new_df
                        st.session_state.train_df = None
                        st.session_state.test_df  = None
                        st.success(f"✅ Applied '{mv_strategy}' on '{mv_col}'")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    log(f"ERROR missing treatment: {e}")

            if st.checkbox("Show data preview", key="show_preview"):
                st.dataframe(st.session_state.spark_df.limit(20).toPandas(), use_container_width=True)
        else:
            st.success("✅ No missing values found!")

    # ── Duplicates ──
    with proc_subtab[2]:
        st.markdown("#### 🔄 Duplicates Analysis")
        duplicates, total_rows, unique_rows = count_duplicates_spark(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows",     f"{total_rows:,}")
        c2.metric("Unique Rows",    f"{unique_rows:,}")
        c3.metric("Duplicate Rows", f"{duplicates:,}",
                  delta=f"{(duplicates/total_rows)*100:.2f}%" if total_rows > 0 else None)

        if duplicates > 0:
            if st.button("🗑️ Remove Duplicates", key="remove_dupes"):
                new_df = remove_duplicates_spark(df)
                st.session_state.spark_df = new_df
                log(f"Removed {duplicates} duplicate rows")
                st.success(f"✅ Removed {duplicates} duplicates. New size: {new_df.count():,} rows")
                st.rerun()
            if st.button("🔍 Show Duplicate Groups", key="show_dupes"):
                dup_groups = df.groupBy(df.columns).count().filter(col("count") > 1)
                st.write(f"Duplicate groups: {dup_groups.count()}")
                st.dataframe(dup_groups.limit(50).toPandas(), use_container_width=True)
        else:
            st.info("ℹ️ No duplicate rows found.")

    # ── Outliers ──
    with proc_subtab[3]:
        st.markdown("#### ⚠️ Outlier Detection & Treatment")
        numeric_cols, _ = detect_columns(df)

        if numeric_cols:
            outlier_col      = st.selectbox("Column", numeric_cols, key="outlier_col_select")
            detection_method = st.selectbox("Detection Method", ["IQR", "Z-Score", "Percentile"], key="detection_method")

            threshold = 3
            lower_pct, upper_pct = 0.01, 0.99
            if detection_method == "Z-Score":
                threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.5, key="zscore_threshold")
            elif detection_method == "Percentile":
                col1, col2 = st.columns(2)
                with col1:
                    lower_pct = st.slider("Lower Percentile", 0.01, 0.10, 0.01, 0.01, key="lower_pct")
                with col2:
                    upper_pct = st.slider("Upper Percentile", 0.90, 0.99, 0.99, 0.01, key="upper_pct")

            if st.button("🔍 Detect Outliers", key="detect_btn"):
                try:
                    before_count = df.count()
                    if detection_method == "IQR":
                        outliers, out_count, lower, upper = iqr_outlier_detection(df, outlier_col)
                        st.markdown(f"""| Metric | Value |
|--------|-------|
| **Lower Bound (Q1 - 1.5×IQR)** | `{lower:.4f}` |
| **Upper Bound (Q3 + 1.5×IQR)** | `{upper:.4f}` |
| **Outliers Detected** | `{out_count:,}` |
| **Percentage** | `{(out_count/before_count)*100:.2f}%` |""")
                    elif detection_method == "Z-Score":
                        outliers, out_count, mean_val, std_val = detect_outliers_zscore(df, outlier_col, threshold)
                        st.markdown(f"""| Metric | Value |
|--------|-------|
| **Mean** | `{mean_val:.4f}` |
| **Std Dev** | `{std_val:.4f}` |
| **Threshold** | `{threshold}` |
| **Outliers Detected** | `{out_count:,}` |
| **Percentage** | `{(out_count/before_count)*100:.2f}%` |""")
                    else:
                        outliers, out_count, lower, upper = detect_outliers_percentile(df, outlier_col, lower_pct, upper_pct)
                        st.markdown(f"""| Metric | Value |
|--------|-------|
| **Lower ({lower_pct*100:.0f}%)** | `{lower:.4f}` |
| **Upper ({upper_pct*100:.0f}%)** | `{upper:.4f}` |
| **Outliers Detected** | `{out_count:,}` |
| **Percentage** | `{(out_count/before_count)*100:.2f}%` |""")

                    if out_count > 0:
                        st.warning(f"⚠️ Found {out_count:,} outliers")
                        st.dataframe(outliers.limit(50).toPandas(), use_container_width=True)
                    else:
                        st.success("✅ No outliers detected!")
                except Exception as e:
                    st.error(f"Error: {e}")

            st.markdown("#### 🛠️ Outlier Treatment")
            treatment_method = st.selectbox("Treatment",
                ["Remove", "Replace with Mean", "Replace with Median", "Log Transform"],
                key="treatment_method")

            if st.button("Apply Treatment", key="apply_treatment"):
                try:
                    if detection_method == "IQR":
                        _, _, lower, upper = iqr_outlier_detection(df, outlier_col)
                    elif detection_method == "Z-Score":
                        _, _, mean_val, std_val = detect_outliers_zscore(df, outlier_col, threshold)
                        lower = mean_val - threshold * std_val
                        upper = mean_val + threshold * std_val
                    else:
                        _, _, lower, upper = detect_outliers_percentile(df, outlier_col, lower_pct, upper_pct)

                    condition = (col(outlier_col) < lower) | (col(outlier_col) > upper)

                    if treatment_method == "Remove":
                        new_df = df.filter(~condition)
                        log(f"Removed outliers from '{outlier_col}'")
                        st.success(f"✅ Removed outliers. New size: {new_df.count():,}")
                    elif treatment_method == "Replace with Mean":
                        mean_val = df.select(mean(outlier_col)).collect()[0][0]
                        new_df = df.withColumn(outlier_col, F.when(condition, mean_val).otherwise(col(outlier_col)))
                        log(f"Replaced outliers with mean in '{outlier_col}'")
                        st.success(f"✅ Replaced with mean: {mean_val:.4f}")
                    elif treatment_method == "Replace with Median":
                        median_val = df.approxQuantile(outlier_col, [0.5], 0.01)[0]
                        new_df = df.withColumn(outlier_col, F.when(condition, median_val).otherwise(col(outlier_col)))
                        log(f"Replaced outliers with median in '{outlier_col}'")
                        st.success(f"✅ Replaced with median: {median_val:.4f}")
                    else:
                        new_df = df.withColumn(outlier_col, log1p(col(outlier_col)))
                        log(f"Applied log transform to '{outlier_col}'")
                        st.success("✅ Log transformation applied.")

                    st.session_state.spark_df = new_df
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("ℹ️ No numeric columns available.")

    # ── Inconsistent Data ──
    with proc_subtab[4]:
        st.markdown("#### 📝 Inconsistent Data Handling")

        st.markdown("##### 🔍 Unique Values")
        col_to_check = st.selectbox("Column", df.columns, key="unique_col")
        if st.button("Show Unique Values", key="show_unique"):
            unique_vals = check_unique_values(df, col_to_check)
            st.write(f"Unique count: {unique_vals.count()}")
            st.dataframe(unique_vals.limit(100).toPandas(), use_container_width=True)

        st.markdown("##### 🧹 Clean Text Columns")
        text_cols = [c for c, tp in df.dtypes if tp == "string"]
        if text_cols:
            cols_to_clean = st.multiselect("Select columns", text_cols, key="clean_cols")
            if cols_to_clean and st.button("Clean Text Columns", key="clean_text"):
                new_df = clean_text_columns(df, cols_to_clean)
                st.session_state.spark_df = new_df
                log(f"Cleaned text columns: {cols_to_clean}")
                st.success(f"✅ Cleaned {len(cols_to_clean)} text columns")
                st.rerun()
        else:
            st.info("No text columns found.")

        st.markdown("##### 📅 Standardize Date Columns")
        date_cols = [c for c, tp in df.dtypes if tp == "string" and "date" in c.lower()]
        if date_cols:
            date_col    = st.selectbox("Date column", date_cols, key="date_col")
            date_format = st.text_input("Date format", "yyyy-MM-dd", key="date_format")
            if st.button("Standardize Date", key="standardize_date"):
                new_df = standardize_date(df, date_col, date_format)
                st.session_state.spark_df = new_df
                log(f"Standardized date column: {date_col}")
                st.success(f"✅ Standardized '{date_col}'")
                st.rerun()

        st.markdown("##### 🔢 Clean Numeric-Text Columns")
        if text_cols:
            num_text_col = st.selectbox("Column with numeric text", text_cols, key="num_text_col")
            if st.button("Clean Numeric Text", key="clean_numeric"):
                new_df = clean_numeric_text(df, num_text_col)
                st.session_state.spark_df = new_df
                log(f"Cleaned numeric text in: {num_text_col}")
                st.success(f"✅ Cleaned numeric text in '{num_text_col}'")
                st.rerun()

    # ── Encoding ──
    with proc_subtab[5]:
        st.markdown("#### 🏷️ Categorical Encoding")
        st.info("⚠️ Split your data first in 'Train/Test Split' before encoding.")

        train_exists = st.session_state.train_df is not None
        test_exists  = st.session_state.test_df  is not None

        if train_exists and test_exists:
            tr = st.session_state.train_df
            te = st.session_state.test_df
            _, categorical_cols = detect_columns(tr)

            if categorical_cols:
                enc_col    = st.selectbox("Column to encode", categorical_cols, key="enc_col")
                enc_method = st.selectbox("Method", ["Label Encoding", "One-Hot Encoding", "Frequency Encoding"], key="enc_method")
                if st.button("Apply Encoding", key="apply_encoding"):
                    try:
                        if enc_method == "Label Encoding":
                            tr_enc, te_enc = apply_label_encoding_spark(tr, te, enc_col)
                        elif enc_method == "One-Hot Encoding":
                            tr_enc, te_enc = apply_onehot_encoding_spark(tr, te, enc_col)
                        else:
                            tr_enc, te_enc = frequency_encoding(tr, te, [enc_col])
                        st.session_state.train_df = tr_enc
                        st.session_state.test_df  = te_enc
                        log(f"Applied {enc_method} on '{enc_col}'")
                        st.success(f"✅ Applied {enc_method} on '{enc_col}'")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.info("No categorical columns found.")
        else:
            st.warning("⚠️ Split your data first.")

    # ── Scaling ──
    with proc_subtab[6]:
        st.markdown("#### 📏 Feature Scaling")
        train_exists = st.session_state.train_df is not None

        if train_exists:
            tr = st.session_state.train_df
            te = st.session_state.test_df
            numeric_cols, _ = detect_columns(tr)

            if numeric_cols:
                scaler_method = st.selectbox("Scaling Method",
                    ["Standard Scaler (Z-score)", "MinMax Scaler"], key="scaler_method")
                if st.button("Apply Scaling", key="apply_scaling"):
                    try:
                        if scaler_method == "Standard Scaler (Z-score)":
                            tr_s, te_s = apply_standard_scaler_spark(tr, te)
                        else:
                            tr_s, te_s = apply_minmax_scaler_spark(tr, te)
                        st.session_state.train_df = tr_s
                        st.session_state.test_df  = te_s
                        log(f"Applied {scaler_method}")
                        st.success(f"✅ Applied {scaler_method}")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.info("No numeric columns found.")
        else:
            st.warning("⚠️ Split your data first.")

    # ── Feature Selection ──
    with proc_subtab[7]:
        st.markdown("#### 🎯 Feature Selection")
        train_exists = st.session_state.train_df is not None

        if train_exists:
            tr = st.session_state.train_df
            numeric_cols, _ = detect_columns(tr)

            if numeric_cols:
                fs_method = st.selectbox("Method", ["Variance Threshold", "Chi-Square"], key="fs_method")
                if fs_method == "Variance Threshold":
                    var_thresh = st.slider("Variance Threshold", 0.0, 0.1, 0.01, 0.001, key="var_threshold")
                    if st.button("Select by Variance", key="select_variance"):
                        selected = select_features_variance_spark(tr, var_thresh)
                        st.success(f"✅ Selected {len(selected)} features (variance > {var_thresh})")
                        st.write("**Selected Features:**", selected)
                        log(f"Variance selection: {len(selected)} features")
                else:
                    label_col = st.selectbox("Target column", tr.columns, key="label_col_chisq")
                    k_feat    = st.slider("Top k features", 1, min(20, len(numeric_cols)), 5, key="k_features")
                    if st.button("Select by Chi-Square", key="select_chisq"):
                        selected = select_features_chisq_spark(tr, label_col, k_feat)
                        st.success(f"✅ Selected {len(selected)} top features")
                        st.write("**Selected Features:**", selected)
                        log(f"Chi-Square selection: {len(selected)} features")
            else:
                st.info("No numeric columns found.")
        else:
            st.warning("⚠️ Split your data first.")

    # ── Train/Test Split ──
    with proc_subtab[8]:
        st.markdown("#### ✂️ Train/Test Split")
        test_size   = st.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05, key="test_size_split")
        random_seed = st.number_input("Random Seed", value=42, key="random_seed")

        if st.button("Split Data", key="split_data"):
            train, test = split_data_spark(df, test_size, random_seed)
            st.session_state.train_df = train
            st.session_state.test_df  = test
            train_n = train.count()
            test_n  = test.count()
            st.success(f"✅ Train: {train_n:,} rows | Test: {test_n:,} rows")
            log(f"Split data — Train: {train_n}, Test: {test_n}")

            st.markdown("#### Train Set Preview")
            st.dataframe(train.limit(10).toPandas(), use_container_width=True)
            st.markdown("#### Test Set Preview")
            st.dataframe(test.limit(10).toPandas(), use_container_width=True)

        if st.session_state.train_df is not None:
            st.info(
                f"Current split — Train: {st.session_state.train_df.count():,} rows | "
                f"Test: {st.session_state.test_df.count():,} rows"
            )