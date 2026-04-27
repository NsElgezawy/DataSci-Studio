import streamlit as st

from ui import get_theme, themed_bar_chart, result_card_html, model_banner_html, log


def modeling_tab(df):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg")
    import numpy as np
    from pyspark.ml.feature import VectorAssembler, StringIndexer
    from pyspark.sql import functions as F

    from models import (
        detect_task_type,
        get_available_classification_models,
        train_classification_model,
        evaluate_classification_model,
        select_best_model,
        XGBOOST_AVAILABLE,
        get_available_regression_models,
        train_regression_model,
        evaluate_regression_model,
        select_best_regression_model,
    )

    t, pal, _ = get_theme()

    st.markdown('<div class="sec-header">🤖 Machine Learning Workspace</div>', unsafe_allow_html=True)

    if st.session_state.train_df is None or st.session_state.test_df is None:
        st.warning("⚠️ Please split your data first in **Processing → Train/Test Split**.")
        return

    train_df = st.session_state.train_df
    test_df  = st.session_state.test_df

    cls_tab, reg_tab = st.tabs(["🏷️  Classification", "📈  Regression"])

    # ── Shared helpers ──────────────────────────────────────────────────────
    def prepare_data(tr, te, feat_cols, tgt, do_index):
        assembler = VectorAssembler(inputCols=feat_cols, outputCol="features", handleInvalid="skip")
        tr = assembler.transform(tr)
        te = assembler.transform(te)
        if do_index:
            indexer   = StringIndexer(inputCol=tgt, outputCol=tgt + "_idx", handleInvalid="keep")
            idx_model = indexer.fit(tr)
            tr = idx_model.transform(tr)
            te = idx_model.transform(te)
            return tr, te, tgt + "_idx"
        return tr, te, tgt

    def get_candidate_features(df_ref, exclude_col):
        nt = ("int", "bigint", "double", "float", "decimal", "long", "short", "tinyint")
        numeric  = [c for c, tp in df_ref.dtypes if any(x in tp.lower() for x in nt) and c != exclude_col]
        encoded  = [c for c in df_ref.columns if (c.endswith("_indexed") or c.endswith("_ohe")) and c != exclude_col]
        return list(dict.fromkeys(numeric + encoded))

    def highlight_max(s):
        is_max = s == s.max()
        return [f"background-color:{t['accent']}22;color:{t['accent']};font-weight:700" if v else "" for v in is_max]

    def highlight_reg(df_s):
        styles = pd.DataFrame("", index=df_s.index, columns=df_s.columns)
        for cn in df_s.columns:
            idx = df_s[cn].idxmax() if cn == "R2" else df_s[cn].idxmin()
            styles.loc[idx, cn] = f"background-color:{t['accent']}22;color:{t['accent']};font-weight:700"
        return styles

    # ── Prediction input widget ──────────────────────────────────────────────
    def prediction_inputs(feat_cols, key_prefix):
        input_vals = {}
        ncols      = 3
        groups     = [feat_cols[i:i+ncols] for i in range(0, len(feat_cols), ncols)]
        for group in groups:
            cols_ui = st.columns(len(group))
            for ci, fc in enumerate(group):
                dtype = dict(train_df.dtypes).get(fc, "double")
                if any(x in dtype for x in ("int", "long", "bigint", "short", "tinyint")):
                    input_vals[fc] = cols_ui[ci].number_input(fc, value=0, step=1, key=f"{key_prefix}_{fc}")
                else:
                    input_vals[fc] = cols_ui[ci].number_input(fc, value=0.0, step=0.01, format="%.4f", key=f"{key_prefix}_{fc}")
        return input_vals

    # ==========================================================================
    # CLASSIFICATION
    # ==========================================================================
    with cls_tab:
        st.markdown("### 🎯 Step 1 — Target Column")
        cls_target = st.selectbox("Target column", train_df.columns, key="cls_target")

        with st.spinner("Detecting task type..."):
            task_type = detect_task_type(train_df, cls_target)
            n_classes = train_df.select(cls_target).distinct().count()

        task_color = t["accent"] if task_type == "binary" else t["secondary"]
        st.markdown(
            f"**Task:** <span style='color:{task_color};font-family:Space Mono,monospace;font-weight:700;'>"
            f"{task_type.upper()} ({n_classes} classes)</span>",
            unsafe_allow_html=True,
        )
        if task_type == "multiclass":
            classes = [r[0] for r in train_df.select(cls_target).distinct().collect()]
            st.caption(f"Classes: {classes}")

        st.markdown("---")
        st.markdown("### ⚙️ Step 2 — Feature Columns")
        cls_candidates = get_candidate_features(train_df, cls_target)
        if not cls_candidates:
            st.error("❌ No usable feature columns found. Preprocess your data first.")
            st.stop()

        cls_features = st.multiselect("Features", cls_candidates, default=cls_candidates, key="cls_features")
        if not cls_features:
            st.warning("Select at least one feature.")
            st.stop()

        cls_needs_index = dict(train_df.dtypes).get(cls_target, "") == "string"
        if cls_needs_index:
            st.info(f"ℹ️ Target `{cls_target}` is string — will be auto-indexed.")

        st.markdown("---")
        st.markdown("### 🧪 Step 3 — Choose Models")
        available_cls = get_available_classification_models()
        if not XGBOOST_AVAILABLE:
            st.info("ℹ️ XGBoost not installed (`pip install xgboost[spark]`).")
        if task_type == "multiclass":
            st.caption("⚠️ GBT supports binary only natively. SVM uses OneVsRest automatically.")

        selected_cls = st.multiselect("Models to train", available_cls, default=available_cls[:2], key="selected_cls_models")
        st.markdown("---")

        if st.button("🚀 Train & Evaluate", key="cls_train_btn"):
            if not selected_cls:
                st.warning("Select at least one model."); st.stop()

            with st.spinner("Assembling feature vector..."):
                try:
                    tr_r, te_r, cls_label = prepare_data(train_df, test_df, cls_features, cls_target, cls_needs_index)
                except Exception as e:
                    st.error(f"❌ Assembly failed: {e}"); st.stop()

            st.success(f"✅ Feature vector: {len(cls_features)} columns | Label: `{cls_label}`")

            cls_results = {}; cls_models_trained = {}; cls_errors = {}
            prog = st.progress(0); status = st.empty()

            for i, mname in enumerate(selected_cls):
                status.markdown(f"⏳ Training **{mname}**...")
                try:
                    m = train_classification_model(mname, tr_r, cls_label, task_type)
                    cls_results[mname] = evaluate_classification_model(m, te_r, cls_label, task_type)
                    cls_models_trained[mname] = m
                except Exception as e:
                    cls_errors[mname] = str(e)
                prog.progress(int((i + 1) / len(selected_cls) * 100))

            status.empty(); prog.empty()
            for name, err in cls_errors.items():
                st.error(f"❌ **{name}** failed: {err}")
            if not cls_results:
                st.error("All models failed."); st.stop()

            st.session_state.cls_models      = cls_models_trained
            st.session_state.cls_results     = cls_results
            st.session_state.cls_task_type   = task_type
            st.session_state.cls_target      = cls_target
            st.session_state.cls_features    = cls_features
            st.session_state.cls_needs_index = cls_needs_index
            st.session_state.cls_label_col   = cls_label

            st.markdown("## 📊 Evaluation Results")
            rdf = pd.DataFrame(cls_results).T.round(4)
            st.dataframe(rdf.style.apply(highlight_max, axis=0), use_container_width=True)

            best_name, best_metrics = select_best_model(cls_results, "F1")
            st.markdown(
                model_banner_html(
                    "🏆 BEST MODEL (by F1)", best_name,
                    f"{t['primary']}14", f"{t['primary']}55", t["primary"],
                ),
                unsafe_allow_html=True,
            )

            st.markdown("### 🏆 Best Model Metrics")
            mc = st.columns(len(best_metrics))
            for ci, (k, v) in enumerate(best_metrics.items()):
                mc[ci].metric(k, f"{v:.4f}")

            st.markdown("### 📈 Model Comparison")
            mplot = [m for m in ["Accuracy", "F1", "Precision", "Recall", "AUC-ROC", "AUC-PR"] if m in rdf.columns]
            fig = themed_bar_chart(rdf, mplot, "Classification Comparison")
            st.pyplot(fig); plt.close()

            st.markdown("### 🔍 Per-Model Detail")
            for mname, metrics in cls_results.items():
                is_best = mname == best_name
                with st.expander(f"{'🥇' if is_best else '📋'} {mname}{'  ← Best' if is_best else ''}", expanded=is_best):
                    dc = st.columns(len(metrics))
                    for ci, (k, v) in enumerate(metrics.items()):
                        dc[ci].metric(k, f"{v:.4f}")

            log(f"[CLS] {list(cls_results.keys())} | Best: {best_name} | F1={best_metrics['F1']:.4f}")

        # ── Classification Prediction ────────────────────────────────────────
        if st.session_state.get("cls_models") and st.session_state.get("cls_features"):
            st.markdown("---")
            st.markdown("## 🔮 Predict — Classification")
            st.caption("Enter feature values and click Predict.")

            pred_model_name = st.selectbox("Model", list(st.session_state.cls_models.keys()), key="pred_cls_model_name")
            input_vals = prediction_inputs(st.session_state.cls_features, "cls_inp")

            if st.button("🎯 Predict", key="cls_predict_btn"):
                try:
                    feat_cols   = st.session_state.cls_features
                    spark_local = train_df.sparkSession
                    input_df    = spark_local.createDataFrame([tuple(input_vals[fc] for fc in feat_cols)], feat_cols)
                    assembler   = VectorAssembler(inputCols=feat_cols, outputCol="features", handleInvalid="keep")
                    assembled   = assembler.transform(input_df)

                    pred_result = st.session_state.cls_models[pred_model_name].transform(assembled)
                    prediction  = pred_result.select("prediction").collect()[0][0]

                    cls_target_col = st.session_state.get("cls_target", "target")
                    unique_labels  = [r[0] for r in train_df.select(cls_target_col).distinct().orderBy(cls_target_col).collect()]
                    pred_int  = int(prediction)
                    label_str = str(unique_labels[pred_int]) if pred_int < len(unique_labels) else str(pred_int)

                    st.markdown(
                        result_card_html(
                            "PREDICTION RESULT", label_str,
                            f"Model: {pred_model_name}",
                            f"{t['accent']}10", f"{t['accent']}50", t["accent"],
                        ),
                        unsafe_allow_html=True,
                    )

                    try:
                        prob      = pred_result.select("probability").collect()[0][0]
                        prob_list = prob.toArray().tolist()
                        prob_df   = pd.DataFrame({
                            "Class": [str(l) for l in unique_labels[:len(prob_list)]],
                            "Probability": [round(p, 4) for p in prob_list],
                        })
                        st.markdown("**Class Probabilities:**")
                        st.dataframe(prob_df, use_container_width=True)
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

    # ==========================================================================
    # REGRESSION
    # ==========================================================================
    with reg_tab:
        st.markdown("### 🎯 Step 1 — Target Column")
        reg_target = st.selectbox("Target column (numeric)", train_df.columns, key="reg_target")

        reg_dtype = dict(train_df.dtypes).get(reg_target, "string")
        nt_check  = ("int", "bigint", "double", "float", "decimal", "long", "short", "tinyint")
        if not any(x in reg_dtype.lower() for x in nt_check):
            st.warning(f"⚠️ `{reg_target}` is type `{reg_dtype}`. Regression requires a numeric target.")

        try:
            stats = train_df.select(
                F.mean(reg_target).alias("mean"), F.stddev(reg_target).alias("std"),
                F.min(reg_target).alias("min"),   F.max(reg_target).alias("max"),
            ).collect()[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean", f"{stats['mean']:.4f}" if stats['mean'] else "N/A")
            c2.metric("Std",  f"{stats['std']:.4f}"  if stats['std']  else "N/A")
            c3.metric("Min",  f"{stats['min']:.4f}"  if stats['min']  else "N/A")
            c4.metric("Max",  f"{stats['max']:.4f}"  if stats['max']  else "N/A")
        except Exception:
            pass

        st.markdown("---")
        st.markdown("### ⚙️ Step 2 — Feature Columns")
        reg_candidates = get_candidate_features(train_df, reg_target)
        if not reg_candidates:
            st.error("❌ No usable feature columns found."); st.stop()

        reg_features = st.multiselect("Features", reg_candidates, default=reg_candidates, key="reg_features")
        if not reg_features:
            st.warning("Select at least one feature."); st.stop()

        st.markdown("---")
        st.markdown("### 🧪 Step 3 — Choose Models")
        available_reg = get_available_regression_models()
        st.caption("ℹ️ Isotonic Regression works best with a single monotonic feature.")

        selected_reg = st.multiselect("Models to train", available_reg, default=available_reg[:2], key="selected_reg_models")
        st.markdown("---")

        if st.button("🚀 Train & Evaluate", key="reg_train_btn"):
            if not selected_reg:
                st.warning("Select at least one model."); st.stop()

            with st.spinner("Assembling feature vector..."):
                try:
                    tr_r, te_r, reg_label = prepare_data(train_df, test_df, reg_features, reg_target, do_index=False)
                except Exception as e:
                    st.error(f"❌ Assembly failed: {e}"); st.stop()

            st.success(f"✅ Feature vector: {len(reg_features)} columns | Label: `{reg_label}`")

            reg_results = {}; reg_models_trained = {}; reg_errors = {}
            prog = st.progress(0); status = st.empty()

            for i, mname in enumerate(selected_reg):
                status.markdown(f"⏳ Training **{mname}**...")
                try:
                    m = train_regression_model(mname, tr_r, reg_label)
                    reg_results[mname] = evaluate_regression_model(m, te_r, reg_label)
                    reg_models_trained[mname] = m
                except Exception as e:
                    reg_errors[mname] = str(e)
                prog.progress(int((i + 1) / len(selected_reg) * 100))

            status.empty(); prog.empty()
            for name, err in reg_errors.items():
                st.error(f"❌ **{name}** failed: {err}")
            if not reg_results:
                st.error("All models failed."); st.stop()

            st.session_state.reg_models   = reg_models_trained
            st.session_state.reg_results  = reg_results
            st.session_state.reg_target   = reg_target
            st.session_state.reg_features = reg_features

            st.markdown("## 📊 Evaluation Results")
            rdf_r = pd.DataFrame(reg_results).T.round(4)
            st.dataframe(rdf_r.style.apply(highlight_reg, axis=None), use_container_width=True)

            best_reg_name, best_reg_metrics = select_best_regression_model(reg_results, "R2")
            st.markdown(
                model_banner_html(
                    "🏆 BEST MODEL (by R²)", best_reg_name,
                    f"{t['secondary']}10", f"{t['secondary']}55", t["secondary"],
                ),
                unsafe_allow_html=True,
            )

            st.markdown("### 🏆 Best Model Metrics")
            cr = st.columns(len(best_reg_metrics))
            for ci, (k, v) in enumerate(best_reg_metrics.items()):
                cr[ci].metric(k, f"{v:.4f}")

            st.markdown("### 📈 Model Comparison")
            fig_r = themed_bar_chart(rdf_r, list(rdf_r.columns), "Regression Comparison")
            st.pyplot(fig_r); plt.close()

            st.markdown("### 🔍 Per-Model Detail")
            for mname, metrics in reg_results.items():
                is_best = mname == best_reg_name
                with st.expander(f"{'🥇' if is_best else '📋'} {mname}{'  ← Best' if is_best else ''}", expanded=is_best):
                    dc = st.columns(len(metrics))
                    for ci, (k, v) in enumerate(metrics.items()):
                        dc[ci].metric(k, f"{v:.4f}")

            log(f"[REG] {list(reg_results.keys())} | Best: {best_reg_name} | R2={best_reg_metrics['R2']:.4f}")

        # ── Regression Prediction ─────────────────────────────────────────────
        if st.session_state.get("reg_models") and st.session_state.get("reg_features"):
            st.markdown("---")
            st.markdown("## 🔮 Predict — Regression")
            st.caption("Enter feature values and click Predict.")

            pred_reg_name = st.selectbox("Model", list(st.session_state.reg_models.keys()), key="pred_reg_model_name")
            reg_input_vals = prediction_inputs(st.session_state.reg_features, "reg_inp")

            if st.button("📐 Predict Value", key="reg_predict_btn"):
                try:
                    feat_cols   = st.session_state.reg_features
                    spark_local = train_df.sparkSession
                    input_df    = spark_local.createDataFrame([tuple(reg_input_vals[fc] for fc in feat_cols)], feat_cols)
                    assembler   = VectorAssembler(inputCols=feat_cols, outputCol="features", handleInvalid="keep")
                    assembled   = assembler.transform(input_df)

                    pred_result = st.session_state.reg_models[pred_reg_name].transform(assembled)
                    prediction  = pred_result.select("prediction").collect()[0][0]

                    st.markdown(
                        result_card_html(
                            "PREDICTED VALUE", f"{prediction:.4f}",
                            f"Model: {pred_reg_name} | Target: {st.session_state.reg_target}",
                            f"{t['secondary']}10", f"{t['secondary']}50", t["secondary"],
                        ),
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")