"""Page 3: Model Lab.

Comprehensive evaluation of 7 PySpark classifiers, precision/recall trade-offs,
confusion matrices with plain-English business translations, cluster training times,
Linear Regression auxiliary analysis, and model feature importances.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from utils.data import (
    get_model_comparison,
    get_feature_importance,
    get_regression_metrics,
)
from utils.theme import (
    apply_custom_theme,
    render_page_header,
    render_kpi_card,
    get_plotly_template,
    COLOR_CHURN,
    COLOR_SAFE,
    COLOR_PRIMARY,
    COLOR_AMBER,
    COLOR_PURPLE,
    COLOR_SKY,
)

apply_custom_theme()

render_page_header(
    title="Model Lab & Algorithm Evaluation",
    subtitle="Rigorous benchmark of 7 distributed PySpark MLlib classifiers, training runtimes, and decision metrics.",
    badge="MLlib Benchmarks",
    icon="🧪",
    how_to_read="""
    **Model Evaluation Guide**:
    - **Recall Priority**: In customer churn, missing a defector (False Negative) is substantially more costly than extending an unnecessary offer (False Positive). Balanced models intentionally optimize for Recall.
    - **Class Weighting**: Balanced variants adjust gradient updates inversely proportional to class frequencies, dramatically improving churn detection from ~57% to ~78%.
    - **Radar Chart**: Compares model silhouettes across 6 normalized performance dimensions simultaneously.
    - **Auxiliary Linear Regression**: Evaluates monthly charges as a target. High $R^2$ (0.999) validates pricing determinism from service add-ons.
    """,
)

models_df = get_model_comparison()
feat_imp_df = get_feature_importance()
reg_df = get_regression_metrics()

if models_df is None or models_df.empty:
    st.info("Model comparison data is unavailable.")
    st.stop()

# 1. Model Comparison Leaderboard Table
st.markdown("### 🏆 PySpark Model Leaderboard")
st.caption("Held-out test set performance (80/20 train/test split, 1,409 evaluation rows). Highest score in each column highlighted.")

# Create display table with highlighted styling
metric_cols = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "train_time_sec"]

display_df = models_df.copy()
display_cols = ["model", "accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "TP", "FP", "TN", "FN", "train_time_sec"]

# Format table for display
st.dataframe(
    display_df[display_cols].style.highlight_max(
        subset=["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"],
        color="rgba(16, 185, 129, 0.25)",
    ).highlight_min(
        subset=["train_time_sec", "FN"],
        color="rgba(129, 140, 248, 0.25)",
    ).format({
        "accuracy": "{:.4f}",
        "precision": "{:.4f}",
        "recall": "{:.4f}",
        "f1": "{:.4f}",
        "auc_roc": "{:.4f}",
        "auc_pr": "{:.4f}",
        "train_time_sec": "{:.2f}s",
        "TP": "{:,}",
        "FP": "{:,}",
        "TN": "{:,}",
        "FN": "{:,}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# 2. Metric Selector Bar Chart & Radar Chart
col_bench1, col_bench2 = st.columns([1.2, 1])

with col_bench1:
    st.markdown("### 📊 Multi-Metric Comparison")
    chosen_metrics = st.multiselect(
        "Select Performance Metrics to Compare",
        options=["recall", "precision", "f1", "auc_roc", "auc_pr", "accuracy"],
        default=["recall", "f1", "auc_roc"],
        key="model_bar_metrics",
    )

    if chosen_metrics:
        melted_df = models_df.melt(
            id_vars=["model"],
            value_vars=chosen_metrics,
            var_name="Metric",
            value_name="Score",
        )
        fig_bar = px.bar(
            melted_df,
            x="model",
            y="Score",
            color="Metric",
            barmode="group",
            text_auto=".3f",
            color_discrete_sequence=[COLOR_PRIMARY, COLOR_SAFE, COLOR_CHURN, COLOR_AMBER, COLOR_PURPLE, COLOR_SKY],
        )
        fig_bar.update_layout(
            template=get_plotly_template(),
            yaxis=dict(range=[0, 1.05], title="Metric Score"),
            xaxis=dict(title="Model Variant"),
            height=370,
            margin=dict(t=20, b=40, l=40, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Select at least one metric to display.")

with col_bench2:
    st.markdown("### 🕸️ Model Performance Radar")
    available_models = models_df["model"].tolist()
    radar_models = st.multiselect(
        "Select 2 to 4 Models for Radar Comparison",
        options=available_models,
        default=["LogReg_balanced", "LogReg", "SVM_balanced"] if "LogReg_balanced" in available_models else available_models[:3],
        max_selections=4,
        key="radar_models_select",
    )

    radar_cats = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"]
    fig_radar = go.Figure()

    radar_colors = [COLOR_CHURN, COLOR_SAFE, COLOR_PRIMARY, COLOR_AMBER]

    for i, m_name in enumerate(radar_models):
        m_row = models_df[models_df["model"] == m_name]
        if not m_row.empty:
            vals = [float(m_row.iloc[0][c]) for c in radar_cats]
            vals.append(vals[0])  # close loop
            cats_closed = [c.upper().replace("_", " ") for c in radar_cats] + [radar_cats[0].upper().replace("_", " ")]

            fig_radar.add_trace(
                go.Scatterpolar(
                    r=vals,
                    theta=cats_closed,
                    name=m_name,
                    line_color=radar_colors[i % len(radar_colors)],
                    fill="toself",
                    fillcolor=radar_colors[i % len(radar_colors)],
                    opacity=0.35,
                )
            )

    fig_radar.update_layout(
        template=get_plotly_template(),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        ),
        showlegend=True,
        height=370,
        margin=dict(t=30, b=30, l=40, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# 3. Confusion Matrix Viewer & Precision-Recall Scatter
col_cm1, col_cm2 = st.columns([1, 1.2])

with col_cm1:
    st.markdown("### 🔲 Confusion Matrix Viewer")
    selected_cm_model = st.selectbox(
        "Select Model to Inspect Confusion Matrix",
        options=models_df["model"].tolist(),
        index=1 if "LogReg_balanced" in models_df["model"].tolist() else 0,
        key="cm_model_select",
    )

    cm_row = models_df[models_df["model"] == selected_cm_model].iloc[0]
    tp = int(cm_row["TP"])
    fp = int(cm_row["FP"])
    tn = int(cm_row["TN"])
    fn = int(cm_row["FN"])
    total_eval = tp + fp + tn + fn
    total_churners = tp + fn
    total_retained = tn + fp

    # Matrix: rows = Actual (Retained=0, Churned=1), cols = Predicted (Retained=0, Churned=1)
    # [ [TN, FP], [FN, TP] ]
    z_cm = [[tn, fp], [fn, tp]]
    text_cm = [
        [f"<b>TN</b><br>{tn:,}<br>({tn/total_retained:.1%})", f"<b>FP (Waste)</b><br>{fp:,}<br>({fp/total_retained:.1%})"],
        [f"<b>FN (Miss)</b><br>{fn:,}<br>({fn/total_churners:.1%})", f"<b>TP (Caught)</b><br>{tp:,}<br>({tp/total_churners:.1%})"],
    ]

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=z_cm,
            x=["Predicted Retained (0)", "Predicted Churned (1)"],
            y=["Actual Retained (0)", "Actual Churned (1)"],
            colorscale=[[0, "rgba(22, 31, 48, 0.8)"], [1, COLOR_PRIMARY]],
            text=text_cm,
            texttemplate="%{text}",
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>",
        )
    )
    fig_cm.update_layout(
        template=get_plotly_template(),
        height=280,
        margin=dict(t=10, b=20, l=40, r=20),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Plain English business explanation
    recall_pct = (tp / total_churners) * 100.0 if total_churners > 0 else 0
    prec_pct = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid {COLOR_PRIMARY}; padding: 0.8rem 1rem;">
            <b>Plain-English Translation for {selected_cm_model}:</b><br>
            • Catches <b>{tp:,} of {total_churners:,}</b> actual churners (<b>{recall_pct:.1f}% Recall</b>).<br>
            • Misses <b>{fn:,}</b> churners who leave undetected.<br>
            • Wastes <b>{fp:,}</b> retention incentives on loyal customers (<b>{prec_pct:.1f}% Precision</b>).
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_cm2:
    st.markdown("### 🎯 Precision vs Recall Trade-Off Space")
    st.caption("Bubble size corresponds to test set AUC-ROC. Notice how balanced variants shift right into high-recall territory.")

    models_df["Family"] = models_df["model"].apply(
        lambda m: "Balanced Variant" if "balanced" in m else "Standard (Unweighted)"
    )

    fig_scatter = px.scatter(
        models_df,
        x="recall",
        y="precision",
        size="auc_roc",
        color="Family",
        text="model",
        color_discrete_map={"Balanced Variant": COLOR_CHURN, "Standard (Unweighted)": COLOR_SAFE},
        hover_data={"accuracy": ":.3f", "f1": ":.3f", "auc_roc": ":.4f"},
    )
    fig_scatter.update_traces(
        textposition="top center",
        marker=dict(sizeref=0.04, sizemin=14, line=dict(width=1.5, color="white")),
    )
    fig_scatter.update_layout(
        template=get_plotly_template(),
        xaxis=dict(title="Recall (Sensitivity)", range=[0.4, 0.88]),
        yaxis=dict(title="Precision", range=[0.4, 0.78]),
        height=370,
        margin=dict(t=20, b=40, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# 4. Distributed Cluster Training Time & Linear Regression Panel
col_time, col_reg = st.columns([1, 1.2])

with col_time:
    st.markdown("### ⏱️ PySpark Distributed Training Time")
    st.caption("Wall-clock execution time on local[*] Apache Spark cluster with 8 executor threads.")

    df_time = models_df.sort_values("train_time_sec", ascending=True)
    fig_time = go.Figure(
        go.Bar(
            x=df_time["train_time_sec"],
            y=df_time["model"],
            orientation="h",
            marker_color=COLOR_PRIMARY,
            text=df_time["train_time_sec"].apply(lambda v: f"{v:.2f}s"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Training Time: %{x:.3f} seconds<extra></extra>",
        )
    )
    fig_time.update_layout(
        template=get_plotly_template(),
        xaxis=dict(title="Seconds (lower is faster)", range=[0, max(df_time["train_time_sec"]) * 1.25]),
        yaxis=dict(title=""),
        height=320,
        margin=dict(t=20, b=30, l=40, r=20),
    )
    st.plotly_chart(fig_time, use_container_width=True)

    st.caption(
        "💡 **Cluster Computing Note**: NaiveBayes executes in single-pass aggregation (0.19s). "
        "LogisticRegression uses iterative L-BFGS optimization across partitioned RDDs, taking 2.87s without class weights."
    )

with col_reg:
    st.markdown("### 📐 Auxiliary Linear Regression: Price Determinism")
    st.caption("Predicting Monthly_Charges from full feature set using PySpark LinearRegression.")

    if reg_df is not None and not reg_df.empty:
        r_row = reg_df.iloc[0]
        r2 = float(r_row["r2"])
        rmse = float(r_row["rmse"])
        mae = float(r_row["mae"])
        b_rmse = float(r_row["baseline_rmse"])
        b_mae = float(r_row["baseline_mae"])

        err_reduction = ((b_rmse - rmse) / b_rmse) * 100.0

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            render_kpi_card(
                title="Model R² Score",
                value=f"{r2:.4f}",
                delta="Near-perfect fit",
                delta_type="mint",
                icon="📊",
                sublabel="Explains 99.89% variance",
            )
        with rc2:
            render_kpi_card(
                title="Model RMSE",
                value=f"${rmse:.2f}",
                delta=f"-{err_reduction:.1f}% vs baseline",
                delta_type="mint",
                icon="📉",
                sublabel=f"Baseline RMSE: ${b_rmse:.2f}",
            )
        with rc3:
            render_kpi_card(
                title="Model MAE",
                value=f"${mae:.2f}",
                delta=f"Baseline: ${b_mae:.2f}",
                delta_type="mint",
                icon="🎯",
                sublabel="Mean absolute error",
            )

        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid {COLOR_SAFE}; padding: 0.8rem 1rem; margin-top: 0.5rem;">
                <b>Why is $R^2 = {r2:.4f}$ so extraordinarily high?</b><br>
                In a telecom enterprise, <code>Monthly_Charges</code> is not an emergent latent variable;
                it is a deterministic mathematical sum of explicit catalog prices (Phone Line + Fiber Optic Tier +
                Streaming Add-ons + Protection Plans). The PySpark Linear Regression model successfully recovers the
                exact billing tariff schedule with only ${rmse:.2f} residual error.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Regression metrics file is missing.")

st.markdown("---")

# 5. Feature Importance Explorer
st.markdown("### 🔬 Feature Importance & Coefficient Explorer")
st.caption("Inspect the mathematical weights driving model predictions.")

if feat_imp_df is not None and not feat_imp_df.empty:
    f_col1, f_col2 = st.columns([1, 1])

    with f_col1:
        model_choice = st.selectbox(
            "Select Model Architecture",
            options=["LogReg_balanced", "LogReg", "DecTree_balanced", "DecTree"],
            index=0,
            key="fi_model_choice",
        )
    with f_col2:
        top_n = st.slider("Number of Top Features to Display", min_value=5, max_value=45, value=15, step=5)

    is_logreg = "LogReg" in model_choice
    df_fi = feat_imp_df[["feature", model_choice]].copy()
    df_fi = df_fi.rename(columns={model_choice: "weight"})

    if is_logreg:
        # Sort by absolute coefficient value
        df_fi["abs_weight"] = df_fi["weight"].abs()
        df_fi = df_fi.sort_values("abs_weight", ascending=False).head(top_n).sort_values("weight", ascending=True)

        bar_colors = [COLOR_CHURN if w > 0 else COLOR_SAFE for w in df_fi["weight"]]
        fig_fi = go.Figure(
            go.Bar(
                x=df_fi["weight"],
                y=df_fi["feature"],
                orientation="h",
                marker_color=bar_colors,
                text=df_fi["weight"].apply(lambda v: f"{v:+.3f}"),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Standardized Coefficient: %{x:+.4f}<extra></extra>",
            )
        )
        x_title = "Standardized Logistic Regression Coefficient (Log-Odds Impact)"
    else:
        # Decision Tree Gini feature importances (non-negative)
        df_fi = df_fi.sort_values("weight", ascending=False).head(top_n).sort_values("weight", ascending=True)
        fig_fi = go.Figure(
            go.Bar(
                x=df_fi["weight"],
                y=df_fi["feature"],
                orientation="h",
                marker_color=COLOR_PRIMARY,
                text=df_fi["weight"].apply(lambda v: f"{v:.4f}"),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Gini Importance: %{x:.4f}<extra></extra>",
            )
        )
        x_title = "Decision Tree Gini Importance"

    fig_fi.update_layout(
        template=get_plotly_template(),
        xaxis=dict(title=x_title),
        yaxis=dict(title=""),
        height=max(400, top_n * 24),
        margin=dict(t=20, b=40, l=120, r=40),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.warning(
        "⚠️ **Collinearity Notice**: One-hot encoded features exhibit structural collinearity across dummy categories "
        "(e.g., `Contract=Month-to-month` vs `Contract=Two year`). Evaluate coefficients within their relative "
        "feature family rather than treating each one-hot column as an independent orthogonal predictor."
    )
else:
    st.info("Feature importance data is missing.")
