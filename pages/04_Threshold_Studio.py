"""Page 4: Threshold and ROC Studio.

Interactive probability decision boundary tuning, real-time confusion matrix
recalculation, live ROC and Precision-Recall curves, and automated business optimization.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from utils.data import get_customer_scores
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
)
from utils.model import (
    compute_confusion_matrix,
    compute_roc_curve,
    compute_pr_curve,
    optimize_threshold,
)

apply_custom_theme()

render_page_header(
    title="Threshold & ROC Studio",
    subtitle="Calibrate the probabilistic decision threshold to align model sensitivity with enterprise economics.",
    badge="Decision Optimization",
    icon="🎯",
    how_to_read="""
    **Decision Boundary Guide**:
    - **Default 0.50 Threshold**: Standard mathematical cutoff, but rarely optimal for asymmetric business costs.
    - **Lowering Threshold (<0.50)**: Increases **Recall** (captures more potential churners) at the cost of more false alarms (lower Precision).
    - **Raising Threshold (>0.50)**: Increases **Precision** (avoids wasted retention incentives) but allows more churners to defect undetected (lower Recall).
    - **Optimization Presets**: Instantly solves for optimal threshold according to F1, constrained precision, or net campaign profit.
    """,
)

scores_df = get_customer_scores()

if scores_df is None or scores_df.empty:
    st.info("Customer scores dataset is required for the Threshold Studio.")
    st.stop()

y_true = scores_df["label"].to_numpy().astype(int)
y_prob = scores_df["churn_prob"].to_numpy()

# Initialize threshold in session state
if "threshold_val" not in st.session_state:
    st.session_state["threshold_val"] = 0.50

# 1. Preset Optimization Controls
st.markdown("### 🎛️ Strategic Threshold Presets")
st.caption("Click a preset to automatically compute and set the mathematically optimal decision boundary.")

p_col1, p_col2, p_col3, p_col4 = st.columns(4)

with p_col1:
    if st.button("⚖️ Maximize F1 Score", use_container_width=True, help="Balances precision and recall harmonic mean"):
        best_t, _ = optimize_threshold(y_true, y_prob, criterion="f1")
        st.session_state["threshold_val"] = round(float(best_t), 2)
        st.rerun()

with p_col2:
    if st.button("🎯 Max Recall (Prec ≥ 50%)", use_container_width=True, help="Maximizes churn capture while ensuring at least 1 in 2 offers is targeted correctly"):
        best_t, _ = optimize_threshold(y_true, y_prob, criterion="recall_with_min_prec", min_precision=0.50)
        st.session_state["threshold_val"] = round(float(best_t), 2)
        st.rerun()

with p_col3:
    if st.button("💰 Maximize Profit ($1k LTV, $50 Offer)", use_container_width=True, help="Maximizes expected net campaign profit with 25% retention success"):
        best_t, _ = optimize_threshold(
            y_true, y_prob, criterion="profit",
            customer_value=1000.0, offer_cost=50.0, success_rate=0.25
        )
        st.session_state["threshold_val"] = round(float(best_t), 2)
        st.rerun()

with p_col4:
    if st.button("🔄 Reset to 0.50 Baseline", use_container_width=True):
        st.session_state["threshold_val"] = 0.50
        st.rerun()

# Threshold slider
active_threshold = st.slider(
    "Decision Cutoff Threshold (Probability >= Threshold is flagged as Churn)",
    min_value=0.05,
    max_value=0.95,
    value=float(st.session_state["threshold_val"]),
    step=0.01,
    help="Drag to evaluate performance changes across the entire customer cohort",
)
st.session_state["threshold_val"] = active_threshold

# Compute live metrics
m = compute_confusion_matrix(y_true, y_prob, active_threshold)
tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
precision, recall, f1, accuracy = m["precision"], m["recall"], m["f1"], m["accuracy"]
specificity, fpr = m["specificity"], m["fpr"]

# Display KPI strip
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_kpi_card(
        title="Recall (Sensitivity)",
        value=f"{recall:.1%}",
        delta=f"Catches {tp:,} of {m['total_churners']:,}",
        delta_type="mint" if recall >= 0.75 else "coral",
        icon="🎯",
        sublabel="Proportion of churners caught",
    )
with k2:
    render_kpi_card(
        title="Precision",
        value=f"{precision:.1%}",
        delta=f"{fp:,} false alarms",
        delta_type="mint" if precision >= 0.50 else "coral",
        icon="🔬",
        sublabel="Accuracy of churn flags",
    )
with k3:
    render_kpi_card(
        title="F1 Score",
        value=f"{f1:.3f}",
        delta="Harmonic Mean",
        delta_type="neutral",
        icon="⚖️",
        sublabel="Combined balance score",
    )
with k4:
    render_kpi_card(
        title="Specificity",
        value=f"{specificity:.1%}",
        delta=f"Retains {tn:,} safe",
        delta_type="mint",
        icon="🛡️",
        sublabel="True Negative rate",
    )
with k5:
    render_kpi_card(
        title="Total Flagged",
        value=f"{(tp + fp):,}",
        delta=f"{((tp + fp) / len(y_true)):.1%} of portfolio",
        delta_type="neutral",
        icon="📣",
        sublabel="Campaign reach volume",
    )

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# 2. Live ROC and PR Curves with Marker
curve_col1, curve_col2 = st.columns(2)

fpr_arr, tpr_arr, _, auc_roc = compute_roc_curve(y_true, y_prob)
rec_arr, prec_arr, _, auc_pr = compute_pr_curve(y_true, y_prob)

with curve_col1:
    st.markdown("### 📈 Receiver Operating Characteristic (ROC)")
    fig_roc = go.Figure()

    # ROC Line
    fig_roc.add_trace(
        go.Scatter(
            x=fpr_arr,
            y=tpr_arr,
            mode="lines",
            name=f"PySpark LogReg (AUC = {auc_roc:.4f})",
            line=dict(color=COLOR_PRIMARY, width=3),
            hovertemplate="FPR: %{x:.3f}<br>TPR (Recall): %{y:.3f}<extra></extra>",
        )
    )

    # Random chance diagonal
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guessing (AUC = 0.50)",
            line=dict(color="rgba(148, 163, 184, 0.4)", width=1.5, dash="dash"),
            hoverinfo="none",
        )
    )

    # Current Threshold Point
    fig_roc.add_trace(
        go.Scatter(
            x=[fpr],
            y=[recall],
            mode="markers+text",
            name=f"Threshold = {active_threshold:.2f}",
            marker=dict(color=COLOR_CHURN, size=14, symbol="circle", line=dict(color="white", width=2)),
            text=[f"  t={active_threshold:.2f} (FPR={fpr:.2f}, Rec={recall:.2f})"],
            textposition="bottom right",
            textfont=dict(color=COLOR_CHURN, size=11, weight="bold"),
        )
    )

    fig_roc.update_layout(
        template=get_plotly_template(),
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[-0.02, 1.02]),
        yaxis=dict(title="True Positive Rate (Recall)", range=[-0.02, 1.02]),
        height=380,
        margin=dict(t=20, b=40, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with curve_col2:
    st.markdown("### 🎯 Precision-Recall (PR) Curve")
    fig_pr = go.Figure()

    # PR Line
    fig_pr.add_trace(
        go.Scatter(
            x=rec_arr,
            y=prec_arr,
            mode="lines",
            name=f"PR Curve (AUC = {auc_pr:.4f})",
            line=dict(color=COLOR_SAFE, width=3),
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        )
    )

    # Baseline positive prevalence line
    prevalence = np.mean(y_true)
    fig_pr.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[prevalence, prevalence],
            mode="lines",
            name=f"Prevalence Baseline ({prevalence:.1%})",
            line=dict(color="rgba(148, 163, 184, 0.4)", width=1.5, dash="dash"),
            hoverinfo="none",
        )
    )

    # Current Threshold Point
    fig_pr.add_trace(
        go.Scatter(
            x=[recall],
            y=[precision],
            mode="markers+text",
            name=f"Threshold = {active_threshold:.2f}",
            marker=dict(color=COLOR_CHURN, size=14, symbol="circle", line=dict(color="white", width=2)),
            text=[f"  t={active_threshold:.2f} (Rec={recall:.2f}, Prec={precision:.2f})"],
            textposition="top right",
            textfont=dict(color=COLOR_CHURN, size=11, weight="bold"),
        )
    )

    fig_pr.update_layout(
        template=get_plotly_template(),
        xaxis=dict(title="Recall (Sensitivity)", range=[-0.02, 1.02]),
        yaxis=dict(title="Precision (Positive Predictive Value)", range=[-0.02, 1.02]),
        height=380,
        margin=dict(t=20, b=40, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_pr, use_container_width=True)

# 3. Dynamic Live Confusion Matrix Heatmap
st.markdown("---")
st.markdown(f"### 🔲 Real-time Confusion Matrix at Cutoff: $p \\ge {active_threshold:.2f}$")

c_mat1, c_mat2 = st.columns([1, 1.4])

with c_mat1:
    cm_z = [[tn, fp], [fn, tp]]
    cm_text = [
        [f"<b>TN (Retained & Safe)</b><br>{tn:,}<br>({tn/m['total_retained']:.1%})", f"<b>FP (False Alarm)</b><br>{fp:,}<br>({fp/m['total_retained']:.1%})"],
        [f"<b>FN (Undetected Defector)</b><br>{fn:,}<br>({fn/m['total_churners']:.1%})", f"<b>TP (Caught Churner)</b><br>{tp:,}<br>({tp/m['total_churners']:.1%})"],
    ]

    fig_live_cm = go.Figure(
        data=go.Heatmap(
            z=cm_z,
            x=["Predicted Retained", "Predicted Churned"],
            y=["Actual Retained", "Actual Churned"],
            colorscale=[[0, "rgba(22, 31, 48, 0.8)"], [1, COLOR_PRIMARY]],
            text=cm_text,
            texttemplate="%{text}",
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Subscribers: %{z:,}<extra></extra>",
        )
    )
    fig_live_cm.update_layout(
        template=get_plotly_template(),
        height=260,
        margin=dict(t=10, b=20, l=40, r=20),
    )
    st.plotly_chart(fig_live_cm, use_container_width=True)

with c_mat2:
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid {COLOR_PRIMARY}; padding: 1.1rem 1.4rem;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #818CF8; margin-bottom: 0.5rem;">
                Live Operational Assessment (Threshold = {active_threshold:.2f})
            </div>
            <div style="font-size: 0.9rem; line-height: 1.6;">
                • <b>Detected Churners (True Positives)</b>: The campaign proactively intercepts <b>{tp:,}</b> churners
                out of {m['total_churners']:,} total defectors (<b>{recall:.1%} capture rate</b>).<br>
                • <b>Missed Defectors (False Negatives)</b>: <b>{fn:,}</b> churners leave without any retention contact.<br>
                • <b>Wasted Offers (False Positives)</b>: <b>{fp:,}</b> retention discounts are extended to accounts
                who were going to stay anyway (<b>{precision:.1%} precision</b>).<br>
                • <b>Total Retention Offers Extended</b>: <b>{(tp + fp):,}</b> accounts flagged for proactive treatment.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "📌 **Evaluation Methodology Caption**: Customer probabilities were scored in-sample across the entire 7,043 "
    "customer cohort using the class-weighted PySpark Logistic Regression model (`scripts/08_export.py`). "
    "Consequently, full-cohort metrics (e.g. AUC ~0.86) appear slightly more optimistic than the held-out "
    "test partition evaluation (AUC 0.8515) reported on the Model Lab page."
)
