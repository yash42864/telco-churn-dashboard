"""Page 7: Predict & What-if Simulator.

Interactive scoring sandbox replicating the PySpark class-weighted Logistic Regression model.
Features real-time probability prediction, waterfall feature attribution,
what-if sensitivity comparison, and high-risk vs loyal benchmark presets.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from utils.data import get_lr_params
from utils.theme import (
    apply_custom_theme,
    render_page_header,
    render_kpi_card,
    get_plotly_template,
    COLOR_CHURN,
    COLOR_SAFE,
    COLOR_PRIMARY,
    COLOR_AMBER,
)
from utils.model import score_customer_profile

apply_custom_theme()

render_page_header(
    title="Predict & What-if Simulator",
    subtitle="Simulate subscription outcomes, score prospective profiles, and test retention interventions live.",
    badge="Inference Simulator",
    icon="🔮",
    how_to_read="""
    **Simulator Guide**:
    - **Inference Pipeline**: Directly implements the PySpark model equation: $P(\\text{churn}) = \\sigma(\\beta^T x + \\beta_0)$ with standard deviation scaling.
    - **Presets**: Load extreme archetypes ("High Risk" vs "Loyal Champion") with a single click.
    - **What-if Intervention Engine**: Test how contractual upgrades or service add-ons mitigate cancellation risk in real-time.
    """,
)

lr_params = get_lr_params()

if lr_params is None:
    st.info("Logistic Regression parameters (`outputs/lr_params.json`) are required.")
    st.stop()

# Parse categories dynamically from feature_names
feature_names = lr_params["feature_names"]
cat_options = {}
for nm in feature_names[4:]:
    if "=" in nm:
        feat, val = nm.split("=", 1)
        cat_options.setdefault(feat, []).append(val)

# Presets Management
if "sim_preset" not in st.session_state:
    st.session_state["sim_preset"] = "default"

st.markdown("### ⚡ Scenario Presets")
st.caption("Select a standard archetype or customize inputs manually below.")

p_c1, p_c2, p_c3 = st.columns(3)

with p_c1:
    if st.button("🔥 Load High-Risk Example", use_container_width=True):
        st.session_state["sim_senior"] = 0
        st.session_state["sim_tenure"] = 2
        st.session_state["sim_monthly"] = 85.0
        st.session_state["sim_total"] = 170.0
        st.session_state["sim_Contract"] = "Month-to-month"
        st.session_state["sim_Internet_Service"] = "Fiber optic"
        st.session_state["sim_Payment_Method"] = "Electronic check"
        st.session_state["sim_Paperless_Billing"] = "Yes"
        st.session_state["sim_Tech_Support"] = "No"
        st.session_state["sim_Online_Security"] = "No"
        st.session_state["sim_Online_Backup"] = "No"
        st.session_state["sim_Device_Protection"] = "No"
        st.session_state["sim_Streaming_TV"] = "No"
        st.session_state["sim_Streaming_Movies"] = "No"
        st.session_state["sim_Multiple_Lines"] = "No"
        st.session_state["sim_Phone_Service"] = "Yes"
        st.session_state["sim_Partner"] = "No"
        st.session_state["sim_Dependents"] = "No"
        st.session_state["sim_Gender"] = "Female"
        st.rerun()

with p_c2:
    if st.button("🛡️ Load Loyal Champion Example", use_container_width=True):
        st.session_state["sim_senior"] = 0
        st.session_state["sim_tenure"] = 60
        st.session_state["sim_monthly"] = 65.0
        st.session_state["sim_total"] = 3900.0
        st.session_state["sim_Contract"] = "Two year"
        st.session_state["sim_Internet_Service"] = "DSL"
        st.session_state["sim_Payment_Method"] = "Bank transfer (automatic)"
        st.session_state["sim_Paperless_Billing"] = "No"
        st.session_state["sim_Tech_Support"] = "Yes"
        st.session_state["sim_Online_Security"] = "Yes"
        st.session_state["sim_Online_Backup"] = "Yes"
        st.session_state["sim_Device_Protection"] = "Yes"
        st.session_state["sim_Streaming_TV"] = "Yes"
        st.session_state["sim_Streaming_Movies"] = "Yes"
        st.session_state["sim_Multiple_Lines"] = "Yes"
        st.session_state["sim_Phone_Service"] = "Yes"
        st.session_state["sim_Partner"] = "Yes"
        st.session_state["sim_Dependents"] = "Yes"
        st.session_state["sim_Gender"] = "Male"
        st.rerun()

with p_c3:
    if st.button("🔄 Reset to Neutral Default", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("sim_"):
                del st.session_state[k]
        st.rerun()

# Default values helper
def get_val(key, default):
    return st.session_state.get(f"sim_{key}", default)

# Form Layout: Numerical Inputs & Categorical Selectors
st.markdown("---")
st.markdown("### 📝 Customer Account Attributes")

col_num1, col_num2, col_num3, col_num4 = st.columns(4)

with col_num1:
    tenure_in = st.number_input(
        "Tenure Duration (Months)",
        min_value=0,
        max_value=72,
        value=int(get_val("tenure", 12)),
        step=1,
    )
with col_num2:
    monthly_in = st.number_input(
        "Monthly Charges ($)",
        min_value=18.0,
        max_value=125.0,
        value=float(get_val("monthly", 70.0)),
        step=5.0,
    )
with col_num3:
    default_total = float(tenure_in * monthly_in) if tenure_in > 0 else 50.0
    total_in = st.number_input(
        "Total Accumulated Spend ($)",
        min_value=0.0,
        max_value=10000.0,
        value=float(get_val("total", default_total)),
        step=50.0,
    )
with col_num4:
    senior_in = st.selectbox(
        "Senior Citizen Status",
        options=[0, 1],
        format_func=lambda x: "Yes (Senior)" if x == 1 else "No (Non-Senior)",
        index=int(get_val("senior", 0)),
    )

st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
st.markdown("#### Service Subscriptions & Demographics")

# Arrange categorical inputs cleanly across 3 columns
cat_cols = st.columns(3)
selected_cats = {}

cat_keys = list(cat_options.keys())
for i, feat in enumerate(cat_keys):
    options = cat_options[feat]
    default_opt = get_val(feat, options[0])
    idx = options.index(default_opt) if default_opt in options else 0

    with cat_cols[i % 3]:
        selected_cats[feat] = st.selectbox(
            feat.replace("_", " "),
            options=options,
            index=idx,
            key=f"cat_input_{feat}",
        )

# Compute Model Probability & Contributions
prob, logit, contributions = score_customer_profile(
    lr_params,
    senior=senior_in,
    tenure=tenure_in,
    monthly=monthly_in,
    total=total_in,
    categories=selected_cats,
)

# Display Simulation Results
st.markdown("---")
st.markdown("### 📊 Churn Prediction & Risk Diagnosis")

res_col1, res_col2 = st.columns([1, 1.8])

with res_col1:
    if prob >= 0.60:
        badge_text = "🚨 CRITICAL ATTRITION RISK"
        badge_color = COLOR_CHURN
        risk_desc = "Account indicates acute defection likelihood. Immediate proactive intervention required."
    elif prob >= 0.40:
        badge_text = "⚠️ ELEVATED ATTRITION RISK"
        badge_color = COLOR_AMBER
        risk_desc = "Moderately elevated vulnerability. Candidate for targeted loyalty offers."
    else:
        badge_text = "🛡️ SECURE / LOW RISK"
        badge_color = COLOR_SAFE
        risk_desc = "Healthy subscriber relationship. Standard engagement cadences recommended."

    fig_sim_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100.0,
            number=dict(suffix="%", font=dict(family="Sora", size=38, weight="bold", color=badge_color)),
            title=dict(text=f"<b>{badge_text}</b>", font=dict(size=14, color=badge_color)),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="white"),
                bar=dict(color=badge_color, thickness=0.32),
                bgcolor="rgba(22, 31, 48, 0.7)",
                borderwidth=1,
                bordercolor="rgba(255,255,255,0.1)",
                steps=[
                    dict(range=[0, 40], color="rgba(16, 185, 129, 0.15)"),
                    dict(range=[40, 60], color="rgba(245, 158, 11, 0.15)"),
                    dict(range=[60, 100], color="rgba(244, 63, 94, 0.15)"),
                ],
            ),
        )
    )
    fig_sim_gauge.update_layout(
        template=get_plotly_template(),
        height=260,
        margin=dict(t=30, b=10, l=20, r=20),
    )
    st.plotly_chart(fig_sim_gauge, use_container_width=True)

    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid {badge_color}; padding: 0.8rem 1rem;">
            <b>Operational Context</b>: {risk_desc}<br>
            <span style="font-size:0.75rem; color:#94A3B8;">Raw Log-Odds (Logit): {logit:+.3f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with res_col2:
    st.markdown("#### 🔬 Factor Contribution Waterfall")
    st.caption("Top factors pushing probability UP (Red) or DOWN (Green) relative to intercept.")

    top_k = contributions[:7]
    names = [c[0] for c in top_k][::-1]
    vals = [c[1] for c in top_k][::-1]
    bar_cols = [COLOR_CHURN if v > 0 else COLOR_SAFE for v in vals]

    fig_waterfall = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=bar_cols,
            text=[f"{v:+.3f}" for v in vals],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Impact: %{x:+.4f}<extra></extra>",
        )
    )
    fig_waterfall.update_layout(
        template=get_plotly_template(),
        xaxis=dict(title="Contribution to Log-Odds ($\\beta_i \\times x_i$)"),
        yaxis=dict(title=""),
        height=280,
        margin=dict(t=10, b=30, l=150, r=40),
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

# 4. What-if Intervention Simulator Panel
st.markdown("---")
st.markdown("### 🎚️ What-If Retention Intervention Engine")
st.caption("Simulate the exact risk reduction achieved by applying strategic account interventions.")

w_c1, w_c2 = st.columns([1, 1.2])

with w_c1:
    intervention_type = st.selectbox(
        "Select Strategic Lever to Modify",
        options=[
            "Upgrade Contract to One Year",
            "Upgrade Contract to Two Year",
            "Add Tech Support Service",
            "Add Online Security Service",
            "Switch Payment to Automatic Bank Transfer",
            "Bundle Fiber + Tech Support + Security",
        ],
    )

    # Clone current categories
    modified_cats = dict(selected_cats)
    modified_tenure = tenure_in
    modified_monthly = monthly_in

    if intervention_type == "Upgrade Contract to One Year":
        modified_cats["Contract"] = "One year"
    elif intervention_type == "Upgrade Contract to Two Year":
        modified_cats["Contract"] = "Two year"
    elif intervention_type == "Add Tech Support Service":
        modified_cats["Tech_Support"] = "Yes"
        modified_monthly += 5.0
    elif intervention_type == "Add Online Security Service":
        modified_cats["Online_Security"] = "Yes"
        modified_monthly += 5.0
    elif intervention_type == "Switch Payment to Automatic Bank Transfer":
        modified_cats["Payment_Method"] = "Bank transfer (automatic)"
    elif intervention_type == "Bundle Fiber + Tech Support + Security":
        modified_cats["Internet_Service"] = "Fiber optic"
        modified_cats["Tech_Support"] = "Yes"
        modified_cats["Online_Security"] = "Yes"
        modified_cats["Contract"] = "One year"

    # Compute modified probability
    new_prob, _, _ = score_customer_profile(
        lr_params,
        senior=senior_in,
        tenure=modified_tenure,
        monthly=modified_monthly,
        total=total_in,
        categories=modified_cats,
    )
    delta_risk = (new_prob - prob) * 100.0

with w_c2:
    w1, w2 = st.columns(2)
    with w1:
        render_kpi_card(
            title="Baseline Risk",
            value=f"{prob:.1%}",
            delta="Current State",
            delta_type="neutral",
            icon="⏱️",
        )
    with w2:
        d_type = "mint" if delta_risk < 0 else "coral"
        d_sign = "+" if delta_risk > 0 else ""
        render_kpi_card(
            title="Post-Intervention Risk",
            value=f"{new_prob:.1%}",
            delta=f"{d_sign}{delta_risk:.1f}% risk shift",
            delta_type=d_type,
            icon="✨",
            sublabel="Projected new probability",
        )

    if delta_risk < 0:
        st.success(
            f"🎉 **Positive Retention Outcome**: Applying **{intervention_type}** is projected to reduce this customer's "
            f"cancellation likelihood from **{prob:.1%}** down to **{new_prob:.1%}** (a **{abs(delta_risk):.1f}% point reduction**)."
        )
    else:
        st.warning(
            f"⚠️ Applying this modification would increase churn risk by **{delta_risk:.1f}% points**."
        )
