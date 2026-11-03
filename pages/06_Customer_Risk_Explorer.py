"""Page 6: Customer Risk Explorer.

Granular subscriber roster exploration with interactive filters,
reactive portfolio KPIs, styled data grid with probability bars, CSV export,
and individual subscriber deep-dive with automated risk factor attribution.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils.data import get_customer_scores, get_lr_params
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
from utils.model import get_customer_row_contributions

apply_custom_theme()

render_page_header(
    title="Customer Risk Explorer",
    subtitle="Operational account lookup, risk cohort filtering, and individual customer risk factor decomposition.",
    badge="Account Roster",
    icon="🔎",
    how_to_read="""
    **Risk Explorer Guide**:
    - **Sidebar Filters**: Segment accounts by contract commitment, service type, tenure span, and predicted churn probability.
    - **Reactive KPI Strip**: Summarizes active filtered volume, mean attrition probability, and monthly recurring revenue exposure.
    - **Customer Lookup**: Select or search any Customer ID to inspect their probability gauge and top positive/negative risk drivers.
    """,
)

scores_df = get_customer_scores()
lr_params = get_lr_params()

if scores_df is None or scores_df.empty:
    st.info("Customer scores dataset is required for the Risk Explorer.")
    st.stop()

# 1. Filters (in an expandable section or sidebar)
with st.sidebar:
    st.markdown("### 🔍 Risk Filters")

    all_contracts = sorted(scores_df["Contract"].dropna().unique().tolist())
    sel_contracts = st.multiselect("Contract Type", options=all_contracts, default=all_contracts)

    all_internets = sorted(scores_df["Internet_Service"].dropna().unique().tolist())
    sel_internets = st.multiselect("Internet Service", options=all_internets, default=all_internets)

    all_payments = sorted(scores_df["Payment_Method"].dropna().unique().tolist())
    sel_payments = st.multiselect("Payment Method", options=all_payments, default=all_payments)

    min_t, max_t = int(scores_df["Tenure_Months"].min()), int(scores_df["Tenure_Months"].max())
    sel_tenure = st.slider("Tenure Span (Months)", min_value=min_t, max_value=max_t, value=(min_t, max_t))

    min_m, max_m = float(scores_df["Monthly_Charges"].min()), float(scores_df["Monthly_Charges"].max())
    sel_monthly = st.slider("Monthly Charges ($)", min_value=min_m, max_value=max_m, value=(min_m, max_m), step=5.0)

    sel_prob = st.slider("Churn Probability Range", min_value=0.0, max_value=1.0, value=(0.0, 1.0), step=0.05)

# Apply filters
filtered_df = scores_df[
    (scores_df["Contract"].isin(sel_contracts)) &
    (scores_df["Internet_Service"].isin(sel_internets)) &
    (scores_df["Payment_Method"].isin(sel_payments)) &
    (scores_df["Tenure_Months"] >= sel_tenure[0]) &
    (scores_df["Tenure_Months"] <= sel_tenure[1]) &
    (scores_df["Monthly_Charges"] >= sel_monthly[0]) &
    (scores_df["Monthly_Charges"] <= sel_monthly[1]) &
    (scores_df["churn_prob"] >= sel_prob[0]) &
    (scores_df["churn_prob"] <= sel_prob[1])
].copy()

# 2. Reactive KPI Strip
f_count = len(filtered_df)
tot_count = len(scores_df)
pct_of_total = (f_count / tot_count) * 100.0 if tot_count > 0 else 0.0

if f_count > 0:
    avg_risk = float(filtered_df["churn_prob"].mean())
    expected_churners = float(filtered_df["churn_prob"].sum())
    high_risk_filtered = filtered_df[filtered_df["churn_prob"] >= 0.50]
    rev_at_risk = float(high_risk_filtered["Monthly_Charges"].sum())
else:
    avg_risk = 0.0
    expected_churners = 0.0
    rev_at_risk = 0.0

k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi_card(
        title="Filtered Accounts",
        value=f"{f_count:,}",
        delta=f"{pct_of_total:.1f}% of total base",
        delta_type="neutral",
        icon="👥",
        sublabel="Accounts meeting filter criteria",
    )
with k2:
    risk_color = "coral" if avg_risk >= 0.40 else ("amber" if avg_risk >= 0.25 else "mint")
    render_kpi_card(
        title="Average Churn Risk",
        value=f"{avg_risk:.1%}",
        delta="Mean predicted probability",
        delta_type=risk_color,
        icon="📉",
        sublabel="Across filtered cohort",
    )
with k3:
    render_kpi_card(
        title="Expected Churners",
        value=f"{expected_churners:,.0f}",
        delta=f"{(expected_churners / max(f_count, 1)):.1%} expected loss",
        delta_type="coral",
        icon="🚪",
        sublabel="Sum of churn probabilities",
    )
with k4:
    render_kpi_card(
        title="Revenue at Risk (p ≥ 0.5)",
        value=f"${rev_at_risk:,.0f}/mo",
        delta=f"${(rev_at_risk * 12):,.0f} Annualized ARR",
        delta_type="coral",
        icon="💳",
        sublabel=f"{len(high_risk_filtered):,} accounts above 50% risk",
    )

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# 3. Customer Deep Dive Card (Lookup by ID)
st.markdown("### 👤 Individual Subscriber Risk Attribution")
st.caption("Search or select an account ID to visualize their individualized probability gauge and top underlying drivers.")

cust_col1, cust_col2 = st.columns([1, 2.5])

with cust_col1:
    # Customer selectbox with top options
    default_ids = filtered_df["CustomerID"].head(100).tolist() if not filtered_df.empty else scores_df["CustomerID"].head(100).tolist()
    search_id = st.text_input("Enter or search Customer ID", value="", placeholder="e.g. 3668-QPYBK")

    if search_id.strip():
        matched_row = scores_df[scores_df["CustomerID"].str.upper() == search_id.strip().upper()]
    else:
        matched_row = pd.DataFrame()

    if matched_row.empty:
        selected_cust_id = st.selectbox("Or choose from top filtered accounts", options=default_ids)
        target_cust = scores_df[scores_df["CustomerID"] == selected_cust_id].iloc[0]
    else:
        target_cust = matched_row.iloc[0]

    prob_val = float(target_cust["churn_prob"])
    label_val = int(target_cust["label"])

    # Gauge Chart for Probability
    gauge_color = COLOR_CHURN if prob_val >= 0.60 else (COLOR_AMBER if prob_val >= 0.35 else COLOR_SAFE)
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_val * 100.0,
            number=dict(suffix="%", font=dict(family="Sora", size=32, weight="bold", color=gauge_color)),
            title=dict(text=f"<b>Account: {target_cust['CustomerID']}</b><br><span style='font-size:11px;color:#94a3b8;'>Status: {'Already Churned' if label_val == 1 else 'Currently Active'}</span>", font=dict(size=14)),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="white"),
                bar=dict(color=gauge_color, thickness=0.3),
                bgcolor="rgba(22, 31, 48, 0.7)",
                borderwidth=1,
                bordercolor="rgba(255,255,255,0.1)",
                steps=[
                    dict(range=[0, 35], color="rgba(16, 185, 129, 0.15)"),
                    dict(range=[35, 60], color="rgba(245, 158, 11, 0.15)"),
                    dict(range=[60, 100], color="rgba(244, 63, 94, 0.15)"),
                ],
            ),
        )
    )
    fig_gauge.update_layout(
        template=get_plotly_template(),
        height=240,
        margin=dict(t=30, b=10, l=20, r=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with cust_col2:
    if lr_params is not None:
        contributions = get_customer_row_contributions(lr_params, target_cust)
        top_factors = contributions[:6]

        factor_names = [f[0] for f in top_factors][::-1]
        factor_vals = [f[1] for f in top_factors][::-1]
        factor_colors = [COLOR_CHURN if v > 0 else COLOR_SAFE for v in factor_vals]

        fig_cont = go.Figure(
            go.Bar(
                x=factor_vals,
                y=factor_names,
                orientation="h",
                marker_color=factor_colors,
                text=[f"{v:+.3f}" for v in factor_vals],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Impact Weight: %{x:+.3f}<extra></extra>",
            )
        )
        fig_cont.update_layout(
            template=get_plotly_template(),
            title=f"Top Risk Drivers for Customer {target_cust['CustomerID']}",
            xaxis=dict(title="Standardized Log-Odds Impact (Red = Increases Churn, Green = Retains)"),
            yaxis=dict(title=""),
            height=260,
            margin=dict(t=40, b=30, l=140, r=40),
        )
        st.plotly_chart(fig_cont, use_container_width=True)

        st.markdown(
            f"""
            <div style="font-size: 0.8rem; color: #94A3B8; margin-top: -0.5rem;">
                <b>Profile Snapshot</b>: Contract: <code>{target_cust['Contract']}</code> ·
                Internet: <code>{target_cust['Internet_Service']}</code> ·
                Payment: <code>{target_cust['Payment_Method']}</code> ·
                Tenure: <b>{int(target_cust['Tenure_Months'])} mo</b> ·
                Monthly Charge: <b>${target_cust['Monthly_Charges']:.2f}</b> ·
                Total Charges: <b>${target_cust['Total_Charges']:.2f}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# 4. Filtered Customer Table & CSV Export
st.markdown("### 📋 Filtered Subscriber Roster")

tbl_col1, tbl_col2 = st.columns([3, 1])
with tbl_col1:
    top_n_table = st.slider("Display Top N Ranked by Churn Probability", min_value=10, max_value=500, value=50, step=10)
with tbl_col2:
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered CSV",
        data=csv_data,
        file_name="telco_filtered_churn_scores.csv",
        mime="text/csv",
        use_container_width=True,
    )

display_roster = filtered_df.sort_values("churn_prob", ascending=False).head(top_n_table)

st.dataframe(
    display_roster[[
        "CustomerID", "Contract", "Internet_Service", "Payment_Method",
        "Tenure_Months", "Monthly_Charges", "Total_Charges", "churn_prob", "label"
    ]],
    column_config={
        "churn_prob": st.column_config.ProgressColumn(
            "Churn Probability",
            help="Predicted risk of cancellation",
            format="%.2f",
            min_value=0.0,
            max_value=1.0,
        ),
        "Monthly_Charges": st.column_config.NumberColumn(
            "Monthly ($)",
            format="$%.2f",
        ),
        "Total_Charges": st.column_config.NumberColumn(
            "Total Spend ($)",
            format="$%.2f",
        ),
        "Tenure_Months": st.column_config.NumberColumn(
            "Tenure (Mo)",
            format="%d mo",
        ),
        "label": st.column_config.CheckboxColumn(
            "Actual Churn",
            help="Observed historical cancellation",
        ),
    },
    use_container_width=True,
    hide_index=True,
)
