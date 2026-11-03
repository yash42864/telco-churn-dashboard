"""Page 1: Executive Overview / Home.

High-level executive cockpit summarizing subscription churn KPIs,
dynamic data-driven key findings, revenue-at-risk analysis, and quick navigation.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils.data import (
    get_kpis,
    get_churn_by_category,
    get_correlation_matrix,
    get_customer_scores,
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
)

apply_custom_theme()

render_page_header(
    title="Executive Overview",
    subtitle="Strategic portfolio health, retention performance, and revenue-at-risk metrics.",
    badge="Portfolio Health",
    icon="🏠",
    how_to_read="""
    **Executive Guide**:
    - **KPI Cards**: Top metrics represent active subscriber base, baseline churn rate, and revenue averages. Delta reflects deviation against chosen industry benchmarks.
    - **Key Findings**: Dynamically computed directly from Apache Spark EDA outputs, highlighting highest-leverage retention drivers without manual bias.
    - **Revenue at Risk**: Direct monetary impact calculation summing monthly recurring revenue for customers crossing the $p \\ge 0.50$ churn probability threshold.
    """,
)

# Load data artifacts
kpi_df = get_kpis()
cat_df = get_churn_by_category()
corr_df = get_correlation_matrix()
scores_df = get_customer_scores()

if kpi_df is None or kpi_df.empty:
    st.stop()

row_kpi = kpi_df.iloc[0]
total_cust = int(row_kpi["total_customers"])
churned_cust = int(row_kpi["churned"])
churn_rate = float(row_kpi["churn_rate_pct"])
avg_tenure = float(row_kpi["avg_tenure_months"])
avg_monthly = float(row_kpi["avg_monthly_charges"])

# Benchmark Selector
b_col1, b_col2 = st.columns([3, 1])
with b_col2:
    benchmark_choice = st.selectbox(
        "Compare against benchmark",
        options=[
            "Telecom Industry Standard (21.0% churn, $60.00/mo, 36.0 mo)",
            "Internal Q3 Target (22.5% churn, $62.50/mo, 34.0 mo)",
            "Prior Year Baseline (28.2% churn, $65.00/mo, 30.0 mo)",
            "Aggressive Retention Goal (18.0% churn, $68.00/mo, 40.0 mo)",
        ],
        index=0,
    )

# Extract benchmark values
if "21.0%" in benchmark_choice:
    b_churn, b_monthly, b_tenure = 21.0, 60.0, 36.0
elif "22.5%" in benchmark_choice:
    b_churn, b_monthly, b_tenure = 22.5, 62.5, 34.0
elif "28.2%" in benchmark_choice:
    b_churn, b_monthly, b_tenure = 28.2, 65.0, 30.0
else:
    b_churn, b_monthly, b_tenure = 18.0, 68.0, 40.0

delta_churn = churn_rate - b_churn
delta_monthly = avg_monthly - b_monthly
delta_tenure = avg_tenure - b_tenure

# 1. Hero KPI Cards Row
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_kpi_card(
        title="Subscribers",
        value=f"{total_cust:,}",
        delta=f"{churned_cust:,} churned",
        delta_type="neutral",
        icon="👥",
        sublabel="Total evaluated accounts",
    )
with c2:
    churn_delta_type = "coral" if delta_churn > 0 else "mint"
    churn_sign = "+" if delta_churn > 0 else ""
    render_kpi_card(
        title="Churn Rate",
        value=f"{churn_rate:.2f}%",
        delta=f"{churn_sign}{delta_churn:.2f}% vs benchmark",
        delta_type=churn_delta_type,
        icon="📉",
        sublabel=f"Benchmark: {b_churn:.1f}%",
    )
with c3:
    render_kpi_card(
        title="Churned Count",
        value=f"{churned_cust:,}",
        delta=f"{(churned_cust / total_cust):.1%} of base",
        delta_type="coral",
        icon="🚪",
        sublabel="Lost subscriptions",
    )
with c4:
    tenure_delta_type = "mint" if delta_tenure >= 0 else "coral"
    tenure_sign = "+" if delta_tenure > 0 else ""
    render_kpi_card(
        title="Avg Tenure",
        value=f"{avg_tenure:.1f} mo",
        delta=f"{tenure_sign}{delta_tenure:.1f} mo vs benchmark",
        delta_type=tenure_delta_type,
        icon="⏳",
        sublabel=f"Benchmark: {b_tenure:.1f} mo",
    )
with c5:
    monthly_delta_type = "mint" if delta_monthly >= 0 else "coral"
    monthly_sign = "+" if delta_monthly > 0 else ""
    render_kpi_card(
        title="Avg Monthly Charge",
        value=f"${avg_monthly:.2f}",
        delta=f"{monthly_sign}${delta_monthly:.2f} vs benchmark",
        delta_type=monthly_delta_type,
        icon="💳",
        sublabel=f"Benchmark: ${b_monthly:.2f}",
    )

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# 2. Dynamic Data-Driven Key Findings
st.markdown("### 💡 Data-Driven Executive Findings")

# Compute findings dynamically
finding_1_text = "Month-to-month contracts carry high relative cancellation risk."
finding_2_text = "Fiber optic internet exhibits elevated churn despite premium pricing."
finding_3_text = "Customer tenure has strong protective power against cancellation."
finding_4_text = "Value-added support services dramatically suppress churn likelihood."

if cat_df is not None and not cat_df.empty:
    # Contract finding
    contract_m2m = cat_df[(cat_df["feature"] == "Contract") & (cat_df["category"] == "Month-to-month")]
    contract_2yr = cat_df[(cat_df["feature"] == "Contract") & (cat_df["category"] == "Two year")]
    if not contract_m2m.empty and not contract_2yr.empty:
        r_m2m = contract_m2m.iloc[0]["churn_rate_pct"]
        r_2yr = contract_2yr.iloc[0]["churn_rate_pct"]
        ratio = r_m2m / max(r_2yr, 0.1)
        finding_1_text = (
            f"**Contract Structure**: Month-to-month contracts churn at **{r_m2m:.1f}%** vs only "
            f"**{r_2yr:.1f}%** on 2-year contracts (**{ratio:.1f}x relative risk**). Locking in annual commitments is paramount."
        )

    # Fiber optic finding
    fiber = cat_df[(cat_df["feature"] == "Internet_Service") & (cat_df["category"] == "Fiber optic")]
    dsl = cat_df[(cat_df["feature"] == "Internet_Service") & (cat_df["category"] == "DSL")]
    if not fiber.empty and not dsl.empty:
        r_fiber = fiber.iloc[0]["churn_rate_pct"]
        r_dsl = dsl.iloc[0]["churn_rate_pct"]
        ratio_fiber = r_fiber / max(r_dsl, 0.1)
        finding_2_text = (
            f"**Fiber Optic Defection**: Fiber optic subscribers churn at **{r_fiber:.1f}%** compared to "
            f"**{r_dsl:.1f}%** for DSL (**{ratio_fiber:.1f}x higher**), pointing to quality-of-service or billing friction."
        )

    # Support finding
    ts_no = cat_df[(cat_df["feature"] == "Tech_Support") & (cat_df["category"] == "No")]
    ts_yes = cat_df[(cat_df["feature"] == "Tech_Support") & (cat_df["category"] == "Yes")]
    if not ts_no.empty and not ts_yes.empty:
        r_ts_no = ts_no.iloc[0]["churn_rate_pct"]
        r_ts_yes = ts_yes.iloc[0]["churn_rate_pct"]
        ratio_ts = r_ts_no / max(r_ts_yes, 0.1)
        finding_4_text = (
            f"**Tech Support Shield**: Subscribers without Tech Support churn at **{r_ts_no:.1f}%** vs "
            f"**{r_ts_yes:.1f}%** with support (**{ratio_ts:.1f}x difference**). Bundling support reduces risk substantially."
        )

if corr_df is not None and not corr_df.empty:
    # Tenure correlation finding
    tenure_row = corr_df[corr_df["feature"] == "Tenure_Months"]
    if not tenure_row.empty and "label" in tenure_row.columns:
        r_tenure = float(tenure_row.iloc[0]["label"])
        finding_3_text = (
            f"**Tenure Protection**: Correlation between tenure and churn is **r = {r_tenure:.3f}** (the strongest "
            f"negative correlation). The first 12 months constitute the critical retention window."
        )

kf1, kf2, kf3, kf4 = st.columns(4)
with kf1:
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid {COLOR_CHURN}; height: 100%;">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: {COLOR_CHURN}; font-weight: 700;">Finding 1 · Contract</div>
            <div style="font-size: 0.88rem; margin-top: 0.4rem; line-height: 1.45;">{finding_1_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kf2:
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid {COLOR_AMBER}; height: 100%;">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: {COLOR_AMBER}; font-weight: 700;">Finding 2 · Product Tier</div>
            <div style="font-size: 0.88rem; margin-top: 0.4rem; line-height: 1.45;">{finding_2_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kf3:
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid {COLOR_SAFE}; height: 100%;">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: {COLOR_SAFE}; font-weight: 700;">Finding 3 · Lifecycle</div>
            <div style="font-size: 0.88rem; margin-top: 0.4rem; line-height: 1.45;">{finding_3_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kf4:
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid {COLOR_PRIMARY}; height: 100%;">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: {COLOR_PRIMARY}; font-weight: 700;">Finding 4 · Service Add-ons</div>
            <div style="font-size: 0.88rem; margin-top: 0.4rem; line-height: 1.45;">{finding_4_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

# 3. Churn Donut & Revenue at Risk Analysis
row_viz1, row_viz2 = st.columns([1, 1.3])

with row_viz1:
    st.markdown("### 🍩 Churn Composition")
    retained_count = total_cust - churned_cust
    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=["Retained", "Churned"],
                values=[retained_count, churned_cust],
                hole=0.68,
                marker=dict(colors=[COLOR_SAFE, COLOR_CHURN]),
                textinfo="percent+label",
                texttemplate="%{label}<br><b>%{percent:.1%}</b>",
                hoverinfo="label+value+percent",
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent:.2%}<extra></extra>",
            )
        ]
    )
    fig_donut.update_layout(
        template=get_plotly_template(),
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{churn_rate:.1f}%</b><br><span style='font-size:12px;color:#94A3B8;'>Churn Rate</span>",
                x=0.5,
                y=0.5,
                font_size=22,
                showarrow=False,
            )
        ],
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with row_viz2:
    st.markdown("### 💰 Revenue-at-Risk Portfolio Analysis")
    if scores_df is not None and not scores_df.empty:
        high_risk_mask = scores_df["churn_prob"] >= 0.50
        high_risk_cust = scores_df[high_risk_mask]
        high_risk_count = len(high_risk_cust)
        monthly_at_risk = float(high_risk_cust["Monthly_Charges"].sum())
        total_monthly_rev = float(scores_df["Monthly_Charges"].sum())
        annual_at_risk = monthly_at_risk * 12.0
        pct_rev_at_risk = (monthly_at_risk / total_monthly_rev) * 100.0 if total_monthly_rev > 0 else 0.0

        expected_monthly_loss = float((scores_df["churn_prob"] * scores_df["Monthly_Charges"]).sum())
        expected_annual_loss = expected_monthly_loss * 12.0

        r1, r2 = st.columns(2)
        with r1:
            render_kpi_card(
                title="Monthly MRR at Risk (p ≥ 0.5)",
                value=f"${monthly_at_risk:,.0f}",
                delta=f"{pct_rev_at_risk:.1f}% of total portfolio MRR",
                delta_type="coral",
                icon="🚨",
                sublabel=f"{high_risk_count:,} accounts above 50% probability",
            )
        with r2:
            render_kpi_card(
                title="Annual ARR Exposure",
                value=f"${annual_at_risk:,.0f}",
                delta=f"Expected ARR Loss: ${expected_annual_loss:,.0f}",
                delta_type="coral",
                icon="📉",
                sublabel="Calculated over 12 billing cycles",
            )

        # Risk tier distribution bar chart
        scores_df["risk_tier"] = pd.cut(
            scores_df["churn_prob"],
            bins=[0.0, 0.25, 0.50, 0.75, 1.0],
            labels=["Low (<25%)", "Moderate (25-50%)", "High (50-75%)", "Critical (≥75%)"],
            include_lowest=True,
        )
        tier_agg = scores_df.groupby("risk_tier", observed=False).agg(
            customers=("CustomerID", "count"),
            mrr_total=("Monthly_Charges", "sum"),
        ).reset_index()

        fig_tiers = go.Figure(
            data=[
                go.Bar(
                    x=tier_agg["risk_tier"],
                    y=tier_agg["mrr_total"],
                    marker=dict(
                        color=[COLOR_SAFE, COLOR_AMBER, "#FB7185", COLOR_CHURN]
                    ),
                    text=tier_agg["mrr_total"].apply(lambda v: f"${v:,.0f}"),
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>MRR: $%{y:,.2f}<br>Customers: %{customdata:,}<extra></extra>",
                    customdata=tier_agg["customers"],
                )
            ]
        )
        fig_tiers.update_layout(
            template=get_plotly_template(),
            height=190,
            margin=dict(t=10, b=25, l=10, r=10),
            yaxis=dict(title="Monthly MRR ($)", showticklabels=True),
            xaxis=dict(title=""),
        )
        st.plotly_chart(fig_tiers, use_container_width=True)
    else:
        st.info("Customer risk scores are being loaded or missing.")

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# 4. Quick Navigation Hub
st.markdown("### 🧭 Strategic Analytics Modules")
st.caption("Access in-depth exploratory data analysis, machine learning models, customer segmentations, and retention simulators.")

nav_cols = st.columns(4)

with nav_cols[0]:
    st.markdown(
        """
        <div class="glass-card" style="height: 180px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-weight: 700; color: #818CF8; font-size: 1rem;">📊 Churn Drivers (EDA)</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.4rem;">
                    Categorical churn distributions, two-way interaction heatmaps, and correlation matrix.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Explore Drivers ➔", key="nav_eda", use_container_width=True):
        st.switch_page("pages/02_Churn_Drivers.py")

with nav_cols[1]:
    st.markdown(
        """
        <div class="glass-card" style="height: 180px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-weight: 700; color: #818CF8; font-size: 1rem;">🧪 Model Lab</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.4rem;">
                    Compare 7 PySpark classifiers, inspect confusion matrices, PR trade-offs, and feature weights.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Launch Model Lab ➔", key="nav_models", use_container_width=True):
        st.switch_page("pages/03_Model_Lab.py")

with nav_cols[2]:
    st.markdown(
        """
        <div class="glass-card" style="height: 180px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-weight: 700; color: #818CF8; font-size: 1rem;">🎯 Threshold & ROC Studio</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.4rem;">
                    Interactive decision boundary slider, real-time ROC/PR updates, and profit optimization.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Tune Thresholds ➔", key="nav_threshold", use_container_width=True):
        st.switch_page("pages/04_Threshold_Studio.py")

with nav_cols[3]:
    st.markdown(
        """
        <div class="glass-card" style="height: 180px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-weight: 700; color: #818CF8; font-size: 1rem;">👥 Customer Segments</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.4rem;">
                    K-Means clustering profiles, elbow/silhouette diagnostics, and behavioral cohort cards.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("View Segments ➔", key="nav_segments", use_container_width=True):
        st.switch_page("pages/05_Customer_Segments.py")

nav_cols2 = st.columns(4)

with nav_cols2[0]:
    st.markdown(
        """
        <div class="glass-card" style="height: 180px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-weight: 700; color: #818CF8; font-size: 1rem;">🔎 Customer Risk Explorer</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.4rem;">
                    Filterable subscriber roster with risk progress bars, individual ID lookup, and CSV export.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Filter Customers ➔", key="nav_risk", use_container_width=True):
        st.switch_page("pages/06_Customer_Risk_Explorer.py")

with nav_cols2[1]:
    st.markdown(
        """
        <div class="glass-card" style="height: 180px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-weight: 700; color: #818CF8; font-size: 1rem;">🔮 Predict Simulator</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.4rem;">
                    Score live what-if scenarios, inspect factor waterfalls, and simulate contract changes.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Simulate What-If ➔", key="nav_predict", use_container_width=True):
        st.switch_page("pages/07_Predict_Simulator.py")

with nav_cols2[2]:
    st.markdown(
        """
        <div class="glass-card" style="height: 180px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-weight: 700; color: #818CF8; font-size: 1rem;">💰 Retention ROI Planner</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.4rem;">
                    Model campaign economics, test offer costs and success rates, and build targeted cohorts.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Plan Campaign ➔", key="nav_roi", use_container_width=True):
        st.switch_page("pages/08_Retention_ROI_Planner.py")

with nav_cols2[3]:
    st.markdown(
        """
        <div class="glass-card" style="height: 180px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-weight: 700; color: #818CF8; font-size: 1rem;">⚡ Spark Architecture</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.4rem;">
                    PySpark DAG pipeline diagram, distributed execution details, and engineering methodology.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("View Architecture ➔", key="nav_pipeline", use_container_width=True):
        st.switch_page("pages/09_Pipeline_Computing.py")
