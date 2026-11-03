"""Page 8: Retention ROI Planner.

Financial retention modeling, strategy benchmarking across 7 PySpark models,
two-dimensional parameter sensitivity matrices, and downloadable campaign target cohorts.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils.data import get_model_comparison, get_customer_scores
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
    title="Retention ROI Planner",
    subtitle="Translate machine learning classifications into actionable bottom-line profitability and campaign targeting.",
    badge="Campaign Economics",
    icon="💰",
    how_to_read="""
    **ROI Economics Guide**:
    - **Net Benefit Formula**: $\\text{Profit} = (\\text{True Positives} \\times \\text{Success Rate} \\times \\text{Customer LTV}) - (\\text{Total Offers} \\times \\text{Offer Cost})$.
    - **Strategy Comparison**: Evaluates which PySpark model generates maximum dollar returns subject to marketing budget constraints.
    - **Sensitivity Heatmap**: Identifies the operational break-even frontier across offer costs and customer acceptance rates.
    - **Actionable Target Export**: Allocates budget to the highest-probability subscribers and outputs a downloadable campaign roster.
    """,
)

models_df = get_model_comparison()
scores_df = get_customer_scores()

if models_df is None or models_df.empty:
    st.info("Model comparison data is unavailable.")
    st.stop()

# 1. Financial & Operational Inputs
st.markdown("### 🎛️ Campaign Financial Parameters")

p_col1, p_col2, p_col3, p_col4 = st.columns(4)

with p_col1:
    cust_ltv = st.number_input(
        "Value of Saved Customer ($ LTV)",
        min_value=200.0,
        max_value=3000.0,
        value=1000.0,
        step=50.0,
        help="Expected customer lifetime value preserved by preventing cancellation",
    )
with p_col2:
    offer_cost = st.number_input(
        "Cost per Retention Offer ($)",
        min_value=5.0,
        max_value=250.0,
        value=50.0,
        step=5.0,
        help="Unit cost of discount incentive, hardware credit, or concierge contact",
    )
with p_col3:
    success_rate_pct = st.slider(
        "Offer Success Rate (%)",
        min_value=5,
        max_value=70,
        value=25,
        step=1,
        help="Percentage of targeted churners who accept the offer and remain subscribed",
    )
    success_rate = success_rate_pct / 100.0
with p_col4:
    budget_cap = st.number_input(
        "Retention Budget Cap ($)",
        min_value=5000.0,
        max_value=200000.0,
        value=50000.0,
        step=5000.0,
        help="Maximum total spend allocated for the retention initiative",
    )

st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

# 2. Strategy Comparison Across 7 Models
st.markdown("### 📊 Net Financial Benefit Across Models")
st.caption("Calculates net revenue preserved after deducting offer marketing expenditure.")

max_possible_offers = int(budget_cap // offer_cost)

model_results = []
for _, row in models_df.iterrows():
    m_name = row["model"]
    tp = int(row["TP"])
    fp = int(row["FP"])
    total_unconstrained = tp + fp

    # Apply budget cap scaling if needed
    if total_unconstrained > max_possible_offers and total_unconstrained > 0:
        scale = max_possible_offers / total_unconstrained
        offers_sent = max_possible_offers
        actual_tp = tp * scale
        actual_fp = fp * scale
    else:
        offers_sent = total_unconstrained
        actual_tp = tp
        actual_fp = fp

    churners_saved = actual_tp * success_rate
    rev_saved = churners_saved * cust_ltv
    total_cost = offers_sent * offer_cost
    net_profit = rev_saved - total_cost
    roi = (net_profit / total_cost) * 100.0 if total_cost > 0 else 0.0

    model_results.append({
        "model": m_name,
        "offers_sent": int(offers_sent),
        "actual_tp": actual_tp,
        "churners_saved": churners_saved,
        "cost": total_cost,
        "revenue_saved": rev_saved,
        "net_benefit": net_profit,
        "roi_pct": roi,
    })

df_roi = pd.DataFrame(model_results).sort_values("net_benefit", ascending=False)
best_model_name = df_roi.iloc[0]["model"]
best_net = df_roi.iloc[0]["net_benefit"]

# Ranked Bar Chart of Net Profit
bar_colors_roi = [COLOR_SAFE if b > 0 else COLOR_CHURN for b in df_roi["net_benefit"]]
# Highlight champion
highlight_colors = [COLOR_PRIMARY if m == best_model_name else c for m, c in zip(df_roi["model"], bar_colors_roi)]

fig_roi = go.Figure(
    go.Bar(
        x=df_roi["model"],
        y=df_roi["net_benefit"],
        marker_color=highlight_colors,
        text=df_roi["net_benefit"].apply(lambda v: f"${v:+,.0f}"),
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Net Benefit: $%{y:+,.2f}<br>"
            "Revenue Saved: $%{customdata[0]:,.0f}<br>"
            "Campaign Spend: $%{customdata[1]:,.0f}<br>"
            "Saves: %{customdata[2]:.1f} customers<extra></extra>"
        ),
        customdata=df_roi[["revenue_saved", "cost", "churners_saved"]].values,
    )
)
fig_roi.update_layout(
    template=get_plotly_template(),
    yaxis=dict(title="Net Benefit / Profit ($)", zerolinecolor="white", zerolinewidth=1.5),
    xaxis=dict(title="Classifier Strategy"),
    height=360,
    margin=dict(t=20, b=40, l=40, r=20),
)
st.plotly_chart(fig_roi, use_container_width=True)

st.success(
    f"🏆 **Champion Strategy: {best_model_name}** yields the highest bottom-line net return of **${best_net:+,.0f}** "
    f"({df_roi.iloc[0]['roi_pct']:.1f}% marketing ROI) by maximizing recall to save **{df_roi.iloc[0]['churners_saved']:.1f}** accounts."
)

st.markdown("---")

# 3. Two-Dimensional Sensitivity Matrix
st.markdown("### 🌡️ Sensitivity Matrix: Offer Cost vs Success Rate")
st.caption(f"Evaluate resilience for **{best_model_name}** across varying offer unit costs and acceptance rates.")

selected_model_row = models_df[models_df["model"] == best_model_name].iloc[0]
base_tp = int(selected_model_row["TP"])
base_fp = int(selected_model_row["FP"])

# Grid ranges
cost_grid = [20, 35, 50, 65, 80, 100]
success_grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

sensitivity_matrix = []
for c in cost_grid:
    row_vals = []
    for s in success_grid:
        tot_off = base_tp + base_fp
        eff_off = min(tot_off, int(budget_cap // c))
        eff_tp = base_tp * (eff_off / max(tot_off, 1))
        profit = (eff_tp * s * cust_ltv) - (eff_off * c)
        row_vals.append(profit)
    sensitivity_matrix.append(row_vals)

fig_sens = go.Figure(
    data=go.Heatmap(
        z=sensitivity_matrix,
        x=[f"{int(s*100)}%" for s in success_grid],
        y=[f"${c}" for c in cost_grid],
        colorscale=[[0, COLOR_CHURN], [0.45, "rgba(245, 158, 11, 0.8)"], [1, COLOR_SAFE]],
        text=[[f"${v:+,.0f}" for v in row] for row in sensitivity_matrix],
        texttemplate="%{text}",
        hovertemplate="Success Rate: %{x}<br>Offer Cost: %{y}<br>Net Profit: %{z:+,.0f}<extra></extra>",
        colorbar=dict(title="Net ($)"),
    )
)
fig_sens.update_layout(
    template=get_plotly_template(),
    xaxis=dict(title="Offer Success / Conversion Rate"),
    yaxis=dict(title="Retention Incentive Unit Cost"),
    height=340,
    margin=dict(t=20, b=40, l=40, r=20),
)
st.plotly_chart(fig_sens, use_container_width=True)

st.markdown("---")

# 4. Campaign Targeting (Roster Generation)
st.markdown("### 🎯 Live Campaign Roster Generator")
st.caption("Allocate available marketing budget to the highest-risk subscriber cohort based on individual probabilities.")

if scores_df is not None and not scores_df.empty:
    ranked_targets = scores_df.sort_values("churn_prob", ascending=False).copy()

    # Determine maximum targetable accounts within budget
    budget_capacity = int(budget_cap // offer_cost)
    max_cohort = min(budget_capacity, len(ranked_targets))

    target_n = st.slider(
        "Accounts to Target (Ranked by Risk)",
        min_value=50,
        max_value=max(max_cohort, 100),
        value=min(max_cohort, 800),
        step=50,
        help="Constrained by current budget cap and unit offer cost",
    )

    campaign_cohort = ranked_targets.head(target_n).copy()
    cohort_spend = target_n * offer_cost
    cohort_exp_churners = float(campaign_cohort["churn_prob"].sum())
    cohort_exp_saves = cohort_exp_churners * success_rate
    cohort_rev_saved = cohort_exp_saves * cust_ltv
    cohort_net_benefit = cohort_rev_saved - cohort_spend
    cohort_roi = (cohort_net_benefit / cohort_spend) * 100.0 if cohort_spend > 0 else 0.0

    cr1, cr2, cr3, cr4 = st.columns(4)
    with cr1:
        render_kpi_card(
            title="Target Accounts",
            value=f"{target_n:,}",
            delta=f"${cohort_spend:,.0f} Budget Spent",
            delta_type="neutral",
            icon="🎯",
            sublabel=f"Max budget cap: ${budget_cap:,.0f}",
        )
    with cr2:
        render_kpi_card(
            title="Expected Churners",
            value=f"{cohort_exp_churners:,.0f}",
            delta=f"{(cohort_exp_churners / target_n):.1%} density",
            delta_type="coral",
            icon="🚪",
            sublabel="In targeted campaign pool",
        )
    with cr3:
        render_kpi_card(
            title="Expected Saved Accounts",
            value=f"{cohort_exp_saves:,.1f}",
            delta=f"${cohort_rev_saved:,.0f} LTV Preserved",
            delta_type="mint",
            icon="🛡️",
            sublabel=f"At {success_rate_pct}% success rate",
        )
    with cr4:
        c_type = "mint" if cohort_net_benefit > 0 else "coral"
        render_kpi_card(
            title="Campaign Net Profit",
            value=f"${cohort_net_benefit:+,.0f}",
            delta=f"{cohort_roi:+.1f}% Marketing ROI",
            delta_type=c_type,
            icon="📈",
            sublabel="Net preserved enterprise value",
        )

    # Add actionable playbook column
    campaign_cohort["Recommended_Offer"] = campaign_cohort["Contract"].apply(
        lambda c: "12-Month Lock-in ($10/mo credit)" if c == "Month-to-month" else "Free Premium Tech Support for 6 mo"
    )
    campaign_cohort["Estimated_Value_Saved"] = (campaign_cohort["churn_prob"] * success_rate * cust_ltv).round(2)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    c_dl1, c_dl2 = st.columns([3, 1])
    with c_dl1:
        st.markdown(f"**Target Roster Preview (Top {target_n:,} Accounts)**")
    with c_dl2:
        roster_csv = campaign_cohort[[
            "CustomerID", "Contract", "Internet_Service", "Monthly_Charges",
            "churn_prob", "Recommended_Offer", "Estimated_Value_Saved"
        ]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Campaign Roster CSV",
            data=roster_csv,
            file_name="retention_campaign_target_list.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.dataframe(
        campaign_cohort[[
            "CustomerID", "Contract", "Internet_Service", "Monthly_Charges",
            "churn_prob", "Recommended_Offer", "Estimated_Value_Saved"
        ]].head(25),
        column_config={
            "churn_prob": st.column_config.ProgressColumn(
                "Risk Probability",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
            "Monthly_Charges": st.column_config.NumberColumn(
                "Monthly ($)",
                format="$%.2f",
            ),
            "Estimated_Value_Saved": st.column_config.NumberColumn(
                "Expected Save ($)",
                format="$%.2f",
            ),
        },
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Customer scores dataset is required for campaign targeting.")
