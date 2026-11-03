"""Page 5: Customer Segments.

Unsupervised customer cohort analysis using PySpark K-Means clustering.
Displays hyperparameter selection (Elbow and Silhouette), segment macro-profiles,
auto-generated cohort business cards, and individual customer scatter plots.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from utils.data import (
    get_kmeans_selection,
    get_cluster_profiles,
    get_cluster_assignments,
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
    title="Customer Segments (K-Means)",
    subtitle="Behavioral clustering across tenure, monthly billing, and total spend to guide personalized retention.",
    badge="Unsupervised K-Means",
    icon="👥",
    how_to_read="""
    **Segmentation Guide**:
    - **Optimal K (k=4)**: Selected based on the PySpark WSSSE elbow curve and silhouette score peak.
    - **Macro Bubble Chart**: Circles represent cluster scale; vertical and horizontal positioning reveal tenure maturity vs pricing tier.
    - **Cohort Strategic Cards**: Auto-labels clusters into actionable business archetypes with tailored retention playbooks.
    - **Individual Customer Scatter**: 2D projection showing how individual subscribers distribute across cluster boundaries.
    """,
)

k_df = get_kmeans_selection()
profiles_df = get_cluster_profiles()
assign_df = get_cluster_assignments()

if profiles_df is None or profiles_df.empty:
    st.info("K-Means clustering outputs are missing.")
    st.stop()

# 1. Cluster Selection Diagnostics (Elbow & Silhouette)
st.markdown("### 📐 K-Means Model Selection Diagnostics")
st.caption("Evaluation of cluster granularity (k = 2 to 8) using Within-Set Sum of Squared Errors (WSSSE) and Silhouette score.")

if k_df is not None and not k_df.empty:
    diag_c1, diag_c2 = st.columns(2)

    with diag_c1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(
            go.Scatter(
                x=k_df["k"],
                y=k_df["cost"],
                mode="lines+markers",
                marker=dict(size=8, color=COLOR_PRIMARY),
                line=dict(color=COLOR_PRIMARY, width=2.5),
                hovertemplate="k=%{x}<br>WSSSE Cost: %{y:,.1f}<extra></extra>",
            )
        )
        # Highlight k=4
        k4_cost = float(k_df[k_df["k"] == 4]["cost"].iloc[0]) if 4 in k_df["k"].values else None
        if k4_cost is not None:
            fig_elbow.add_trace(
                go.Scatter(
                    x=[4],
                    y=[k4_cost],
                    mode="markers+text",
                    marker=dict(size=14, color=COLOR_CHURN, symbol="star"),
                    text=[" Chosen k=4"],
                    textposition="top right",
                    textfont=dict(color=COLOR_CHURN, weight="bold"),
                    name="Selected k=4",
                )
            )
        fig_elbow.update_layout(
            template=get_plotly_template(),
            title="Elbow Method (Cost / WSSSE)",
            xaxis=dict(title="Number of Clusters (k)", tickmode="linear", dtick=1),
            yaxis=dict(title="WSSSE Objective Cost"),
            height=280,
            showlegend=False,
            margin=dict(t=40, b=30, l=40, r=20),
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with diag_c2:
        fig_sil = go.Figure()
        fig_sil.add_trace(
            go.Scatter(
                x=k_df["k"],
                y=k_df["silhouette"],
                mode="lines+markers",
                marker=dict(size=8, color=COLOR_SAFE),
                line=dict(color=COLOR_SAFE, width=2.5),
                hovertemplate="k=%{x}<br>Silhouette: %{y:.4f}<extra></extra>",
            )
        )
        # Highlight k=4
        k4_sil = float(k_df[k_df["k"] == 4]["silhouette"].iloc[0]) if 4 in k_df["k"].values else None
        if k4_sil is not None:
            fig_sil.add_trace(
                go.Scatter(
                    x=[4],
                    y=[k4_sil],
                    mode="markers+text",
                    marker=dict(size=14, color=COLOR_CHURN, symbol="star"),
                    text=[" Chosen k=4"],
                    textposition="bottom left",
                    textfont=dict(color=COLOR_CHURN, weight="bold"),
                    name="Selected k=4",
                )
            )
        fig_sil.update_layout(
            template=get_plotly_template(),
            title="Silhouette Coefficient",
            xaxis=dict(title="Number of Clusters (k)", tickmode="linear", dtick=1),
            yaxis=dict(title="Mean Silhouette Score"),
            height=280,
            showlegend=False,
            margin=dict(t=40, b=30, l=40, r=20),
        )
        st.plotly_chart(fig_sil, use_container_width=True)

st.markdown("---")

# 2. Bubble Chart of Segment Macro-Profiles
st.markdown("### 🪐 Segment Portfolio Landscape")
st.caption("Bubble location reflects tenure vs monthly charge; bubble size reflects subscriber volume; color indicates churn rate.")

fig_bubble = px.scatter(
    profiles_df,
    x="avg_tenure",
    y="avg_monthly",
    size="customers",
    color="churn_rate_pct",
    text="cluster",
    color_continuous_scale=[[0.0, COLOR_SAFE], [0.5, "#FBBF24"], [1.0, COLOR_CHURN]],
    size_max=55,
    hover_data={
        "cluster": True,
        "customers": ":,",
        "avg_tenure": ":.1f",
        "avg_monthly": ":.2f",
        "avg_total": ":,.0f",
        "churn_rate_pct": ":.2f",
    },
)

fig_bubble.update_traces(
    textposition="middle center",
    textfont=dict(color="white", size=13, weight="bold"),
    marker=dict(line=dict(width=2, color="white")),
)

fig_bubble.update_layout(
    template=get_plotly_template(),
    xaxis=dict(title="Average Tenure (Months)", range=[0, 70]),
    yaxis=dict(title="Average Monthly Charge ($)", range=[30, 110]),
    coloraxis_colorbar=dict(title="Churn Rate %"),
    height=400,
    margin=dict(t=20, b=40, l=40, r=20),
)
st.plotly_chart(fig_bubble, use_container_width=True)

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# 3. Auto-Generated Segment Business Cards
st.markdown("### 🏷️ Strategic Cohort Archetypes & Action Playbooks")
st.caption("Algorithmic categorization based on tenure lifecycle, expenditure level, and observed attrition.")

def derive_cluster_meta(row: pd.Series):
    cid = int(row["cluster"])
    tenure = float(row["avg_tenure"])
    monthly = float(row["avg_monthly"])
    churn = float(row["churn_rate_pct"])

    if churn >= 40.0:
        label = "🚨 High-Risk Flight Risk"
        badge = "Urgent Retention"
        badge_cls = "badge-danger"
        action = "Immediate proactive outreach; offer introductory 12-month contract lock-in with free Tech Support upgrade."
    elif tenure >= 45.0 and monthly >= 75.0:
        label = "💎 High-Value Champions"
        badge = "VIP Loyalty"
        badge_cls = "badge-primary"
        action = "Executive loyalty concierge, early access to fiber speed boosts, device upgrades, and multi-year renewals."
    elif churn <= 12.0:
        label = "🛡️ Steady Loyalists"
        badge = "Low Maintenance"
        badge_cls = "badge-success"
        action = "Low churn risk; focus on organic expansion, automated renewal billing, and digital customer self-service."
    else:
        label = "🌱 Budget Growth Segment"
        badge = "Cross-Sell Target"
        badge_cls = "badge-amber"
        action = "Upsell streaming bundles and security add-ons; provide annual discount incentive to solidify commitment."

    return cid, label, badge, badge_cls, action

c_cards = st.columns(len(profiles_df))

for idx, (_, row) in enumerate(profiles_df.iterrows()):
    cid, label, badge, badge_cls, action = derive_cluster_meta(row)
    with c_cards[idx]:
        st.markdown(
            f"""
            <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-weight: 800; font-size: 1.1rem; color: #818CF8;">Cluster {cid}</span>
                        <span class="badge-pill {badge_cls}">{badge}</span>
                    </div>
                    <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 0.6rem;">{label}</div>
                    <div style="font-size: 0.82rem; line-height: 1.5; color: #94A3B8;">
                        • <b>Subscribers</b>: {int(row['customers']):,} accounts<br>
                        • <b>Avg Tenure</b>: {row['avg_tenure']:.1f} months<br>
                        • <b>Avg Spend</b>: ${row['avg_monthly']:.2f}/mo<br>
                        • <b>Avg LTV</b>: ${row['avg_total']:,.0f}<br>
                        • <b>Churn Rate</b>: <span style="color:{COLOR_CHURN if row['churn_rate_pct']>25 else COLOR_SAFE}; font-weight:700;">{row['churn_rate_pct']:.1f}%</span>
                    </div>
                </div>
                <div style="margin-top: 0.8rem; padding-top: 0.6rem; border-top: 1px solid rgba(255,255,255,0.08); font-size: 0.78rem; line-height: 1.4;">
                    <b>Retention Playbook:</b><br>{action}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# 4. Individual Customer 2D Scatter Plot
st.markdown("### 🔬 Micro-Level Customer Scatter (Tenure vs Monthly Charges)")
st.caption("Inspect individual customer data points mapped to clusters. Use sampling slider for optimal rendering performance.")

if assign_df is not None and not assign_df.empty:
    sc_col1, sc_col2 = st.columns([1, 1])

    with sc_col1:
        sample_size = st.slider(
            "Customer Sample Size for Performance",
            min_value=500,
            max_value=len(assign_df),
            value=min(2500, len(assign_df)),
            step=500,
        )
    with sc_col2:
        churn_only_toggle = st.toggle("Filter: Show Churned Customers Only (label = 1)", value=False)

    df_plot = assign_df.copy()
    if churn_only_toggle:
        df_plot = df_plot[df_plot["label"] == 1.0]

    if len(df_plot) > sample_size:
        df_sample = df_plot.sample(n=sample_size, random_state=42)
    else:
        df_sample = df_plot

    df_sample["Cluster_Label"] = df_sample["cluster"].apply(lambda c: f"Cluster {c}")

    cluster_color_map = {
        "Cluster 0": COLOR_CHURN,
        "Cluster 1": COLOR_AMBER,
        "Cluster 2": COLOR_SAFE,
        "Cluster 3": COLOR_PRIMARY,
    }

    fig_indiv = px.scatter(
        df_sample,
        x="Tenure_Months",
        y="Monthly_Charges",
        color="Cluster_Label",
        color_discrete_map=cluster_color_map,
        opacity=0.55,
        hover_data={"CustomerID": True, "Tenure_Months": ":.0f", "Monthly_Charges": ":.2f", "label": True},
    )

    # Overlay cluster centroids from profiles
    for _, prow in profiles_df.iterrows():
        cid = int(prow["cluster"])
        fig_indiv.add_trace(
            go.Scatter(
                x=[prow["avg_tenure"]],
                y=[prow["avg_monthly"]],
                mode="markers+text",
                marker=dict(size=16, color="white", symbol="cross", line=dict(width=2, color="black")),
                text=[f"  C{cid} Centroid"],
                textposition="top center",
                textfont=dict(color="white", size=11, weight="bold"),
                name=f"Centroid C{cid}",
                showlegend=False,
            )
        )

    fig_indiv.update_layout(
        template=get_plotly_template(),
        xaxis=dict(title="Tenure (Months)", range=[0, 75]),
        yaxis=dict(title="Monthly Charges ($)", range=[15, 125]),
        height=420,
        margin=dict(t=20, b=40, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_indiv, use_container_width=True)
else:
    st.info("Cluster assignments dataset is unavailable.")
