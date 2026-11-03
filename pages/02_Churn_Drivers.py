"""Page 2: Churn Drivers (Exploratory Data Analysis).

Explores categorical feature churn rates, two-feature intersection heatmaps,
overall Pearson correlation matrix, and distribution curves for continuous attributes.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from utils.data import (
    get_churn_by_category,
    get_correlation_matrix,
    get_customer_scores,
    get_kpis,
)
from utils.theme import (
    apply_custom_theme,
    render_page_header,
    get_plotly_template,
    COLOR_CHURN,
    COLOR_SAFE,
    COLOR_PRIMARY,
    COLOR_AMBER,
)

apply_custom_theme()

render_page_header(
    title="Churn Drivers (EDA)",
    subtitle="Identify the structural, behavioral, and contractual drivers influencing subscriber attrition.",
    badge="Exploratory Analysis",
    icon="📊",
    how_to_read="""
    **How to Interpret EDA**:
    - **Categorical Risk**: The dashed reference line marks the portfolio baseline churn rate (26.54%). Categories rising above this line are statistical risk multipliers.
    - **Intersection Heatmap**: Cross-tabulates two strategic features to uncover compounding risk pockets (e.g., month-to-month contracts coupled with fiber optic service).
    - **Correlation Matrix**: Displays linear relationships across binary and numerical variables. Negative correlation with churn denotes protective attributes.
    - **Distribution Explorer**: Visualizes how early tenure and premium monthly price points correlate with customer departure.
    """,
)

cat_df = get_churn_by_category()
corr_df = get_correlation_matrix()
scores_df = get_customer_scores()
kpi_df = get_kpis()

baseline_churn = float(kpi_df.iloc[0]["churn_rate_pct"]) if kpi_df is not None and not kpi_df.empty else 26.54

# Section 1: Single Feature Categorical Churn Explorer
st.markdown("### 🔍 Categorical Feature Risk Profiles")

if cat_df is not None and not cat_df.empty:
    all_features = list(cat_df["feature"].unique())

    c_sel1, c_sel2 = st.columns([2, 1])
    with c_sel1:
        selected_feature = st.selectbox(
            "Select Categorical Feature to Inspect",
            options=all_features,
            index=0,
            key="eda_feature_select",
        )
    with c_sel2:
        sort_by_churn = st.checkbox("Sort bars by churn rate descending", value=True)

    feat_data = cat_df[cat_df["feature"] == selected_feature].copy()
    if sort_by_churn:
        feat_data = feat_data.sort_values("churn_rate_pct", ascending=False)

    # Bar chart with overall average line
    fig_bars = go.Figure()

    # Dynamic colors: above average = Coral, below average = Mint
    bar_colors = [
        COLOR_CHURN if r >= baseline_churn else COLOR_SAFE
        for r in feat_data["churn_rate_pct"]
    ]

    fig_bars.add_trace(
        go.Bar(
            x=feat_data["category"],
            y=feat_data["churn_rate_pct"],
            marker_color=bar_colors,
            text=feat_data.apply(
                lambda row: f"<b>{row['churn_rate_pct']:.1f}%</b><br>({row['customers']:,} cust)",
                axis=1,
            ),
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Churn Rate: %{y:.2f}%<br>"
                "Subscribers: %{customdata:,}<br>"
                "<extra></extra>"
            ),
            customdata=feat_data["customers"],
        )
    )

    # Overall benchmark line
    fig_bars.add_shape(
        type="line",
        x0=-0.5,
        x1=len(feat_data) - 0.5,
        y0=baseline_churn,
        y1=baseline_churn,
        line=dict(color=COLOR_AMBER, width=2, dash="dash"),
    )

    fig_bars.add_annotation(
        x=len(feat_data) - 0.5,
        y=baseline_churn + 1.5,
        text=f"Portfolio Baseline ({baseline_churn:.1f}%)",
        showarrow=False,
        font=dict(color=COLOR_AMBER, size=11, weight="bold"),
        xanchor="right",
    )

    fig_bars.update_layout(
        template=get_plotly_template(),
        yaxis=dict(
            title="Churn Rate (%)",
            range=[0, max(feat_data["churn_rate_pct"].max() * 1.25, 45)],
        ),
        xaxis=dict(title="Category Option"),
        height=380,
        margin=dict(t=30, b=40, l=40, r=40),
    )
    st.plotly_chart(fig_bars, use_container_width=True)
else:
    st.info("Categorical churn data not loaded.")

st.markdown("---")

# Section 2: Two-Feature Interaction Heatmap & Correlation Matrix
col_hm1, col_hm2 = st.columns([1.1, 1.1])

with col_hm1:
    st.markdown("### 🔀 Two-Feature Compounding Risk Heatmap")
    st.caption("Identify intersecting friction zones across multiple attributes.")

    if scores_df is not None and not scores_df.empty:
        available_cat_cols = [c for c in ["Contract", "Internet_Service", "Payment_Method"] if c in scores_df.columns]

        if len(available_cat_cols) >= 2:
            sc1, sc2 = st.columns(2)
            with sc1:
                feat_x = st.selectbox("Horizontal Axis (Feature X)", available_cat_cols, index=0)
            with sc2:
                feat_y = st.selectbox(
                    "Vertical Axis (Feature Y)",
                    available_cat_cols,
                    index=1 if len(available_cat_cols) > 1 else 0,
                )

            if feat_x == feat_y:
                st.warning("Please select two distinct features for the interaction heatmap.")
            else:
                # Pivot table of mean churn rate and count
                pivot_churn = scores_df.pivot_table(
                    index=feat_y,
                    columns=feat_x,
                    values="label",
                    aggfunc="mean",
                ) * 100.0

                pivot_count = scores_df.pivot_table(
                    index=feat_y,
                    columns=feat_x,
                    values="label",
                    aggfunc="count",
                )

                # Format annotations
                text_annotations = []
                for y_idx in pivot_churn.index:
                    row_txt = []
                    for x_idx in pivot_churn.columns:
                        val = pivot_churn.loc[y_idx, x_idx]
                        cnt = pivot_count.loc[y_idx, x_idx]
                        row_txt.append(f"<b>{val:.1f}%</b><br><span style='font-size:10px;'>({cnt:,})</span>")
                    text_annotations.append(row_txt)

                fig_2feat = go.Figure(
                    data=go.Heatmap(
                        z=pivot_churn.values,
                        x=pivot_churn.columns.tolist(),
                        y=pivot_churn.index.tolist(),
                        colorscale=[[0.0, COLOR_SAFE], [0.5, "#FBBF24"], [1.0, COLOR_CHURN]],
                        text=text_annotations,
                        texttemplate="%{text}",
                        hovertemplate=(
                            f"<b>{feat_y}</b>: %{{y}}<br>"
                            f"<b>{feat_x}</b>: %{{x}}<br>"
                            "Churn Rate: %{z:.2f}%<extra></extra>"
                        ),
                        colorbar=dict(title="Churn %"),
                    )
                )
                fig_2feat.update_layout(
                    template=get_plotly_template(),
                    height=360,
                    margin=dict(t=20, b=30, l=40, r=20),
                )
                st.plotly_chart(fig_2feat, use_container_width=True)
        else:
            st.info("Insufficient categorical features available in scores dataset.")
    else:
        st.info("Scores dataset is required to compute dynamic multi-feature intersections.")

with col_hm2:
    st.markdown("### 📈 Pearson Correlation Matrix")
    st.caption("Evaluates pairwise linear associations across binary and numerical variables.")

    if corr_df is not None and not corr_df.empty:
        # Prepare matrix
        df_corr = corr_df.copy()
        if "feature" in df_corr.columns:
            feat_names = df_corr["feature"].tolist()
            num_cols = [c for c in df_corr.columns if c != "feature"]
            z_vals = df_corr[num_cols].values
        else:
            feat_names = df_corr.index.tolist()
            num_cols = df_corr.columns.tolist()
            z_vals = df_corr.values

        corr_view_toggle = st.toggle("Rank only correlations with Churn (label)", value=False)

        if corr_view_toggle and "label" in num_cols:
            # Rank bar chart with label
            label_col_idx = num_cols.index("label")
            churn_corrs = []
            for i, name in enumerate(feat_names):
                if name != "label":
                    val = float(z_vals[i, label_col_idx])
                    churn_corrs.append({"feature": name, "correlation": val})

            df_churn_corr = pd.DataFrame(churn_corrs).sort_values("correlation", ascending=True)

            bar_colors_corr = [
                COLOR_CHURN if v > 0 else COLOR_SAFE for v in df_churn_corr["correlation"]
            ]

            fig_rank = go.Figure(
                go.Bar(
                    x=df_churn_corr["correlation"],
                    y=df_churn_corr["feature"],
                    orientation="h",
                    marker_color=bar_colors_corr,
                    text=df_churn_corr["correlation"].apply(lambda v: f"{v:+.3f}"),
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Correlation with Churn: %{x:+.3f}<extra></extra>",
                )
            )
            fig_rank.update_layout(
                template=get_plotly_template(),
                xaxis=dict(title="Pearson Correlation (r)", range=[-0.45, 0.35]),
                yaxis=dict(title=""),
                height=360,
                margin=dict(t=20, b=30, l=40, r=20),
            )
            st.plotly_chart(fig_rank, use_container_width=True)
        else:
            fig_corr = go.Figure(
                data=go.Heatmap(
                    z=z_vals,
                    x=num_cols,
                    y=feat_names,
                    colorscale="RdBu_r",
                    zmin=-0.8,
                    zmax=0.8,
                    text=np.round(z_vals, 2),
                    texttemplate="%{text}",
                    hovertemplate="Row: %{y}<br>Col: %{x}<br>r: %{z:.3f}<extra></extra>",
                    colorbar=dict(title="r"),
                )
            )
            fig_corr.update_layout(
                template=get_plotly_template(),
                height=360,
                margin=dict(t=20, b=30, l=40, r=20),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Correlation matrix dataset is missing.")

st.markdown("---")

# Section 3: Distribution Analysis Split by Churn
st.markdown("### 🎻 Continuous Feature Distributions Split by Churn")
st.caption("Assess how customer tenure duration and monthly billing levels differ between loyal vs churned cohorts.")

if scores_df is not None and not scores_df.empty:
    dist_c1, dist_c2 = st.columns([1, 1])

    with dist_c1:
        dist_feature = st.radio(
            "Select Numerical Attribute",
            options=["Tenure_Months", "Monthly_Charges"],
            format_func=lambda x: "Tenure (Months)" if x == "Tenure_Months" else "Monthly Charges ($)",
            horizontal=True,
        )
    with dist_c2:
        dist_plot_type = st.radio(
            "Visualization Type",
            options=["Violin Plot", "Histogram / Density"],
            horizontal=True,
        )

    # Prepare labels for visualization
    scores_df["Churn_Status"] = scores_df["label"].apply(lambda v: "Churned" if v == 1.0 else "Retained")

    if dist_plot_type == "Violin Plot":
        fig_dist = go.Figure()
        for status, color in [("Retained", COLOR_SAFE), ("Churned", COLOR_CHURN)]:
            sub = scores_df[scores_df["Churn_Status"] == status]
            fig_dist.add_trace(
                go.Violin(
                    x=sub["Churn_Status"],
                    y=sub[dist_feature],
                    name=status,
                    box_visible=True,
                    meanline_visible=True,
                    line_color=color,
                    fillcolor=color,
                    opacity=0.6,
                )
            )
        fig_dist.update_layout(
            template=get_plotly_template(),
            yaxis=dict(title=dist_feature.replace("_", " ")),
            height=380,
            showlegend=False,
            margin=dict(t=30, b=40, l=40, r=20),
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        fig_dist = px.histogram(
            scores_df,
            x=dist_feature,
            color="Churn_Status",
            barmode="overlay",
            opacity=0.65,
            color_discrete_map={"Retained": COLOR_SAFE, "Churned": COLOR_CHURN},
            labels={"Churn_Status": "Cohort", dist_feature: dist_feature.replace("_", " ")},
        )
        fig_dist.update_layout(
            template=get_plotly_template(),
            yaxis=dict(title="Number of Customers"),
            height=380,
            margin=dict(t=30, b=40, l=40, r=20),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Narrative takeaway
    if dist_feature == "Tenure_Months":
        st.info(
            "📌 **Tenure Takeaway**: The churned cohort is heavily concentrated in months 1–12 (a severe early "
            "attrition spike). Conversely, retained customers exhibit a bimodal surge towards long-term longevity (50–72 months)."
        )
    else:
        st.info(
            "📌 **Monthly Charge Takeaway**: Churned subscribers concentrate in the higher price bands ($70–$105/month), "
            "principally driven by Fiber Optic bundles without bundled retention discounts."
        )
else:
    st.info("Scores dataset required for distribution analysis.")
