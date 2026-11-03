"""Predicting Subscription Cancellation (Churn) with PySpark.

Main multipage Streamlit entry point.
Configures global theme, responsive styling, persistent sidebar controls, and navigation.
"""

import streamlit as st
from utils.theme import apply_custom_theme, render_sidebar_footer

# Configure page metadata and layout
st.set_page_config(
    page_title="Telco Churn Analytics | PySpark",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global styling injection
apply_custom_theme()

# Sidebar persistent controls (theme toggle, dataset details, cache refresh)
render_sidebar_footer()

# Multipage Navigation Setup
pages = [
    st.Page("pages/01_Executive_Overview.py", title="Executive Overview", icon="🏠", default=True),
    st.Page("pages/02_Churn_Drivers.py", title="Churn Drivers (EDA)", icon="📊"),
    st.Page("pages/03_Model_Lab.py", title="Model Lab", icon="🧪"),
    st.Page("pages/04_Threshold_Studio.py", title="Threshold & ROC Studio", icon="🎯"),
    st.Page("pages/05_Customer_Segments.py", title="Customer Segments", icon="👥"),
    st.Page("pages/06_Customer_Risk_Explorer.py", title="Customer Risk Explorer", icon="🔎"),
    st.Page("pages/07_Predict_Simulator.py", title="Predict Simulator", icon="🔮"),
    st.Page("pages/08_Retention_ROI_Planner.py", title="Retention ROI Planner", icon="💰"),
    st.Page("pages/09_Pipeline_Computing.py", title="Pipeline Architecture", icon="⚡"),
]

pg = st.navigation(pages)
pg.run()
