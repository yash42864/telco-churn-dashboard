"""Data loading and caching utilities for Churn Prediction Dashboard.

Reads processed output files exported from the PySpark training pipeline.
Handles missing files gracefully with actionable script instructions.
"""

import json
import os
from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

# Locate the outputs folder: checks dashboard/outputs first (for standalone/container deployment),
# then falls back to ../outputs relative to dashboard (churn-project/outputs)
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_OUTPUTS = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "outputs"))
_PROJECT_OUTPUTS = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "..", "outputs"))

if os.path.exists(_LOCAL_OUTPUTS):
    OUTPUTS_DIR = _LOCAL_OUTPUTS
else:
    OUTPUTS_DIR = _PROJECT_OUTPUTS

# Mapping of output files to the PySpark script responsible for generating them
SCRIPT_MAP: Dict[str, str] = {
    "kpis.csv": "scripts/03_eda.py",
    "churn_by_category.csv": "scripts/03_eda.py",
    "correlation_matrix.csv": "scripts/03_eda.py",
    "model_comparison.csv": "scripts/05_train_classifiers.py",
    "feature_importance.csv": "scripts/05_train_classifiers.py",
    "regression_metrics.csv": "scripts/06_regression.py",
    "kmeans_selection.csv": "scripts/07_kmeans.py",
    "cluster_profiles.csv": "scripts/07_kmeans.py",
    "cluster_assignments.csv": "scripts/07_kmeans.py",
    "customer_scores.csv": "scripts/08_export.py",
    "lr_params.json": "scripts/08_export.py",
    "feature_names.json": "scripts/04_features.py",
    "churn_by_category.png": "scripts/03_eda.py",
    "correlation_heatmap.png": "scripts/03_eda.py",
    "numeric_distributions.png": "scripts/03_eda.py",
    "model_comparison.png": "scripts/05_train_classifiers.py",
    "confusion_matrices.png": "scripts/05_train_classifiers.py",
    "regression_actual_vs_pred.png": "scripts/06_regression.py",
    "kmeans_elbow_silhouette.png": "scripts/07_kmeans.py",
}


def get_file_path(filename: str) -> str:
    """Return the absolute path to a file in the outputs directory."""
    return os.path.join(OUTPUTS_DIR, filename)


def warn_missing_file(filename: str) -> None:
    """Display an informative, styled warning in Streamlit when a file is missing."""
    script = SCRIPT_MAP.get(filename, "the appropriate pipeline script")
    st.warning(
        f"⚠️ **Missing Data Artifact**: `{filename}` is not present in `outputs/`.\n\n"
        f"To generate this file, please execute:\n"
        f"```bash\npython {script}\n```\n"
        f"Once complete, click **Reload Data** in the sidebar to refresh."
    )


@st.cache_data(show_spinner=False)
def load_csv(filename: str) -> Optional[pd.DataFrame]:
    """Safely load a CSV file from outputs/ with caching and error handling."""
    path = get_file_path(filename)
    if not os.path.exists(path):
        warn_missing_file(filename)
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error reading `{filename}`: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_json(filename: str) -> Optional[Dict[str, Any]]:
    """Safely load a JSON file from outputs/ with caching and error handling."""
    path = get_file_path(filename)
    if not os.path.exists(path):
        warn_missing_file(filename)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error reading `{filename}`: {e}")
        return None


# Specialized typed loaders
def get_kpis() -> Optional[pd.DataFrame]:
    return load_csv("kpis.csv")


def get_churn_by_category() -> Optional[pd.DataFrame]:
    return load_csv("churn_by_category.csv")


def get_correlation_matrix() -> Optional[pd.DataFrame]:
    df = load_csv("correlation_matrix.csv")
    if df is not None and not df.empty:
        # The first column is the row index (e.g. Unnamed: 0)
        first_col = df.columns[0]
        if first_col.startswith("Unnamed") or first_col == "":
            df = df.rename(columns={first_col: "feature"})
    return df


def get_model_comparison() -> Optional[pd.DataFrame]:
    return load_csv("model_comparison.csv")


def get_feature_importance() -> Optional[pd.DataFrame]:
    df = load_csv("feature_importance.csv")
    if df is not None and not df.empty:
        first_col = df.columns[0]
        if first_col.startswith("Unnamed") or first_col == "":
            df = df.rename(columns={first_col: "feature"})
    return df


def get_regression_metrics() -> Optional[pd.DataFrame]:
    return load_csv("regression_metrics.csv")


def get_kmeans_selection() -> Optional[pd.DataFrame]:
    return load_csv("kmeans_selection.csv")


def get_cluster_profiles() -> Optional[pd.DataFrame]:
    return load_csv("cluster_profiles.csv")


def get_cluster_assignments() -> Optional[pd.DataFrame]:
    return load_csv("cluster_assignments.csv")


def get_customer_scores() -> Optional[pd.DataFrame]:
    return load_csv("customer_scores.csv")


def get_lr_params() -> Optional[Dict[str, Any]]:
    return load_json("lr_params.json")


def get_image_path(filename: str) -> Optional[str]:
    """Return image path if exists, otherwise display warning."""
    path = get_file_path(filename)
    if os.path.exists(path):
        return path
    warn_missing_file(filename)
    return None


def clear_all_caches() -> None:
    """Clear Streamlit cache data."""
    st.cache_data.clear()
