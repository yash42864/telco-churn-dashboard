"""Machine learning scoring, attribution, and threshold evaluation utilities.

Replicates the PySpark LogisticRegressionModel scoring pipeline offline without Spark.
Computes ROC, PR curves, threshold sweeps, and feature contributions in pure NumPy.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def score_customer_profile(
    params: Dict[str, Any],
    senior: int,
    tenure: float,
    monthly: float,
    total: float,
    categories: Dict[str, str],
) -> Tuple[float, float, List[Tuple[str, float, str]]]:
    """Score a customer profile using the class-weighted Logistic Regression parameters.

    Returns:
        prob: Predicted churn probability (0.0 to 1.0)
        logit: Raw log-odds (dot product + intercept)
        contributions: List of (feature_label, contribution_value, direction)
                       where direction is 'risk_up' or 'risk_down'.
    """
    feature_names = params["feature_names"]
    std = params["std"]
    coef = np.array(params["coef"])
    intercept = float(params["intercept"])

    x = np.zeros(len(feature_names))

    # Standardized continuous features (first 4: Senior_Citizen, Tenure_Months, Monthly_Charges, Total_Charges)
    raw_vals = [float(senior), float(tenure), float(monthly), float(total)]
    for i in range(4):
        x[i] = raw_vals[i] / std[i]

    # One-hot encoded categorical features
    for feat, val in categories.items():
        key = f"{feat}={val}"
        if key in feature_names:
            x[feature_names.index(key)] = 1.0

    # Logit and sigmoid
    logit = float(np.dot(coef, x) + intercept)
    prob = 1.0 / (1.0 + np.exp(-logit))

    # Calculate individual factor contributions
    contributions = []
    raw_labels = [
        f"Senior Citizen ({'Yes' if senior == 1 else 'No'})",
        f"Tenure ({int(tenure)} months)",
        f"Monthly Charges (${monthly:.2f})",
        f"Total Charges (${total:.2f})",
    ]
    for i in range(4):
        val = float(coef[i] * x[i])
        direction = "risk_up" if val > 0 else "risk_down"
        contributions.append((raw_labels[i], val, direction))

    for feat, val in categories.items():
        key = f"{feat}={val}"
        if key in feature_names:
            idx = feature_names.index(key)
            c_val = float(coef[idx] * x[idx])
            direction = "risk_up" if c_val > 0 else "risk_down"
            contributions.append((f"{feat}: {val}", c_val, direction))

    # Sort contributions by absolute magnitude descending
    contributions.sort(key=lambda t: abs(t[1]), reverse=True)

    return prob, logit, contributions


def get_customer_row_contributions(
    params: Dict[str, Any],
    row: pd.Series,
) -> List[Tuple[str, float, str]]:
    """Compute risk driver contributions for a specific customer from customer_scores.csv."""
    feature_names = params["feature_names"]
    std = params["std"]
    coef = params["coef"]
    feat_map = dict(zip(feature_names, coef))

    contributions = []

    # 1. Tenure
    tenure = float(row["Tenure_Months"])
    val_tenure = feat_map.get("Tenure_Months", 0.0) * (tenure / std[1])
    contributions.append((
        f"Tenure ({int(tenure)} mo)",
        val_tenure,
        "risk_up" if val_tenure > 0 else "risk_down",
    ))

    # 2. Monthly Charges
    monthly = float(row["Monthly_Charges"])
    val_monthly = feat_map.get("Monthly_Charges", 0.0) * (monthly / std[2])
    contributions.append((
        f"Monthly Charges (${monthly:.2f})",
        val_monthly,
        "risk_up" if val_monthly > 0 else "risk_down",
    ))

    # 3. Total Charges
    total = float(row["Total_Charges"])
    val_total = feat_map.get("Total_Charges", 0.0) * (total / std[3])
    contributions.append((
        f"Total Charges (${total:.2f})",
        val_total,
        "risk_up" if val_total > 0 else "risk_down",
    ))

    # 4. Known categorical features in customer_scores.csv
    for cat_col in ["Contract", "Internet_Service", "Payment_Method"]:
        if cat_col in row and pd.notna(row[cat_col]):
            val = str(row[cat_col])
            key = f"{cat_col}={val}"
            if key in feat_map:
                c_val = feat_map[key]
                contributions.append((
                    f"{cat_col}: {val}",
                    c_val,
                    "risk_up" if c_val > 0 else "risk_down",
                ))

    # Sort by absolute impact
    contributions.sort(key=lambda t: abs(t[1]), reverse=True)
    return contributions


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute classification metrics for a specific decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = len(y_true)
    total_pos = tp + fn
    total_neg = tn + fp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "fpr": fpr,
        "total_churners": total_pos,
        "total_retained": total_neg,
    }


def compute_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_thresholds: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute vectorized ROC curve (FPR, TPR, thresholds, AUC)."""
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    tprs = []
    fprs = []

    pos_count = np.sum(y_true == 1)
    neg_count = np.sum(y_true == 0)

    for t in sorted(thresholds, reverse=True):
        pred = (y_prob >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        tprs.append(tp / pos_count if pos_count > 0 else 0.0)
        fprs.append(fp / neg_count if neg_count > 0 else 0.0)

    tprs = np.array(tprs)
    fprs = np.array(fprs)

    # Calculate trapezoidal AUC
    auc_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc_roc = float(auc_func(tprs, fprs)) if auc_func else 0.0

    return fprs, tprs, thresholds, auc_roc


def compute_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_thresholds: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute vectorized Precision-Recall curve (Recall, Precision, thresholds, AUC-PR)."""
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    recalls = []
    precisions = []

    pos_count = np.sum(y_true == 1)

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        rec = tp / pos_count if pos_count > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recalls.append(rec)
        precisions.append(prec)

    recalls = np.array(recalls)
    precisions = np.array(precisions)

    # Sort by recall ascending for integration
    idx = np.argsort(recalls)
    rec_sorted = recalls[idx]
    prec_sorted = precisions[idx]

    auc_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc_pr = float(auc_func(prec_sorted, rec_sorted)) if auc_func else 0.0

    return recalls, precisions, thresholds, auc_pr


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    criterion: str = "f1",
    min_precision: float = 0.50,
    customer_value: float = 1000.0,
    offer_cost: float = 50.0,
    success_rate: float = 0.25,
) -> Tuple[float, Dict[str, Any]]:
    """Find the optimal decision threshold based on a business or statistical goal."""
    sweep_thresholds = np.linspace(0.05, 0.95, 181)
    best_thresh = 0.50
    best_score = -float("inf")
    best_metrics: Dict[str, Any] = {}

    for t in sweep_thresholds:
        m = compute_confusion_matrix(y_true, y_prob, float(t))
        if criterion == "f1":
            score = m["f1"]
        elif criterion == "recall_with_min_prec":
            score = m["recall"] if m["precision"] >= min_precision else -1.0
        elif criterion == "profit":
            # Net Benefit = TP * success * value - (TP + FP) * cost
            score = (m["tp"] * success_rate * customer_value) - ((m["tp"] + m["fp"]) * offer_cost)
        else:
            score = m["accuracy"]

        if score > best_score:
            best_score = score
            best_thresh = float(t)
            best_metrics = m

    return best_thresh, best_metrics
