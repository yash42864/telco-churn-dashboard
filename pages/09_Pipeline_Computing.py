"""Page 9: Pipeline and Cluster Computing Architecture.

Comprehensive technical architecture documentation covering the PySpark ETL DAG,
distributed MLlib pipeline stages, core Apache Spark concepts, algorithm design,
data leakage prevention, and computational constraints.
"""

import streamlit as st
import pandas as pd

from utils.theme import (
    apply_custom_theme,
    render_page_header,
    COLOR_CHURN,
    COLOR_SAFE,
    COLOR_PRIMARY,
    COLOR_AMBER,
    COLOR_PURPLE,
)

apply_custom_theme()

render_page_header(
    title="Pipeline & Cluster Architecture",
    subtitle="Distributed data engineering and machine learning workflows built with Apache Spark (PySpark MLlib).",
    badge="Architecture & Engineering",
    icon="⚡",
    how_to_read="""
    **Architecture Guide**:
    - **Pipeline Topology**: Outlines the sequence from raw unstructured CSV through schema enforcement, one-hot encoding, and parallel model training.
    - **Spark Primitives**: Summarizes core distributed computing concepts (Catalyst Optimizer, RDD caching, lazy DAG execution).
    - **Leakage Prevention**: Explains the deliberate isolation and removal of future-derived attributes.
    - **Limitations**: Outlines constraints and production enhancements for enterprise deployment.
    """,
)

# 1. Pipeline Diagram
st.markdown("### 🗺️ PySpark End-to-End Pipeline DAG")
st.caption("Directed Acyclic Graph (DAG) detailing transformation stages from raw data ingestion to dashboard consumption.")

pipeline_dot = """
digraph G {
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fontname="Sora, Helvetica", fontsize=11, margin="0.2,0.15"];
    edge [color="#818CF8", fontname="Sora, Helvetica", fontsize=10, arrowsize=0.8];

    subgraph cluster_ingestion {
        label="1. Ingestion & Storage";
        fontname="Sora, Helvetica";
        fontcolor="#94A3B8";
        style="dashed";
        color="rgba(255,255,255,0.15)";
        
        raw [label="Raw Telco CSV\\n(7,043 rows · 21 cols)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#64748B"];
        clean [label="Spark Clean & Cast\\n(02_load_and_clean.py)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#818CF8"];
        parquet [label="Parquet Storage\\n(telco_clean.parquet)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
    }

    subgraph cluster_fe {
        label="2. Distributed Feature Engineering";
        fontname="Sora, Helvetica";
        fontcolor="#94A3B8";
        style="dashed";
        color="rgba(255,255,255,0.15)";

        indexer [label="StringIndexer\\n(15 Categoricals)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#818CF8"];
        ohe [label="OneHotEncoder\\n(41 Binary Features)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#818CF8"];
        assembler [label="VectorAssembler\\n(45 Raw Features)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#818CF8"];
        scaler [label="StandardScaler\\n(Zero-bias Unit Variance)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#818CF8"];
    }

    subgraph cluster_ml {
        label="3. Model Training & Validation";
        fontname="Sora, Helvetica";
        fontcolor="#94A3B8";
        style="dashed";
        color="rgba(255,255,255,0.15)";

        train_test [label="80/20 Train/Test Split\\n(train.parquet / test.parquet)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F59E0B"];
        classifiers [label="7 Parallel Classifiers\\n(L-BFGS, Trees, SVM, NB)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#F43F5E"];
        reg_kmeans [label="Linear Regression &\\nK-Means Clustering", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#A855F7"];
    }

    subgraph cluster_serving {
        label="4. Export & Presentation";
        fontname="Sora, Helvetica";
        fontcolor="#94A3B8";
        style="dashed";
        color="rgba(255,255,255,0.15)";

        export [label="Model Artifact Export\\n(lr_params.json & CSVs)", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#10B981"];
        dashboard [label="Streamlit Analytics UI\\n(Zero Java / Spark runtime)", fillcolor="#818CF8", fontcolor="#0B0F19", color="#FFFFFF", style="rounded,filled,bold"];
    }

    raw -> clean -> parquet;
    parquet -> indexer -> ohe -> assembler -> scaler -> train_test;
    train_test -> classifiers -> export;
    train_test -> reg_kmeans -> export;
    export -> dashboard;
}
"""

try:
    st.graphviz_chart(pipeline_dot, use_container_width=True)
except Exception:
    st.info("Pipeline DAG topology: Raw CSV ➔ Spark Cleaning ➔ Parquet ➔ StringIndexer ➔ OneHotEncoder ➔ VectorAssembler ➔ StandardScaler ➔ Classifiers / Regression / KMeans ➔ Export ➔ Streamlit Dashboard.")

st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

# 2. Spark Distributed Concepts Cards
st.markdown("### 🧩 Core Apache Spark Distributed Concepts")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        """
        <div class="glass-card" style="height: 100%;">
            <div style="font-weight: 700; color: #818CF8; font-size: 0.95rem;">DataFrame & Catalyst</div>
            <div style="font-size: 0.8rem; line-height: 1.5; color: #94A3B8; margin-top: 0.5rem;">
                Distributed collections of partitioned records. The Catalyst Optimizer builds logical query plans,
                prunes columns, and generates optimized JVM bytecode before execution.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="glass-card" style="height: 100%;">
            <div style="font-weight: 700; color: #10B981; font-size: 0.95rem;">Transformers & Estimators</div>
            <div style="font-size: 0.8rem; line-height: 1.5; color: #94A3B8; margin-top: 0.5rem;">
                <b>Estimators</b> implement <code>.fit()</code> to learn parameters from distributed data (e.g., LogisticRegression).
                <b>Transformers</b> implement <code>.transform()</code> (e.g., VectorAssembler, StandardScalerModel).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="glass-card" style="height: 100%;">
            <div style="font-weight: 700; color: #F59E0B; font-size: 0.95rem;">ML Pipelines</div>
            <div style="font-size: 0.8rem; line-height: 1.5; color: #94A3B8; margin-top: 0.5rem;">
                Encapsulates multi-stage workflows into a single repeatable DAG. Ensures identical preprocessing
                transformations are applied to test sets without data leakage or manual column sync.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        """
        <div class="glass-card" style="height: 100%;">
            <div style="font-weight: 700; color: #F43F5E; font-size: 0.95rem;">Lazy Evaluation & Caching</div>
            <div style="font-size: 0.8rem; line-height: 1.5; color: #94A3B8; margin-top: 0.5rem;">
                Transformations build an execution lineage graph without computing results until an action
                (e.g., <code>.count()</code> or <code>.save()</code>) is invoked. <code>.cache()</code> persists intermediate RDDs in executor memory.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

# Partitions & Hardware Cluster callout
st.markdown(
    """
    <div class="glass-card" style="border-left: 4px solid #818CF8; padding: 0.9rem 1.2rem;">
        <b>Cluster Configuration & Partitioning</b>: The PySpark cluster is initialized using 
        <code>SparkSession.builder.master("local[*]").appName(...)</code>, spawning 8 parallel worker threads 
        matching local CPU cores. DataFrames are partitioned across executors to facilitate parallel vectorized 
        aggregations and concurrent mini-batch gradient evaluations.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# 3. Algorithms & Metrics Reference Table
st.markdown("### 📚 Model Architectures & Evaluation Metrics")

algo_data = [
    {
        "Algorithm": "Logistic Regression (Unweighted)",
        "Objective / Math": "$\\min_w -\\sum [y_i \\log(p_i) + (1-y_i) \\log(1-p_i)] + \\lambda \\|w\\|_2^2$",
        "Role in Project": "Linear baseline; optimizes maximum overall accuracy across all accounts.",
        "Primary Metric": "Accuracy (80.8%), AUC-ROC (0.8517)",
    },
    {
        "Algorithm": "Logistic Regression (Class-Weighted)",
        "Objective / Math": "Loss re-weighted inversely by class frequency: $w_1 = N / (2 N_1)$",
        "Role in Project": "Production champion; heavily penalizes false negatives to maximize recall.",
        "Primary Metric": "Recall (78.3%), AUC-ROC (0.8515)",
    },
    {
        "Algorithm": "Decision Tree Classifier",
        "Objective / Math": "Recursive binary partitioning maximizing Information Gain via Gini impurity",
        "Role in Project": "Non-linear benchmark; captures feature interaction thresholds.",
        "Primary Metric": "F1 (0.579), Train Time (0.45s)",
    },
    {
        "Algorithm": "Decision Tree (Balanced)",
        "Objective / Math": "Class-weighted Gini split criterion",
        "Role in Project": "Non-linear high-recall variant; identifies multi-variable churn risk rules.",
        "Primary Metric": "Recall (84.0%), Train Time (0.33s)",
    },
    {
        "Algorithm": "Naive Bayes (Bernoulli)",
        "Objective / Math": "Conditional independence assumption: $P(y|x) \\propto P(y) \\prod P(x_i|y)$",
        "Role in Project": "Ultra-fast distributed baseline; single-pass aggregation.",
        "Primary Metric": "Train Time (0.19s), Recall (77.8%)",
    },
    {
        "Algorithm": "Linear Support Vector Machine (SVM)",
        "Objective / Math": "Hinge loss with L2 regularization: $\\min_w \\frac{1}{2}\\|w\\|^2 + C \\sum \\max(0, 1 - y_i w^T x_i)$",
        "Role in Project": "Maximum margin hyperplane separator; robust to boundary outliers.",
        "Primary Metric": "Accuracy (81.1%), Recall (54.7%)",
    },
    {
        "Algorithm": "Linear Regression",
        "Objective / Math": "Ordinary Least Squares (OLS) minimizing $\\|Xw - y\\|_2^2$",
        "Role in Project": "Auxiliary model predicting Monthly_Charges to verify pricing determinism.",
        "Primary Metric": "$R^2$ (0.9989), RMSE ($1.03)",
    },
    {
        "Algorithm": "K-Means Clustering",
        "Objective / Math": "Minimizing Within-Cluster Sum of Squares (WCSS / WSSSE) using k-means|| initialization",
        "Role in Project": "Unsupervised segmentation of customer portfolio into 4 behavioral archetypes.",
        "Primary Metric": "Silhouette (0.344), WSSSE Cost",
    },
]

st.dataframe(pd.DataFrame(algo_data), use_container_width=True, hide_index=True)

st.markdown("---")

# 4. Data Leakage Prevention Case Study
st.markdown("### 🛡️ Data Leakage Prevention Case Study")

leakage_cols = st.columns([1.5, 1])
with leakage_cols[0]:
    st.markdown(
        """
        In commercial churn modeling, data leakage is the most prevalent cause of catastrophic post-deployment failure.
        Four columns present in the raw IBM Telco dataset were systematically audited, flagged, and excised in `scripts/02_load_and_clean.py`:

        1. **`Churn Label`**: The redundant string target identical to `Churn Value`.
        2. **`Churn Score`**: A pre-existing credit-risk style synthetic score (0–100) generated using a proprietary historical model. Including it would create a circular dependency.
        3. **`CLTV` (Customer Lifetime Value)**: Computed retroactively by IBM using churn duration and total tenure; incorporates knowledge of the future cancellation event.
        4. **`Churn Reason`**: Text commentary populated solely *after* an exit interview (e.g., *"Competitor offered higher download speeds"*). In production, this field is strictly null for active subscribers.

        **Result**: All 7 PySpark classifiers were trained purely on observable behavioral, transactional, and service attributes available *prior* to cancellation.
        """
    )
with leakage_cols[1]:
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid {COLOR_CHURN}; height: 100%;">
            <div style="font-weight: 700; color: {COLOR_CHURN}; margin-bottom: 0.4rem;">Leakage Audit Summary</div>
            <div style="font-size: 0.8rem; line-height: 1.5; color: #94A3B8;">
                • Removed <b>4 leaky columns</b> prior to vectorization.<br>
                • Preserved <b>21 pristine features</b> (15 categorical, 4 numerical, 2 binary identifiers).<br>
                • Validated zero post-event data contamination.<br>
                • Prevents inflated offline metrics from degrading during live production inference.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# 5. Limitations & Future Roadmap
st.markdown("### 🔭 Engineering Limitations & Future Roadmap")

lim_c1, lim_c2 = st.columns(2)
with lim_c1:
    st.markdown(
        """
        <div class="glass-card">
            <div style="font-weight: 700; color: #F59E0B; margin-bottom: 0.4rem;">Current Engineering Limitations</div>
            <div style="font-size: 0.82rem; line-height: 1.5; color: #94A3B8;">
                • <b>Single 80/20 Train/Test Partition</b>: While stratified, a single split remains vulnerable to sample variance compared with full k-fold cross-validation.<br>
                • <b>In-Sample Full-Cohort Scoring</b>: Exported customer scores in <code>customer_scores.csv</code> represent in-sample evaluations rather than out-of-fold predictions.<br>
                • <b>Static Snapshot Dataset</b>: The IBM Telco dataset is cross-sectional; time-series behavioral dynamics (e.g., declining data usage over the past 30 days) are absent.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with lim_c2:
    st.markdown(
        """
        <div class="glass-card">
            <div style="font-weight: 700; color: #10B981; margin-bottom: 0.4rem;">Production Deployment Roadmap</div>
            <div style="font-size: 0.82rem; line-height: 1.5; color: #94A3B8;">
                • <b>PySpark CrossValidator</b>: Integrate 5-fold cross-validation and hyperparameter grid searches via <code>ParamGridBuilder</code>.<br>
                • <b>Out-of-Fold Probability Stacking</b>: Generate unbiased customer risk probabilities via cross-validated out-of-fold scoring.<br>
                • <b>Spark Structured Streaming</b>: Stream real-time call center logs and payment failures via Kafka into a continuous Spark streaming inference pipeline.<br>
                • <b>Automated Retraining DAG</b>: Airflow / Prefect orchestration scheduled to retrain models monthly on rolling historical windows.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
