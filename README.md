# ⚡ Subscription Cancellation (Churn) Analytics Dashboard

A high-performance, multipage executive analytics dashboard and decision support system built for the university project: **"Predicting Subscription Cancellation (Churn) with PySpark"**.

This dashboard allows business leaders, data scientists, and marketing retention teams to diagnose why subscribers leave, compare distributed machine learning classifiers, explore behavioral customer segments, score individual accounts in real-time, and design high-ROI retention campaigns.

---

## 🚀 Quickstart & Run Instructions

The dashboard operates completely offline from pre-computed PySpark export artifacts in `../outputs/`. It requires **zero** Java or Apache Spark dependencies at runtime, ensuring instant startup and sub-second page transitions.

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Virtual environment recommended

### 2. Installation
Navigate into the dashboard directory and install the lightweight dependencies:
```bash
cd churn-project/dashboard
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Deploy to Streamlit Community Cloud (Free & 1-Click)

The dashboard is 100% self-contained in `dashboard/` with pre-packaged data artifacts in `dashboard/outputs/`. You can deploy it live on the web in 3 minutes for free:

### Step 1: Push `dashboard/` to a GitHub Repository
If you haven't initialized Git yet:
```bash
cd churn-project/dashboard
git init
git add .
git commit -m "Initial commit: Telco Churn PySpark Dashboard"
git branch -M main
```
Create a new repository on [GitHub](https://github.com/new) (e.g. `telco-churn-dashboard`), then link and push:
```bash
git remote add origin https://github.com/<YOUR-USERNAME>/telco-churn-dashboard.git
git push -u origin main
```

*(Note: If you push the entire `churn-project` folder instead, simply specify `dashboard/app.py` as your main file path in Step 2).*

### Step 2: Deploy on Streamlit Community Cloud
1. Sign in to **[share.streamlit.io](https://share.streamlit.io)** using your GitHub account.
2. Click the **"New app"** button.
3. Select your repository: `<YOUR-USERNAME>/telco-churn-dashboard`.
4. Branch: `main`.
5. Main file path: `app.py` (or `dashboard/app.py` if your repository root is `churn-project`).
6. Click **Deploy!**

Streamlit Cloud will automatically detect `requirements.txt` and `.streamlit/config.toml`, install dependencies, and launch your live application with a public HTTPS URL (e.g. `https://<your-username>-telco-churn.streamlit.app`).

---

## 🐳 Containerized Deployment (Docker)

A production-ready `Dockerfile` and `.dockerignore` are included:

```bash
cd churn-project/dashboard

# Build container image
docker build -t telco-churn-dashboard:latest .

# Run container on port 8501
docker run -p 8501:8501 telco-churn-dashboard:latest
```
Access the application at `http://localhost:8501`. Compatible with AWS ECS, GCP Cloud Run, Render, Railway, or any Docker host.

---

## 🏛️ Architecture & Data Contract

All data consumed by the dashboard is exported by the Apache Spark pipeline scripts in `../scripts/` and stored in `../outputs/`:

| Output Artifact | Generating Script | Description |
| :--- | :--- | :--- |
| `kpis.csv` | `03_eda.py` | Portfolio totals: 7,043 customers, 26.54% churn rate, tenure, and ARPU |
| `churn_by_category.csv` | `03_eda.py` | Categorical churn rates across 8 key attributes (Contract, Internet, etc.) |
| `correlation_matrix.csv` | `03_eda.py` | 9x9 Pearson correlation matrix across numerical and binary attributes |
| `model_comparison.csv` | `05_train_classifiers.py` | Held-out test set metrics (Accuracy, Precision, Recall, F1, AUC, Runtimes) for 7 models |
| `feature_importance.csv` | `05_train_classifiers.py` | Standardized coefficients (Logistic Regression) and Gini importances (Decision Trees) |
| `regression_metrics.csv` | `06_regression.py` | $R^2$ (0.9989), RMSE, and baseline comparisons predicting `Monthly_Charges` |
| `kmeans_selection.csv` | `07_kmeans.py` | WSSSE cost and Silhouette scores across cluster counts $k = 2 \dots 8$ |
| `cluster_profiles.csv` | `07_kmeans.py` | Behavioral centroids, subscriber volumes, and observed churn rates for $k=4$ |
| `cluster_assignments.csv`| `07_kmeans.py` | 7,043 customer rows assigned to clusters with 2D coordinates |
| `customer_scores.csv` | `08_export.py` | Full portfolio scored with class-weighted Logistic Regression probabilities |
| `lr_params.json` | `08_export.py` | Standardized feature scales, coefficients, and intercept for offline inference |

> **Graceful Error Handling**: If any artifact is missing, the dashboard will not crash. It displays a clean, user-friendly notice indicating the exact PySpark script required to generate the file.

---

## 📑 Page-by-Page Feature Tour

### 1. 🏠 Executive Overview
- **Hero KPI Cards**: Active subscribers (7,043), baseline churn rate (26.54%), lost accounts (1,869), average tenure (32.4 mo), and average monthly charge ($64.76) with dynamic delta indicators against selectable industry benchmarks.
- **Dynamic Findings**: 4 computed insight cards highlighting contract risks (15.1x risk on month-to-month), fiber optic attrition (2.2x higher than DSL), tenure correlation ($r = -0.352$), and tech support protection.
- **Revenue at Risk**: Direct monetary quantification summing monthly recurring revenue (MRR) for accounts crossing the $p \ge 0.50$ churn threshold.
- **Quick Navigation**: Direct one-click launcher buttons to all strategic dashboard modules.

### 2. 📊 Churn Drivers (EDA)
- **Categorical Risk Profiles**: Interactive bar chart across all 8 categorical dimensions with overall-portfolio reference lines and sort toggles.
- **Compounding Risk Heatmap**: Dynamic two-way cross-tabulation uncovering intersecting friction zones (e.g. Month-to-Month contracts $\times$ Fiber Optic service).
- **Correlation Matrix**: Plotly diverging heatmap with a toggle to rank attributes purely by their correlation with churn ($label$).
- **Distribution Split**: Violin and histogram density plots contrasting tenure tenure and monthly charges between retained and churned cohorts.

### 3. 🧪 Model Lab
- **Leaderboard Comparison**: Comprehensive table benchmarking 7 PySpark classifiers with automated best-in-column green highlighting.
- **Multi-Metric Bar & Polar Radar**: Visualizes multi-dimensional model silhouettes across Accuracy, Precision, Recall, F1, AUC-ROC, and AUC-PR.
- **Confusion Matrix Viewer**: 2x2 matrix with plain-English business translations ("Catches 289 of 369 churners, wastes 256 offers").
- **Precision/Recall Trade-off Space**: Scatter plot illustrating how class-weighted models shift into high-recall territory.
- **Cluster Training Time**: Execution benchmark illustrating PySpark distributed training runtimes across model architectures.
- **Linear Regression Panel**: Validates tariff determinism ($R^2 = 0.9989$) for monthly charges.
- **Feature Importance Explorer**: Visualizes standardized coefficients and Gini importances with collinearity warnings.

### 4. 🎯 Threshold & ROC Studio
- **Dynamic Threshold Slider ($0.05 \dots 0.95$)**: Recalculates confusion matrix counts, sensitivity, specificity, and precision in real-time.
- **Live Curve Trackers**: Interactive ROC and Precision-Recall curves marking the active threshold point with dynamic coordinates.
- **Optimization Presets**: Instant one-click solvers for:
  - *Maximize F1 Score* (balances harmonic mean)
  - *Maximize Recall at Precision $\ge 50\%$* (ensures at least 1 in 2 offers is targeted correctly)
  - *Maximize Net Profit* (calibrated to customer LTV and offer costs)

### 5. 👥 Customer Segments (K-Means)
- **Model Selection**: WSSSE Elbow curve and Silhouette score highlighting the optimal choice of $k=4$.
- **Segment Landscape**: Macro bubble chart positioning clusters by tenure duration, monthly expenditure, customer volume, and churn rate.
- **Archetype Cards**: Algorithmic categorization into actionable business personas (*High-Risk Flight Risk*, *High-Value Champions*, *Steady Loyalists*, *Budget Growth*) with concrete retention playbooks.
- **Micro-Level Scatter**: Individual customer 2D scatter with sampling controls, churn filters, and centroid overlays.

### 6. 🔎 Customer Risk Explorer
- **Interactive Multi-Filter Sidebar**: Filter subscriber base by contract, internet type, payment method, tenure span, monthly price, and predicted risk.
- **Reactive KPI Strip**: Instantly updates filtered customer count, average risk, expected defectors, and revenue exposure.
- **Individual Risk Card**: Search any Customer ID to view their personalized gauge and top positive/negative standardized risk drivers.
- **Searchable Roster & CSV Export**: Interactive data table with formatted probability progress bars and one-click filtered CSV download.

### 7. 🔮 Predict & What-if Simulator
- **Live Inference Engine**: Replicates PySpark class-weighted Logistic Regression scoring offline without mean-centering bias.
- **Scenario Presets**: Instant loading of *High-Risk Example* (86.5% churn risk) vs *Loyal Champion Example* (2.4% churn risk).
- **Waterfall Decomposition**: Visualizes individual feature contributions to log-odds.
- **What-If Intervention Engine**: Live simulation testing how contract extensions or service add-ons mitigate cancellation risk before taking action.

### 8. 💰 Retention ROI Planner
- **Campaign Financial Modeling**: Customizable customer lifetime value (LTV), offer cost, success rate, and marketing budget cap.
- **Cross-Model ROI Benchmark**: Compares net dollar benefit across all 7 classifiers, proving that high-recall balanced models maximize profitability.
- **Sensitivity Matrix**: 2D grid identifying break-even frontiers across varying offer unit costs and acceptance rates.
- **Target Roster Generator**: Allocates budget to the highest-risk subscribers and generates a downloadable targeted campaign CSV.

### 9. ⚡ Pipeline & Cluster Architecture
- **Distributed DAG Flowchart**: Clean Graphviz diagram visualizing the workflow from raw CSV to Parquet, MLlib pipelines, and dashboard export.
- **Apache Spark Primitives**: Detailed technical breakdowns of DataFrames, Catalyst Optimizer, Transformers, Estimators, ML Pipelines, and RDD Caching.
- **Data Leakage Case Study**: Explains why `Churn Label`, `Churn Score`, `CLTV`, and `Churn Reason` were identified as targets leaks and removed.
- **Limitations & Roadmap**: Outlines production considerations including k-fold cross-validation, streaming ingestion, and automated Airflow retraining.

---

## 🎨 Design System & Accessibility

- **Theme Palette**: Deep space navy background (`#0B0F19`), crisp slate surfaces (`#161F30`), electric periwinkle accents (`#818CF8`), vibrant coral for churn (`#F43F5E`), and emerald mint for retention (`#10B981`).
- **Crisp Light Variant**: Easily selectable in the sidebar with matching accessible contrast ratios.
- **Typography**: Google Font **Sora** with an executive typographic hierarchy.
- **Micro-Interactions**: Subtle glassmorphism, floating card hover elevation, and full adherence to `prefers-reduced-motion`.

---

## 📸 Capturing Screenshots for Reports

To capture presentation screenshots for project documentation:
1. Open the dashboard in Chrome, Safari, or Firefox at full HD resolution ($1920 \times 1080$).
2. In the sidebar, select your preferred theme mode (**Dark Space** or **Crisp Light**).
3. Use your browser's Developer Tools (`Cmd + Shift + P` -> `Capture full size screenshot` in Chrome) or native macOS shortcut (`Cmd + Shift + 4` -> `Space` to capture window).
4. Recommended views to capture:
   - Executive Overview (Hero KPI cards and Churn Donut)
   - Model Lab (Leaderboard and Radar chart)
   - Threshold Studio (Interactive slider and ROC/PR curves)
   - Predict Simulator (What-if intervention comparison)
