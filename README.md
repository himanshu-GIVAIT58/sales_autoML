# SKU-Level Sales Forecasting & Inventory Optimization Pipeline

## 🚀 Overview

This project provides an end-to-end, automated pipeline for forecasting future sales demand at the individual SKU level. It leverages **AutoML with AutoGluon** for robust time-series modeling and includes a complete MLOps workflow for model validation, promotion, and versioning. Results are visualized in an interactive Streamlit dashboard.

---

## ✨ Key Features

- Automated training pipeline (`main.py`)
- SKU-level forecasting across channels
- AutoML with AutoGluon
- Rich feature engineering (lags, calendar, holidays, inventory, pricing)
- ABC analysis for SKU prioritization
- Champion-challenger model validation
- Inventory metric calculation (Safety Stock, ROP, EOQ)
- Interactive Streamlit dashboard

---

## ⚙️ Pipeline Steps

1. **Data Loading**: Pulls sales, inventory, and holiday data from MongoDB.
2. **Feature Engineering**: Cleans and transforms data into input features.
3. **ABC Analysis**: Segments SKUs for prioritization.
4. **Model Training**: Trains multiple time-series models using AutoGluon.
5. **Model Validation**: Compares new models to the current champion.
6. **Recommendation Generation**: Generates forecasts and inventory recommendations.
7. **Saving & Logging**: Saves recommendations and logs artifacts to MongoDB.

---

## 🛠️ Technology Stack

- Python, pandas, scikit-learn, joblib
- AutoGluon (TimeSeries)
- MongoDB
- Streamlit

---

## 📂 File-by-File Explanation

### Top-Level Files

- **README.md**: Project documentation and usage instructions.
- **Dockerfile**, **docker-compose.yml**: Containerization for reproducible environments.
- **.env**, **.env.example**, **.env.local**: Environment variable configuration.
- **service_account.json**: Credentials for external services (if used).
- **training.log**: Log file for training runs.
- **dev.sh**, **vm_connect.sh**: Shell scripts for development and VM access.

### Folders

- **.devcontainer/**: VS Code dev container configuration.
- **.github/**: GitHub Actions workflows for CI/CD.
- **.vscode/**: VS Code workspace settings and tasks.
- **artifacts/**: Stores static data artifacts (e.g., holidays.csv, feature columns).
- **autogluon_models/**: Stores trained model artifacts and logs.
- **k8s/**: Kubernetes deployment manifests.

### `src/` Directory

#### Main Pipeline

- **main.py**  
  Orchestrates the entire pipeline: loads data, runs feature engineering, trains models, validates, and saves recommendations. Entry point for retraining (`python3 -m src.main`).

#### Data Handling

- **data_loader.py**  
  Loads data from MongoDB, saves recommendations, and provides utility functions for data access.

#### Feature Engineering

- **feature_engineering.py**  
  Cleans and transforms raw sales/inventory/holiday data into model-ready features (lags, rolling stats, calendar, stockout, pricing).

#### Model Training & Validation

- **advanced_pipeline.py**  
  Contains advanced logic for model training, champion-challenger selection, forecasting, clustering, and inventory metric calculation. Handles deep learning, XGBoost, Prophet, and ensemble models.

- **model_handler.py**  
  Manages AutoGluon model training, prediction, and ensemble logic. Handles model loading and artifact management.

#### Inventory & Recommendation

- **inventory_calculator.py**  
  Converts forecasts into actionable inventory metrics (Safety Stock, ROP, EOQ) and applies business rules for recommendations.

#### Dashboard & Visualization

- **inventory_dashboard_streamlit.py**  
  Streamlit dashboard for interactive visualization of forecasts, SKU analysis, inventory simulation, and executive summaries.

#### Promotion Analysis

- **promo_analyzer.py**  
  Analyzes the impact of promotions on SKU sales, calculates revenue lift and ROI.

#### Seasonal Analysis

- **seasonal_analysis.py**  
  Identifies seasonal SKUs and calculates seasonal strength using time-series decomposition.

#### Configuration

- **config.py**  
  Centralized configuration for paths, model parameters, and pipeline settings.

#### Utilities

- **dbConnect.py**  
  Handles MongoDB connection logic.

---

## 🚀 How to Use

### Running the Training Pipeline

```bash
python3 -m src.main
```
