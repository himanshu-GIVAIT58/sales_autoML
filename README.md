# SKU-Level Sales Forecasting & Inventory Optimization Pipeline

## 🚀 Overview

This project provides an end-to-end, automated pipeline for forecasting future sales demand at the individual SKU level. It leverages the power of **AutoML with AutoGluon** to train robust time-series models, and includes a complete MLOps workflow for model validation, promotion, and versioning.

The primary goal is to move beyond simple historical averages and generate **accurate, actionable inventory recommendations** (like Reorder Points and Economic Order Quantity) to:
- Optimize stock levels
- Reduce carrying costs
- Prevent stockouts

The entire system is managed through a **Python pipeline** that pulls data from **MongoDB**, and the results are visualized in an **interactive Streamlit dashboard**.

---

## ✨ Key Features

- **Automated Training Pipeline**  
  A single script (`main.py`) handles data loading, feature engineering, model training, and validation.

- **SKU-Level Forecasting**  
  Generates granular demand forecasts for every unique product SKU across different sales channels (Offline, App, Web).

- **AutoML with AutoGluon**  
  Automatically trains and evaluates a suite of time-series models—from simple baselines to complex deep learning models.

- **Rich Feature Engineering**  
  Includes:
  - Sales history (lags, rolling averages, standard deviations)
  - Calendar features (day of week, month)
  - Holiday and special event data
  - Inventory levels and stockout history
  - Pricing and discount information

- **ABC Analysis for Prioritization**  
  Classifies products into **A, B, and C** categories based on sales volume to prioritize high-value items.

- **Automated Model Validation**  
  Implements a **champion-challenger** model comparison system. A new model is promoted only if it outperforms the current one.

- **Inventory Metric Calculation**  
  Converts forecasts into metrics like:
  - Safety Stock
  - Reorder Point (ROP)
  - Economic Order Quantity (EOQ)

- **Interactive Dashboard**  
  A user-friendly **Streamlit app** for:
  - Forecast visualization
  - SKU performance analysis
  - Inventory simulation

---

## ⚙️ How It Works: The Pipeline

The core of the project is an automated pipeline orchestrated by `src/main.py`. It executes the following steps:

1. **Data Loading**  
   Pulls sales, inventory, and holiday data from **MongoDB**.

2. **Feature Engineering**  
   Cleans and transforms data into meaningful input features.

3. **ABC Analysis**  
   Segments SKUs to focus on critical items.

4. **Model Training**  
   Uses **AutoGluon** to train multiple time-series models using specified quality presets (`fast_training`, `high_quality`, etc.).

5. **Model Validation**
   - Loads the current “champion” model.
   - Compares it to the new model using recent data.
   - Decides to **promote** or **reject** the new model based on improvement thresholds.

6. **Recommendation Generation**
   - If promoted, generates forecasts.
   - Calculates inventory recommendations.

7. **Saving & Logging**
   - Saves recommendations to **MongoDB** (versioned).
   - Logs model artifacts and validation metrics.
   - Archives old recommendations.

---

## 🛠️ Technology Stack

- **Backend:** Python  
- **Machine Learning:** AutoGluon (TimeSeries)  
- **Data Storage:** MongoDB  
- **Dashboard:** Streamlit  
- **Core Libraries:** `pandas`, `scikit-learn`, `joblib`

---

## 🚀 How to Use

### Running the Training Pipeline

To retrain the model with the latest data and generate new recommendations:

```bash
python3 -m src.main
