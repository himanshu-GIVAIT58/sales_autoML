# main.py
"""
Main function to run the entire, unified forecasting pipeline.
It now saves training artifacts to enable fast, on-demand predictions.
"""

import datetime
import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from autogluon.timeseries import TimeSeriesDataFrame
import joblib

# Import project modules
from . import config
from . import data_loader
from . import feature_engineering
from . import model_handler
from . import inventory_calculator
from . import model_monitor

def main():
    """Main function to run the entire pipeline."""
    print("🚀 Starting Unified Forecasting Pipeline...")
    print("-" * 60)

    # --- 1. Load, Combine, and Prepare All Data ---
    print("Step 1: Loading and combining all data from MongoDB...")
    try:
        sales_df = data_loader.load_dataframe_from_mongo("sales_data")
        print(f"   -> Loaded {len(sales_df)} historical sales records.")
        new_sales_df = data_loader.load_dataframe_from_mongo("new_sales_data_uploads")
        if not new_sales_df.empty:
            print(f"   -> Found {len(new_sales_df)} new records to integrate.")
            new_sales_df.rename(columns={'sku': 'item_id', 'timestamp': 'order_date', 'disc': 'discount_percentage'}, inplace=True)
            new_sales_df['order_date'] = pd.to_datetime(new_sales_df['order_date'])
            sales_df = pd.concat([sales_df, new_sales_df], ignore_index=True)
            sales_df.drop_duplicates(subset=['item_id', 'order_date'], keep='last', inplace=True)
            print(f"   -> Combined dataset now has {len(sales_df)} unique records.")
        else:
            print("   -> No new sales data found.")
        inventory_df = data_loader.load_dataframe_from_mongo("query_result")
        holidays_df = data_loader.load_dataframe_from_mongo("holidays_data")
        print("   -> Successfully loaded inventory and holidays data.")
    except Exception as e:
        print(f"❌ Critical Error: Failed to load data. Aborting. Error: {e}")
        return

    # --- 2. Prepare Data and Engineer Features ---
    print("\nStep 2: Preparing data and running feature engineering...")
    processed_data, static_features_base = feature_engineering.prepare_data(
        sales_df, inventory_df, holidays_df, config.MAX_SKUS
    )
    print("   -> Feature engineering complete.")

    data_loader.save_dataframe_to_mongo(processed_data, "processed_training_data")
    print("   -> Saved processed training data snapshot to MongoDB.")

    # --- 3. ABC Analysis & Data Filtering ---
    print("\nStep 3: Performing ABC Analysis and data filtering...")
    total_sales = processed_data.groupby('sku')['target'].sum().sort_values(ascending=False)
    total_sales_df = total_sales.to_frame()
    a_cutoff = int(len(total_sales_df) * config.ABC_CONFIG['A_class_percentage'])
    b_cutoff = a_cutoff + int(len(total_sales_df) * config.ABC_CONFIG['B_class_percentage'])
    total_sales_df['class'] = 'C'
    total_sales_df.loc[total_sales_df.index[:a_cutoff], 'class'] = 'A'
    total_sales_df.loc[total_sales_df.index[a_cutoff:b_cutoff], 'class'] = 'B'
    item_to_class_map = total_sales_df['class'].to_dict()
    item_counts = processed_data["item_id"].value_counts()
    items_with_sufficient_data = item_counts[item_counts > config.PREDICTION_LENGTH].index
    filtered_data = processed_data[processed_data["item_id"].isin(items_with_sufficient_data)]
    print(f"   -> Number of SKU-channel combos with sufficient history: {len(items_with_sufficient_data)}")

    # --- 4. Prepare Data for AutoGluon (Corrected) ---
    print("\nStep 4: Preparing data for AutoGluon TimeSeriesDataFrame...")
    for col in config.KNOWN_COVARIATES_NAMES:
        if col not in filtered_data.columns:
            filtered_data[col] = 0

    # --- KEY FIX ---
    # The 'static_features_base' DataFrame returned by prepare_data is already what we need.
    # It is indexed by 'item_id', so we just reset the index to make 'item_id' a column.
    final_static_features = static_features_base.reset_index()

    ts_data = TimeSeriesDataFrame.from_data_frame(
        filtered_data,
        id_column="item_id",
        timestamp_column="timestamp",
        static_features_df=final_static_features
    )
    print("   -> TimeSeriesDataFrame created successfully.")

    # --- 5. Train and Evaluate Model ---
    print("\nStep 5: Training new model with AutoGluon...")
    predictor = model_handler.train_predictor(ts_data, config)
    print("   -> Model training complete. Evaluating...")
    metrics = model_handler.evaluate_predictor(predictor, ts_data)
    if metrics is None:
        metrics = {'MAPE': 999}
    print(f"   -> Model evaluation metrics: {metrics}")

    # --- 6. Save Training Artifacts for Fast Predictions ---
    print("\nStep 6: Saving training artifacts for fast prediction...")
    os.makedirs('artifacts', exist_ok=True)
    static_feature_columns = list(final_static_features.drop(columns=['item_id']).columns)
    joblib.dump(static_feature_columns, 'artifacts/static_feature_columns.joblib')
    print("   -> Saved static feature columns.")

    holidays_df_for_future = data_loader.load_dataframe_from_mongo("holidays_data")
    holidays_df_for_future.rename(columns={'Date': 'timestamp'}, inplace=True)
    holidays_df_for_future['timestamp'] = pd.to_datetime(holidays_df_for_future['timestamp'].astype(str).str.split('T').str[0])
    holidays_df_for_future['is_holiday'] = 1
    holidays_df_for_future.drop_duplicates(subset=['timestamp']).to_csv('artifacts/holidays.csv', index=False)
    print("   -> Saved holidays data.")

    # --- 7. Generate Predictions for the Main Batch ---
    print("\nStep 7: Generating future predictions for the batch...")
    predictions = model_handler.make_predictions(predictor, ts_data, holidays_df_for_future)
    print("   -> Predictions generated.")

    # --- 8. Calculate Inventory Recommendations ---
    print("\nStep 8: Calculating inventory recommendations (EOQ, Reorder Point)...")
    recommendations_df = inventory_calculator.generate_recommendations(predictions, item_to_class_map, config)
    final_recommendations = inventory_calculator.apply_business_rules(recommendations_df)
    print("   -> Inventory metrics calculated and business rules applied.")

    # --- 9. Save Results to MongoDB ---
    print("\nStep 9: Saving final recommendations to MongoDB...")
    trained_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_collection_name = f"inventory_recommendations_{trained_date}"
    data_loader.save_dataframe_to_mongo(final_recommendations, new_collection_name)
    print(f"   -> Saved new recommendations to collection: '{new_collection_name}'")

    # --- 10. Log and Monitor Model ---
    print("\nStep 10: Logging model run and checking performance...")
    model_monitor.log_model_run(
        predictor=predictor,
        collection_name=new_collection_name,
        performance_metrics={"MASE": metrics.get("MASE", 999)},
        data_snapshot_info={"data_rows": len(processed_data)},
        trigger_source="manual_main_run"
    )
    
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("MONGO_DB")]
    is_better = model_monitor.compare_model_performance(db, new_collection_name)
    if not is_better:
        print("   -> WARNING: New model underperforms the previous model. Rolling back.")
        model_monitor.rollback_to_previous_version(db)
    else:
        print("   -> New model performance is acceptable.")
    client.close()
    
    print("-" * 60)
    print("✅ Unified Forecasting Pipeline Finished Successfully!")

if __name__ == "__main__":
    load_dotenv()
    main()
