import datetime
import os
import pandas as pd
from dotenv import load_dotenv
from autogluon.timeseries import TimeSeriesDataFrame
import joblib
from src import incremental_utils, config, data_loader, feature_engineering, model_handler, inventory_calculator, model_monitor

def main():
    print("🚀 Starting Unified Forecasting Pipeline...")
    print("-" * 60)

    # --- Step 1: Data Loading ---
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
        feedback_df = data_loader.load_dataframe_from_mongo("feedback_data")
        print("   -> Successfully loaded inventory and holidays data and feedback data.")
    except Exception as e:
        print(f"❌ Critical Error: Failed to load data. Aborting. Error: {e}")
        return

    # --- Step 2: Feature Engineering ---
    print("\nStep 2: Preparing data and running feature engineering...")
    print(f"   -> Processing data for a maximum of {config.MAX_SKUS} SKUs.")
    processed_data, static_features_base = feature_engineering.prepare_data(
        sales_df, inventory_df, holidays_df, config.MAX_SKUS
    )

    processed_data = feature_engineering.add_feedback_features(processed_data, feedback_df)

    print(f"   -> Generated {len(processed_data)} rows after feature engineering.")
    data_loader.save_dataframe_to_mongo(processed_data, "processed_training_data")
    print("   -> Saved processed training data snapshot to MongoDB.")

    # --- Step 3: ABC Analysis and Data Filtering ---
    print("\nStep 3: Performing ABC Analysis and data filtering...")
    print(f"   -> Using ABC class cutoffs: A={config.ABC_CONFIG['A_class_percentage']*100}%, B={config.ABC_CONFIG['B_class_percentage']*100}%")
    total_sales = processed_data.groupby('sku')['target'].sum().sort_values(ascending=False)
    total_sales_df = total_sales.to_frame()
    a_cutoff = int(len(total_sales_df) * config.ABC_CONFIG['A_class_percentage'])
    b_cutoff = a_cutoff + int(len(total_sales_df) * config.ABC_CONFIG['B_class_percentage'])
    total_sales_df['class'] = 'C'
    total_sales_df.loc[total_sales_df.index[:a_cutoff], 'class'] = 'A'
    total_sales_df.loc[total_sales_df.index[a_cutoff:b_cutoff], 'class'] = 'B'
    item_to_class_map = total_sales_df['class'].to_dict()
    class_counts = total_sales_df['class'].value_counts()
    print(f"   -> ABC Analysis Results: Class A={class_counts.get('A', 0)}, Class B={class_counts.get('B', 0)}, Class C={class_counts.get('C', 0)}")
    
    print(f"   -> Filtering for items with history > prediction length ({config.PREDICTION_LENGTH} days).")
    item_counts = processed_data["item_id"].value_counts()
    items_with_sufficient_data = item_counts[item_counts > config.PREDICTION_LENGTH].index
    data_with_history = processed_data[processed_data["item_id"].isin(items_with_sufficient_data)]
    skus_to_train = incremental_utils.get_skus_to_train(data_with_history)
    
    if not skus_to_train:
        print("✅ All SKUs up to date. No retraining needed.")
        return
    
    print(f"   -> Number of SKU-channel combos with sufficient history: {len(items_with_sufficient_data)}")
    print(f"   -> SKUs to retrain (new data detected): {len(skus_to_train)}")
    filtered_data = data_with_history[data_with_history["item_id"].isin(skus_to_train)]

    # --- Step 4: Preparing Data for AutoGluon ---
    print("\nStep 4: Preparing data for AutoGluon TimeSeriesDataFrame...")
    for col in config.KNOWN_COVARIATES_NAMES:
        if col not in filtered_data.columns:
            filtered_data[col] = 0
    final_static_features = static_features_base.reset_index()
    print(f"   -> Preparing data with {len(final_static_features.columns) - 1} static features.")
    ts_data = TimeSeriesDataFrame.from_data_frame(
        filtered_data,
        id_column="item_id",
        timestamp_column="timestamp",
        static_features_df=final_static_features
    )
    print("   -> TimeSeriesDataFrame created successfully.")

    # --- Step 5: Model Training and Evaluation ---
    print("\nStep 5: Training or fine-tuning the model...")
    # MODIFIED: Call the new training function
    predictors = model_handler.train_multiple_predictors(ts_data, config)
    print("   -> All model presets trained. Evaluating...")
    # MODIFIED: Evaluate all trained predictors
    leaderboard_df = model_handler.evaluate_predictors(predictors, ts_data)

    # To get a single metric for the validation pipeline, we can use the best score
    # from the best preset. Let's assume the last preset in the list is the "best".
    best_preset_name = config.TRAINING['PRESETS'][-1]
    best_predictor = predictors[best_preset_name]
    new_performance_metrics = best_predictor.evaluate(ts_data)
    if new_performance_metrics is None:
        new_performance_metrics = {'MASE': 999}
    
    mase = new_performance_metrics.get('MASE', 'N/A')
    if mase != 'N/A':
        print(f"   -> 📈 Best preset '{best_preset_name}' evaluation metrics: MASE={mase:.4f}")
    else:
        print(f"   -> 📈 Best preset '{best_preset_name}' evaluation metrics: MASE={mase}")

    # ====================================================================
    # --- NEW: Model Validation Gate ---
    # ====================================================================
    
    print("\nStep 6: Running Model Validation Pipeline...")
    unique_run_id = f"inventory_recommendations_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # The validation pipeline can still run on your highest-quality predictor
    validation_result = model_monitor.run_model_validation_pipeline(
        new_predictor=best_predictor,
        new_training_data=processed_data,
        new_performance_metrics=new_performance_metrics
    )

    print("\nStep 7: Logging model run with validation results...")
    model_monitor.log_model_run(
        predictor=best_predictor,
        collection_name=unique_run_id,
        performance_metrics=new_performance_metrics,
        validation_result=validation_result,
        data_snapshot_info={"collection_name": "processed_training_data", "record_count": len(processed_data)},
        trigger_source="scheduled_main_run"
    )

    # --- Step 8: Act on Validation Decision ---
    print("\nStep 8: Acting on validation decision...")
    if validation_result.decision == "promote":
        print("  -> ✅ Model promoted. Generating ensembled artifacts and recommendations...")
        
        # Save artifacts for the newly promoted model
        os.makedirs(config.ARTIFACTS_PATH, exist_ok=True)
        static_feature_columns = list(final_static_features.drop(columns=['item_id']).columns)
        joblib.dump(static_feature_columns, os.path.join(config.ARTIFACTS_PATH, 'static_feature_columns.joblib'))
        print("  -> Saved static feature columns.")
        
        holidays_df_for_future = data_loader.load_dataframe_from_mongo("holidays_data")
        holidays_df_for_future.rename(columns={'Date': 'timestamp'}, inplace=True)
        holidays_df_for_future['timestamp'] = pd.to_datetime(holidays_df_for_future['timestamp'].astype(str).str.split('T').str[0])
        holidays_df_for_future['is_holiday'] = 1
        holidays_df_for_future.drop_duplicates(subset=['timestamp']).to_csv(os.path.join(config.ARTIFACTS_PATH, 'holidays.csv'), index=False)
        print("  -> Saved holidays data.")

        # MODIFIED: Generate and save new recommendations using ensembled predictions
        print("  -> Generating ensembled predictions from all trained models...")
        ensembled_predictions = model_handler.make_ensembled_predictions(predictors, ts_data, holidays_df_for_future)
        
        # Use 'ensembled_predictions' instead of 'predictions' from here on
        recommendations_df = inventory_calculator.generate_recommendations(ensembled_predictions, item_to_class_map, config)
        final_recommendations = inventory_calculator.apply_business_rules(recommendations_df)
        print(f"  -> Generated {len(final_recommendations)} recommendation rows using ensemble predictions.")
        data_loader.save_dataframe_to_mongo(final_recommendations, unique_run_id)
        print(f"  -> Saved new recommendations to collection: '{unique_run_id}'")
    else:
        print(f"  -> ❌ Model rejected. Reasons: {validation_result.reasons}")
        print("  -> No new recommendations will be generated. The champion model remains active.")

    # --- Step 9: Cleanup and Version Management ---
    print("\nStep 9: Performing cleanup and version management...")
    model_monitor.manage_recommendation_versions(
        decision=validation_result.decision,
        new_collection_name=unique_run_id
    )

    print("-" * 60)
    print("✅ Unified Forecasting Pipeline Finished Successfully!")

if __name__ == "__main__":
    load_dotenv()
    main()
