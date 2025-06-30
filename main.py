# main.py
"""
Main orchestration script for the inventory forecasting pipeline.
This script imports modules and executes the end-to-end process:
1. Load Configuration
2. Load Data
3. Engineer Features
4. Train Model
5. Generate Predictions
6. Calculate Inventory Recommendations
7. Save Results
"""
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame

# Import project modules
import config
import data_loader
import feature_engineering
import model_handler
import inventory_calculator

def main():
    """Main function to run the entire pipeline."""
    
    # --- 1. Load Data ---
    sales_df, inventory_df, holidays_df = data_loader.load_data(
        config.SOURCE_FILENAME,
        config.INVENTORY_FILENAME,
        config.HOLIDAYS_FILENAME
    )

    # --- 2. Prepare Data and Engineer Features ---
    processed_data, static_features_base = feature_engineering.prepare_data(
        sales_df,
        inventory_df,
        holidays_df,
        config.MAX_SKUS
    )

    # --- 3. ABC Analysis & Data Filtering ---
    print("\nPerforming ABC Analysis...")
    total_sales = processed_data.groupby('sku')['target'].sum().sort_values(ascending=False)
    total_sales_df = total_sales.to_frame()
    
    a_cutoff = int(len(total_sales_df) * config.ABC_CONFIG['A_class_percentage'])
    b_cutoff = a_cutoff + int(len(total_sales_df) * config.ABC_CONFIG['B_class_percentage'])
    
    total_sales_df['class'] = 'C'
    total_sales_df.iloc[:a_cutoff, total_sales_df.columns.get_loc('class')] = 'A'
    total_sales_df.iloc[a_cutoff:b_cutoff, total_sales_df.columns.get_loc('class')] = 'B'
    item_to_class_map = total_sales_df['class'].to_dict()

    # Filter for items with enough historical data for modeling
    item_counts = processed_data["item_id"].value_counts()
    items_with_sufficient_data = item_counts[item_counts > config.PREDICTION_LENGTH].index
    filtered_data = processed_data[processed_data["item_id"].isin(items_with_sufficient_data)]
    print(f"\nNumber of SKU-channel combos with sufficient history: {len(items_with_sufficient_data)}")

    # --- 4. Prepare Data for AutoGluon ---
    # Ensure all required covariate columns exist
    for col in config.KNOWN_COVARIATES_NAMES:
        if col not in filtered_data.columns:
            filtered_data[col] = 0

    # Prepare static features, ensuring 'item_id' is a column
    id_sku_map = filtered_data[['item_id', 'sku']].drop_duplicates()
    final_static_features = pd.merge(id_sku_map, static_features_base, on='sku', how='left')
    final_static_features.drop(columns='sku', inplace=True)

    # Create the TimeSeriesDataFrame
    ts_data = TimeSeriesDataFrame.from_data_frame(
        filtered_data,
        id_column="item_id",
        timestamp_column="timestamp",
        static_features_df=final_static_features
    )

    # --- 5. Train and Evaluate Model ---
    predictor = model_handler.train_predictor(ts_data, config)
    model_handler.evaluate_predictor(predictor, ts_data)
    
    # --- 6. Generate Predictions ---
    # We need the original holidays dataframe for future feature generation
    _, _, holidays_df_for_future = data_loader.load_data(
        config.SOURCE_FILENAME, config.INVENTORY_FILENAME, config.HOLIDAYS_FILENAME
    )
    holidays_df_for_future.rename(columns={'Date': 'timestamp'}, inplace=True)
    holidays_df_for_future['timestamp'] = pd.to_datetime(holidays_df_for_future['timestamp'].astype(str).str.split('T').str[0])
    holidays_df_for_future['is_holiday'] = 1
    holidays_df_for_future = holidays_df_for_future.drop_duplicates(subset=['timestamp'])

    predictions = model_handler.make_predictions(predictor, ts_data, holidays_df_for_future)

    # --- 7. Calculate Inventory Recommendations ---
    recommendations_df = inventory_calculator.generate_recommendations(
        predictions, 
        item_to_class_map, 
        config
    )
    
    # Apply any final business logic
    final_recommendations = inventory_calculator.apply_business_rules(recommendations_df)

    # --- 8. Save and Display Results ---
    final_recommendations.to_csv(config.OUTPUT_FILENAME, index=False)
    print(f"\nInventory recommendations saved to '{config.OUTPUT_FILENAME}'")
    print("\n--- Sample of Final Inventory Recommendations ---")
    print(final_recommendations.head(15))
    if len(final_recommendations) > 15:
        print("...")
        print(final_recommendations.tail())

    # --- 9. Optional: Plotting ---
    print("\nGenerating plot of predictions for the first 3 items...")
    try:
        predictor.plot(
            data=ts_data,
            predictions=predictions,
            item_ids=ts_data.item_ids[:3],
            max_history_length=200,
        )
        print("Plot generated successfully.")
    except Exception as e:
        print(f"Could not generate plot. Error: {e}")


if __name__ == "__main__":
    main()

