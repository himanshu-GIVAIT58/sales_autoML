
"""
This script loads data from MongoDB, processes it, and then runs AutoViz 
to generate and save exploratory data analysis (EDA) charts.
"""

import os
import pandas as pd
from autoviz.AutoViz_Class import AutoViz_Class
from dotenv import load_dotenv


import config
import data_loader
import feature_engineering


def run_eda_from_mongo(
    csv_output: str = "eda_data.csv",
    dep_var: str = "target",
    save_folder: str = "src/eda",
):
    """Loads, processes, and runs EDA on the data from MongoDB."""
    
    print("🚀 Starting EDA process...")

    
    print("Step 1/4: Loading data from MongoDB...")
    sales_df = data_loader.load_dataframe_from_mongo("sales_data")
    inventory_df = data_loader.load_dataframe_from_mongo("query_result")
    holidays_df = data_loader.load_dataframe_from_mongo("holidays_data")

    
    if sales_df.empty or inventory_df.empty:
        print("⚠️ Error: Sales or inventory data could not be loaded. Aborting EDA.")
        return

    
    print("Step 2/4: Preparing data with feature engineering...")
    processed_data, _ = feature_engineering.prepare_data(
        source_data=sales_df,
        inventory_data=inventory_df,
        holidays_data=holidays_df,
        max_skus=config.MAX_SKUS,
    )

    
    print(f"Step 3/4: Saving processed data to '{csv_output}'...")
    os.makedirs(save_folder, exist_ok=True)
    processed_data.to_csv(csv_output, index=False)
    print(f"   -> Data saved successfully.")

    
    print(f"Step 4/4: Running AutoViz to generate charts...")
    av = AutoViz_Class()
    av.AutoViz(
        filename=csv_output,
        sep=",",
        depVar=dep_var,
        verbose=2,
        lowess=False,
        chart_format="png",
        save_plot_dir=os.path.abspath(save_folder),
    )

    print(f"✅ EDA complete! Check the '{save_folder}' folder for charts.")


if __name__ == "__main__":
    load_dotenv()
    eda_csv_path = "eda_data.csv"
    run_eda_from_mongo(csv_output=eda_csv_path, dep_var="target")
