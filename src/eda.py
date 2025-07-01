import os
import pandas as pd
from autoviz.AutoViz_Class import AutoViz_Class
from dotenv import load_dotenv

# Project modules
import src.config as config
import graphs.data_loader as data_loader
import src.feature_engineering as feature_engineering

def run_eda_from_mongo(
    csv_output: str = "eda_data.csv",
    dep_var: str = "target",
    sep: str = ",",
    save_folder: str = "eda"
):
    print("Loading data for EDA...")
    # 1. Load raw data using our data_loader
    sales_df, inventory_df, holidays_df = data_loader.load_data(use_mongo=True)

    # 2. Prepare data via feature_engineering pipeline
    print("Preparing data with feature_engineering...")
    processed_data, _ = feature_engineering.prepare_data(
        source_data=sales_df,
        inventory_data=inventory_df,
        holidays_data=holidays_df,
        max_skus=config.MAX_SKUS
    )

    # Ensure the save_folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # 3. Write processed data to a CSV so AutoViz can read it
    processed_data.to_csv(csv_output, index=False, sep=sep)
    print(f"Saved processed data to {os.path.abspath(csv_output)}")

    # 4. Run AutoViz on the CSV
    print("Running AutoViz for automatic EDA...")
    av = AutoViz_Class()
    df_eda = av.AutoViz(
        filename=csv_output,
        sep=sep,
        depVar=dep_var,
        verbose=2,
        lowess=False,  # Disable LOWESS smoothing for faster performance
        chart_format="png",  # Save charts as PNG files
        save_plot_dir=save_folder  # FIX: Changed 'save_dir' to 'save_plot_dir'
    )
    print(f"EDA complete. Check the '{save_folder}' folder for charts.")
    return df_eda

if __name__ == "__main__":
    # Example usage
    load_dotenv()
    # Use an absolute path for the output CSV to avoid ambiguity
    eda_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda_data.csv")
    analyzed_df = run_eda_from_mongo(csv_output=eda_csv, dep_var="target", sep=",")
