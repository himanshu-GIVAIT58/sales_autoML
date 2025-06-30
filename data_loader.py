# data_loader.py
"""
Handles loading all necessary data from source CSV files.
"""

import pandas as pd
import sys

def load_data(source_file, inventory_file, holidays_file):
    """
    Loads sales, inventory, and holiday data from specified CSV files.

    Args:
        source_file (str): Path to the sales data CSV.
        inventory_file (str): Path to the inventory data CSV.
        holidays_file (str): Path to the holidays data CSV.

    Returns:
        tuple: A tuple containing three pandas DataFrames:
               (sales_data, inventory_data, holidays_data).
               Exits the program if a file is not found.
    """
    try:
        sales_data = pd.read_csv(source_file)
        print(f"Loaded '{source_file}' successfully.")

        inventory_data = pd.read_csv(inventory_file)
        print(f"Loaded '{inventory_file}' successfully.")

        holidays_data = pd.read_csv(holidays_file)
        print(f"Loaded '{holidays_file}' successfully.")

        return sales_data, inventory_data, holidays_data

    except FileNotFoundError as e:
        print(f"Error: A required file was not found: {e.filename}")
        print("Please ensure all data files are in the correct directory.")
        sys.exit(1) # Exit the script with an error code

