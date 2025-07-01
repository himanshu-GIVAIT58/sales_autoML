# data_loader.py
"""
Handles loading all necessary data from source CSV files or MongoDB.
"""

import pandas as pd
import sys
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "sales_automl")

def save_dataframe_to_mongo(df, collection_name, mongo_uri=MONGO_URI, db_name=MONGO_DB):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    collection.delete_many({})  # Clear old data
    if not df.empty:
        collection.insert_many(df.to_dict("records"))
    client.close()

def load_dataframe_from_mongo(collection_name, mongo_uri=MONGO_URI, db_name=MONGO_DB):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    df = pd.DataFrame(list(collection.find()))
    client.close()
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    return df

def load_data(use_mongo=False):
    try:
        if use_mongo:
            # Use the correct collection names as per your MongoDB
            sales_data = load_dataframe_from_mongo("sales_data")
            print(f"Loaded 'sales_data' from MongoDB successfully.")
            inventory_data = load_dataframe_from_mongo("query_result")
            print(f"Loaded 'query_result' (inventory) from MongoDB successfully.")
            holidays_data = load_dataframe_from_mongo("holidays_data")
            print(f"Loaded 'holidays_data' from MongoDB successfully.")
        return sales_data, inventory_data, holidays_data

    except FileNotFoundError as e:
        print(f"Error: A required file was not found: {e.filename}")
        print("Please ensure all data files are in the correct directory.")
        sys.exit(1) # Exit the script with an error code

