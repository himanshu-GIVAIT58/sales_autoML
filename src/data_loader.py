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

MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "sales_automl")

def save_dataframe_to_mongo(df, collection_name, mongo_uri=MONGO_URI, db_name=MONGO_DB):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    collection.delete_many({})  # Clear old data

    # Split the DataFrame into smaller chunks (e.g., 1000 rows each) and insert
    chunk_size = 10000
    total_rows = len(df)
    for start in range(0, total_rows, chunk_size):
        end = start + chunk_size
        chunk = df.iloc[start:end]
        if not chunk.empty:
            collection.insert_many(chunk.to_dict("records"))

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

def get_latest_recommendation_collection(db) -> str:
    """
    Fetches the latest inventory recommendation collection based on the timestamp in the collection name.
    """
    try:
        # List all collections in the database
        collections = db.list_collection_names()

        # Filter collections that start with "inventory_recommendations_"
        recommendation_collections = [coll for coll in collections if coll.startswith("inventory_recommendations_")]

        # Sort collections by timestamp (assumes the format is "inventory_recommendations_<YYYYMMDD_HHMMSS>")
        recommendation_collections.sort(reverse=True)  # Latest first

        # Return the latest collection name
        if recommendation_collections:
            return recommendation_collections[0]
        else:
            print("⚠️ No inventory recommendation collections found in MongoDB.")
            return 'None'
    except Exception as e:
        print(f"⚠️ An error occurred while fetching the latest recommendation collection: {str(e)}")
        return 'None'

def load_latest_recommendation_data(mongo_uri=MONGO_URI, db_name=MONGO_DB) -> pd.DataFrame:
    """
    Loads the latest recommendation data from MongoDB.
    """
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]

        # Get the latest recommendation collection
        latest_collection = get_latest_recommendation_collection(db)
        if not latest_collection:
            return pd.DataFrame()

        # Load data from the latest collection
        collection = db[latest_collection]
        df = pd.DataFrame(list(collection.find()))
        client.close()

        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        return df
    except Exception as e:
        print(f"⚠️ An error occurred while loading the latest recommendation data: {str(e)}")
        return pd.DataFrame()

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

    df_eda = sales_data.copy()
    df_eda["order_date"] = pd.to_datetime(df_eda["order_date"])
    df_eda["year"] = df_eda["order_date"].dt.year
    df_eda["month"] = df_eda["order_date"].dt.month
    df_eda["day"] = df_eda["order_date"].dt.day
    df_eda["day_of_week"] = df_eda["order_date"].dt.day_name()

    return df_eda

