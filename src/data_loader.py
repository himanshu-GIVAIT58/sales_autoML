# data_loader.py
"""
Handles loading all necessary data from source CSV files or MongoDB.
"""

import pandas as pd
import sys
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# --- Environment Variables ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "sales_automl")

def save_dataframe_to_mongo(df: pd.DataFrame, collection_name: str, mongo_uri: str = MONGO_URI, db_name: str = MONGO_DB):
    """Saves a DataFrame to a specified MongoDB collection, clearing old data first."""
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        # Clear old data before inserting new data
        collection.delete_many({})
        
        # Insert new data if the DataFrame is not empty
        if not df.empty:
            collection.insert_many(df.to_dict("records"))
        
        print(f"✅ Successfully saved {len(df)} records to '{collection_name}'.")

    except Exception as e:
        print(f"⚠️ Error saving data to MongoDB: {e}")
    finally:
        if 'client' in locals() and client:
            client.close()

def load_dataframe_from_mongo(collection_name: str, mongo_uri: str = MONGO_URI, db_name: str = MONGO_DB) -> pd.DataFrame:
    """Loads a full collection from MongoDB into a pandas DataFrame."""
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        df = pd.DataFrame(list(collection.find()))
        
        # Drop the MongoDB-specific '_id' column if it exists
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
            
        return df

    except Exception as e:
        print(f"⚠️ Error loading data from '{collection_name}': {e}")
        return pd.DataFrame() # Return an empty DataFrame on error
    finally:
        if 'client' in locals() and client:
            client.close()

def get_latest_recommendation_collection(db) -> Optional[str]:
    """
    Finds the latest inventory recommendation collection name.
    
    Collection names are expected in the format: "inventory_recommendations_<YYYYMMDD_HHMMSS>"
    """
    try:
        collection_names = db.list_collection_names()
        
        # Filter for recommendation collections and sort them to find the latest
        recommendation_collections = sorted(
            [name for name in collection_names if name.startswith("inventory_recommendations_")],
            reverse=True
        )
        
        if recommendation_collections:
            print(f"🔍 Found latest collection: '{recommendation_collections[0]}'")
            return recommendation_collections[0]
        else:
            print("⚠️ No inventory recommendation collections found.")
            return None # Return None object, not the string 'None'

    except Exception as e:
        print(f"⚠️ Error fetching latest collection name: {e}")
        return None

def load_latest_recommendation_data(mongo_uri: str = MONGO_URI, db_name: str = MONGO_DB) -> pd.DataFrame:
    """Loads data from the most recent recommendation collection in MongoDB."""
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        
        latest_collection_name = get_latest_recommendation_collection(db)
        
        # If no collection was found, return an empty DataFrame
        if not latest_collection_name:
            return pd.DataFrame()
            
        # Load data from the identified collection
        df = load_dataframe_from_mongo(latest_collection_name, mongo_uri, db_name)
        return df

    except Exception as e:
        print(f"⚠️ An error occurred while loading the latest recommendation data: {e}")
        return pd.DataFrame()
    finally:
        if 'client' in locals() and client:
            client.close()
