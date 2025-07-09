import pandas as pd
import os
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from datetime import datetime
from dateutil.relativedelta import relativedelta


load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB", "sales_automl")

@contextmanager
def get_mongo_client(mongo_uri: str = MONGO_URI) -> MongoClient:
    """Provides a MongoDB client connection as a context manager."""
    client = None
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster')
        logger.info("MongoDB connection successful.")
        yield client
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed.")

def save_dataframe_to_mongo(df: pd.DataFrame, collection_name: str, mongo_uri: str = MONGO_URI, db_name: str = MONGO_DB_NAME):
    """Saves a DataFrame to a specified MongoDB collection, clearing old data first."""
    try:
        with get_mongo_client(mongo_uri) as client:
            db = client[db_name]
            collection = db[collection_name]
            
            logger.info(f"Clearing old data from collection '{collection_name}'...")
            collection.delete_many({})
            
            if not df.empty:
                records = df.to_dict("records")
                collection.insert_many(records)
                logger.info(f"✅ Successfully saved {len(records)} records to '{collection_name}'.")
            else:
                logger.info(f"DataFrame was empty. No new records saved to '{collection_name}'.")

    except (ConnectionFailure, OperationFailure) as e:
        logger.error(f"⚠️ Error saving data to MongoDB collection '{collection_name}': {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during save: {e}")

def load_dataframe_from_mongo(collection_name: str, query: Dict = None, mongo_uri: str = MONGO_URI, db_name: str = MONGO_DB_NAME) -> pd.DataFrame:
    """Loads data from a MongoDB collection into a pandas DataFrame based on a query."""
    try:
        with get_mongo_client(mongo_uri) as client:
            db = client[db_name]
            collection = db[collection_name]
            
            if query is None:
                query = {}
                
            cursor = collection.find(query, {"_id": 0})
            df = pd.DataFrame(list(cursor))
            logger.info(f"Successfully loaded {len(df)} records from '{collection_name}' for query: {query}")
            return df
    except (ConnectionFailure, OperationFailure) as e:
        logger.error(f"⚠️ Error loading data from '{collection_name}': {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during load: {e}")
    return pd.DataFrame()

def _clean_numeric_string(value: Any) -> float:
    """Removes commas from a string and converts it to a number, handling non-string types."""
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        try:
            return float(value.replace(',', ''))
        except ValueError:
            return 0.0
    elif isinstance(value, (int, float)):
        return float(value)
    return 0.0

def get_last_n_months_sales(
    sku_list: Optional[List[str]] = None,
    months_back: int = 6,
    quantity_col: str = 'qty',
    mongo_uri: str = MONGO_URI,
    db_name: str = MONGO_DB_NAME
) -> pd.DataFrame:
    """
    Correctly loads historical sales data by parsing the base SKU from the item_id
    and then filtering by date in pandas to handle string-based dates.
    """
    logger.info(f"Attempting to load sales data for item_ids: {sku_list} for the last {months_back} months.")
    
    query = {}
    if sku_list:
        base_skus = list(set([s.split('_')[0] for s in sku_list]))
        logger.info(f"Parsed base SKUs for query: {base_skus}")
        query["sku"] = {"$in": base_skus}
    
    df = load_dataframe_from_mongo("sales_data", query=query, mongo_uri=mongo_uri, db_name=db_name)
    
    if df.empty:
        logger.warning(f"No sales data found in the database for SKUs: {sku_list}.")
        return pd.DataFrame()

    logger.info("Cleaning and preparing loaded sales data...")
    
    if quantity_col in df.columns:
        df[quantity_col] = df[quantity_col].apply(_clean_numeric_string)
    else:
        logger.error(f"Column '{quantity_col}' not found in the sales data.")
        return pd.DataFrame()
        
    if 'created_at' in df.columns:
        df['timestamp'] = pd.to_datetime(df['created_at'], format='%d/%m/%Y', errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
    else:
        logger.error("Column 'created_at' not found in the sales data.")
        return pd.DataFrame()

    if df.empty:
        logger.warning("No valid data remaining after cleaning and date parsing.")
        return pd.DataFrame()

    latest_date = df['timestamp'].max()
    start_date = latest_date - relativedelta(months=months_back)
    
    logger.info(f"Latest data point for selected SKUs is on: {latest_date.date()}. Filtering from: {start_date.date()} onwards.")
    
    recent_sales_df = df[df['timestamp'] >= start_date].copy()
    
    logger.info(f"Found {len(recent_sales_df)} records in the last {months_back} months for the selected SKUs.")
    
    return recent_sales_df

def get_latest_recommendation_collection(mongo_uri: str = MONGO_URI, db_name: str = MONGO_DB_NAME) -> Optional[str]:
    """Finds the most recent 'inventory_recommendations' collection name."""
    try:
        with get_mongo_client(mongo_uri) as client:
            db = client[db_name]
            collection_names = db.list_collection_names()
            
            recommendation_collections = sorted(
                [name for name in collection_names if name.startswith("inventory_recommendations_")],
                reverse=True
            )
            
            if recommendation_collections:
                latest_collection = recommendation_collections[0]
                logger.info(f"🔍 Found latest recommendation collection: '{latest_collection}'")
                return latest_collection
            else:
                logger.warning("⚠️ No inventory recommendation collections found.")
                return None
    except (ConnectionFailure, OperationFailure) as e:
        logger.error(f"⚠️ Error fetching latest collection name: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while listing collections: {e}")
    return None

def load_latest_recommendation_data(mongo_uri: str = MONGO_URI, db_name: str = MONGO_DB_NAME) -> pd.DataFrame:
    """Loads data from the most recent recommendation collection."""
    latest_collection_name = get_latest_recommendation_collection(mongo_uri, db_name)
    if not latest_collection_name:
        return pd.DataFrame()
    return load_dataframe_from_mongo(latest_collection_name, mongo_uri=mongo_uri, db_name=db_name)


def load_product_prices(mongo_uri, db_name, collection_name="sales_data"):
    print(f"Loading product prices from '{collection_name}'...")
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        
        data = list(collection.find({}, {'sku': 1, 'revenue': 1, 'disc': 1, 'qty': 1}))
        if not data:
            print("Warning: No sales data found to calculate prices.")
            return {}

        df = pd.DataFrame(data)

        
        
        for col in ['revenue', 'disc']:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['qty'] = pd.to_numeric(df['qty'], errors='coerce')

        
        df.dropna(subset=['revenue', 'disc', 'qty'], inplace=True)
        
        
        df = df[df['qty'] > 0]

        
        df['price'] = (df['revenue'] - df['disc']) / df['qty']

        
        
        avg_prices = df.groupby('sku')['price'].mean()

        print(f"-> Successfully calculated average prices for {len(avg_prices)} SKUs.")
        client.close()
        
        
        return avg_prices.to_dict()

    except Exception as e:
        print(f"❌ Critical Error: Failed to load product prices. Error: {e}")
        return {}
