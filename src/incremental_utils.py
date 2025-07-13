import pandas as pd
import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import os

def get_skus_to_train(processed_data: pd.DataFrame) -> list[str]:
    load_dotenv()
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017/"))
    db = client[os.getenv("MONGO_DB", "sales_automl")]
    
    
    sku_latest = processed_data.groupby("item_id")["timestamp"].max().reset_index()
    
    
    status_docs = list(db.sku_training_status.find({}))
    status_map = {d["sku"]: pd.to_datetime(d["last_trained"]) for d in status_docs}

    skus_to_retrain = []
    for _, row in sku_latest.iterrows():
        sku = row["item_id"]
        latest_ts = pd.to_datetime(row["timestamp"])

        if sku not in status_map:
            
            skus_to_retrain.append(sku)
        elif latest_ts > status_map[sku]:
            
            skus_to_retrain.append(sku)
        
    
    return skus_to_retrain


def update_sku_training_status(trained_skus: list[str], timestamp: datetime.datetime | None = None):
    """
    Updates the last_trained timestamp of SKUs after successful training.
    """
    load_dotenv()
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017/"))
    db = client[os.getenv("MONGO_DB", "sales_automl")]

    now = timestamp or datetime.datetime.utcnow()

    for sku in trained_skus:
        db.sku_training_status.update_one(
            {"sku": sku},
            {"$set": {"last_trained": now.isoformat()}},
            upsert=True
        )
