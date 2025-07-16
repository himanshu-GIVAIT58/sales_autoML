import bz2
import json
from pymongo import MongoClient
from datetime import datetime
from src import dbConnect

MONGO_URI = dbConnect.connection_uri
DATABASE_NAME = dbConnect.mongo_db_name
OUTPUT_FILE = "mongo_data.json.bz2"

def custom_serializer(obj):
  
    if isinstance(obj, datetime):
        return obj.isoformat()  
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def export_collections_to_compressed_json():
    
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]

    
    all_data = {}

    
    collections = db.list_collection_names()
    print(f"Found collections: {collections}")

    for collection_name in collections:
        print(f"Exporting data from collection: {collection_name}")
        collection = db[collection_name]

        
        documents = list(collection.find({}, {"_id": 0}))  
        all_data[collection_name] = documents

    
    with bz2.BZ2File(OUTPUT_FILE, "w") as compressed_file:
        compressed_file.write(json.dumps(all_data, default=custom_serializer).encode("utf-8"))

    print(f"Data exported and compressed to {OUTPUT_FILE}")

if __name__ == "__main__":
    export_collections_to_compressed_json()
