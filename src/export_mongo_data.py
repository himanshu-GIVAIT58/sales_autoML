import bz2
import json
from pymongo import MongoClient
from datetime import datetime

# MongoDB connection details
MONGO_URI = "mongodb://root:example@localhost:27017/"
DATABASE_NAME = "sales_automl"
OUTPUT_FILE = "mongo_data.json.bz2"  # Compressed output file

def custom_serializer(obj):
    """
    Custom serializer for non-serializable objects like datetime.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to ISO 8601 string
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def export_collections_to_compressed_json():
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]

    # Dictionary to store all collections and their data
    all_data = {}

    # Fetch all collection names
    collections = db.list_collection_names()
    print(f"Found collections: {collections}")

    for collection_name in collections:
        print(f"Exporting data from collection: {collection_name}")
        collection = db[collection_name]

        # Fetch all documents from the collection
        documents = list(collection.find({}, {"_id": 0}))  # Exclude the `_id` field for compactness
        all_data[collection_name] = documents

    # Compress and save the data to a .json.bz2 file
    with bz2.BZ2File(OUTPUT_FILE, "w") as compressed_file:
        compressed_file.write(json.dumps(all_data, default=custom_serializer).encode("utf-8"))

    print(f"Data exported and compressed to {OUTPUT_FILE}")

if __name__ == "__main__":
    export_collections_to_compressed_json()
