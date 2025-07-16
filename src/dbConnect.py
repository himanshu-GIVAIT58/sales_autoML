import os
import pymongo
from urllib.parse import quote_plus

# Get all credentials from environment variables set by Kubernetes
mongo_user = os.getenv("MONGO_USERNAME")
# Provide a default empty string to satisfy quote_plus
mongo_pass = quote_plus(os.getenv("MONGO_PASSWORD", ""))
mongo_host = os.getenv("MONGO_HOST")
mongo_port = int(os.getenv("MONGO_PORT", 27017))
mongo_db_name = os.getenv("MONGO_DB", "sales_automl")

# Build the full, correct connection string for Kubernetes
# The "?authSource=admin" part is added for proper authentication
connection_uri = f"mongodb://{mongo_user}:{mongo_pass}@{mongo_host}:{mongo_port}/{mongo_db_name}?authSource=admin"

# Use the dynamically built URI
client = pymongo.MongoClient(connection_uri)
db = client[mongo_db_name]
