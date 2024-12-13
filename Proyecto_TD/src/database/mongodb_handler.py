# database/mongodb_handler.py
from pymongo import MongoClient

class MongoDBHandler:
    def __init__(self, database_name, collection_name, uri="mongodb://localhost:27017/"):
        self.client = MongoClient(uri)
        self.database = self.client[database_name]
        self.collection = self.database[collection_name]

    def insert_one(self, document):
        """Insert a single document into the collection."""
        return self.collection.insert_one(document)

    def insert_many(self, documents):
        """Insert multiple documents into the collection."""
        return self.collection.insert_many(documents)

    def find_one(self, query):
        """Find a single document matching the query."""
        return self.collection.find_one(query)

    def find_all(self):
        """Find all documents in the collection."""
        return list(self.collection.find())

    def delete_one(self, query):
        """Delete a single document matching the query."""
        return self.collection.delete_one(query)

    def delete_many(self, query):
        """Delete multiple documents matching the query."""
        return self.collection.delete_many(query)

    def update_one(self, query, update_values):
        """Update a single document matching the query."""
        return self.collection.update_one(query, {"$set": update_values})

    def update_many(self, query, update_values):
        """Update multiple documents matching the query."""
        return self.collection.update_many(query, {"$set": update_values})

