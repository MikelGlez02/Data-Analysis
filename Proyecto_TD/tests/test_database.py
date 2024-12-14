# tests/test_database.py
from database.mongodb_handler import MongoDBHandler

def test_mongodb_integration():
    db_handler = MongoDBHandler(database_name="test_db", collection_name="test_collection")

    # Test insert and find
    test_data = {"key": "value"}
    db_handler.insert_one(test_data)
    retrieved_data = db_handler.find_one({"key": "value"})
    assert retrieved_data["key"] == "value", "MongoDB integration test failed."
    print("Database integration test passed.")
