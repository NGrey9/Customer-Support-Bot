import os
import datetime

from pymongo import MongoClient

from dotenv import load_dotenv
load_dotenv()


MONGODB_URI = os.environ['MONGODB_URI']
DB_NAME = os.environ['DB_NAME']
COLLECTION = os.environ['COLLECTION']


class ChatHistoryManager:
    def __init__(self, db_uri: str = MONGODB_URI,
                 db_name: str = DB_NAME,
                 collection: str = COLLECTION):
        self.client = MongoClient(db_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection]

    def append_message(self, session_id: str, user_id: str, sender: str, message: str):
        chat_entry = {
            "session_id": session_id,
            "user_id": user_id,
            "sender": sender,
            "message": message,
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
        }
        result = self.collection.insert_one(chat_entry)
        return result.inserted_id

    def get_history(self, session_id: str):
        messages = list(self.collection.find(
            {"session_id": session_id}).sort("timestamp", 1))
        return messages

    def close(self):
        self.client.close()
