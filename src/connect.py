from pymongo import MongoClient
from config import mongo_url

class Connect(object):
    @staticmethod    
    def get_connection():
        return MongoClient(mongo_url)

    def close():
        MongoClient.close()