from motor.motor_asyncio import AsyncIOMotorClient

from config import DATABASE_NAME, DATABASE_URL

MONGO_URI = DATABASE_URL
print(f"Connettendosi a MongoDB su: {MONGO_URI}")

DATABASE_NAME = DATABASE_NAME

client = AsyncIOMotorClient(MONGO_URI)
database = client[DATABASE_NAME]

# Collezione MongoDB
users_collection = database["users"]
