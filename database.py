from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    client: AsyncIOMotorClient = None
    db: AsyncIOMotorDatabase = None

db_manager = DatabaseManager()

async def connect_to_mongo():
    logger.info(f"Connecting to MongoDB at {settings.MONGODB_URI}...")
    try:
        db_manager.client = AsyncIOMotorClient(settings.MONGODB_URI)
        db_manager.db = db_manager.client[settings.DATABASE_NAME]
        
        # Ping database to verify connection
        await db_manager.client.admin.command('ping')
        logger.info(f"Connected to MongoDB database: {settings.DATABASE_NAME}")
        
        # Ensure indexes
        await init_db_indexes()
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise e

async def close_mongo_connection():
    if db_manager.client:
        logger.info("Closing MongoDB connection...")
        db_manager.client.close()
        logger.info("MongoDB connection closed.")

async def init_db_indexes():
    """Create essential MongoDB indexes for optimal performance."""
    try:
        db = db_manager.db
        if db is not None:
            # Users: unique email
            await db.users.create_index("email", unique=True)
            # Sessions: user_id index and updated_at
            await db.research_sessions.create_index("user_id")
            await db.research_sessions.create_index("updated_at")
            # Messages: session_id index and timestamp
            await db.messages.create_index("session_id")
            await db.messages.create_index("timestamp")
            # Papers: session_id index
            await db.papers.create_index("session_id")
            # Hypotheses: session_id and user_id index
            await db.hypotheses.create_index("session_id")
            await db.hypotheses.create_index("user_id")
            # Experiments: hypothesis_id and session_id index
            await db.experiments.create_index("hypothesis_id")
            await db.experiments.create_index("session_id")
            logger.info("MongoDB indexes verified/created successfully.")
    except Exception as e:
        logger.warning(f"Note on creating indexes: {e}")

def get_db() -> AsyncIOMotorDatabase:
    if db_manager.db is None:
        db_manager.client = AsyncIOMotorClient(settings.MONGODB_URI)
        db_manager.db = db_manager.client[settings.DATABASE_NAME]
    return db_manager.db
