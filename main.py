import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import settings
from app.config.database import connect_to_mongo, close_mongo_connection

# Import routes
from app.routes.auth import router as auth_router
from app.routes.sessions import router as sessions_router
from app.routes.chat import router as chat_router
from app.routes.hypotheses import router as hypotheses_router
from app.routes.experiments import router as experiments_router
from app.routes.literature import router as literature_router
from app.routes.papers import router as papers_router
from app.routes.validation import router as validation_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hypothesis_generator")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing AI Scientific Hypothesis Generator Backend...")
    await connect_to_mongo()
    yield
    await close_mongo_connection()
    logger.info("Shutdown completed.")

app = FastAPI(
    title="AI Scientific Hypothesis Generator API",
    description="Agentic scientific research assistant powered by Gemini and LangGraph",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
origins = settings.cors_origins_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(auth_router)
app.include_router(sessions_router)
app.include_router(chat_router)
app.include_router(hypotheses_router)
app.include_router(experiments_router)
app.include_router(literature_router)
app.include_router(papers_router)
app.include_router(validation_router)

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "AI Scientific Hypothesis Generator",
        "version": "1.0.0",
        "gemini_active": bool(settings.GEMINI_API_KEY)
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "database": settings.DATABASE_NAME,
        "llm": settings.GEMINI_MODEL
    }
