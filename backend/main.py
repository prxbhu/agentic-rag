"""
FastAPI application entry point
"""
import logging
from contextlib import asynccontextmanager
import uvicorn
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.database import get_db_session
from sqlalchemy import text

from app.config import settings
from app.database import init_db, close_db
from app.services.hardware import HardwareDetector

# Import and include routers
from app.api import resources, conversations, health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for FastAPI application"""
    # Startup
    logger.info("Starting application...")
    
    # Initialize database
    await init_db()
    
    # Log system information
    system_info = HardwareDetector.get_system_info()
    logger.info(f"System Info: {system_info}")
    
    # Warm up embedding service
    from app.services.embedding import embedding_service
    logger.info(f"Embedding model info: {embedding_service.get_model_info()}")
    
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    await close_db()
    logger.info("Application shut down complete")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoints
@app.get("/api/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "system": HardwareDetector.get_system_info()
    }


@app.get("/api/health/db")
async def db_health_check():
    """Database health check"""
    
    
    try:
        async with get_db_session() as db:
            await db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "database": "disconnected", "error": str(e)}
        )


@app.get("/api/health/ollama")
async def ollama_health_check():
    """Ollama service health check"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return {
                    "status": "healthy",
                    "ollama": "connected",
                    "models": [m["name"] for m in models]
                }
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "ollama": "disconnected", "error": str(e)}
        )


app.include_router(resources.router, prefix="/api/resources", tags=["Resources"])
app.include_router(conversations.router, prefix="/api/conversations", tags=["Conversations"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/api/health"
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle uncaught exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")