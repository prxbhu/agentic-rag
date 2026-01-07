"""
Health check endpoints for monitoring system status
"""
import logging
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from app.database import get_db
from app.config import settings
from app.services.hardware import HardwareDetector

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "service": settings.API_TITLE
    }


@router.get("/health/detailed")
async def detailed_health_check(db: AsyncSession = Depends(get_db)):
    """
    Detailed health check including all dependencies
    """
    health_status = {
        "status": "healthy",
        "version": settings.API_VERSION,
        "components": {}
    }
    
    # Check database
    try:
        await db.execute(text("SELECT 1"))
        health_status["components"]["database"] = {
            "status": "healthy",
            "message": "Connected"
        }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "message": str(e)
        }
    
    # Check Redis
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        health_status["components"]["redis"] = {
            "status": "healthy",
            "message": "Connected"
        }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["redis"] = {
            "status": "unhealthy",
            "message": str(e)
        }
    
    # Check Ollama
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.OLLAMA_BASE_URL}/api/tags",
                timeout=5.0
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                health_status["components"]["ollama"] = {
                    "status": "healthy",
                    "message": "Connected",
                    "models": [m["name"] for m in models]
                }
            else:
                health_status["status"] = "degraded"
                health_status["components"]["ollama"] = {
                    "status": "unhealthy",
                    "message": f"HTTP {response.status_code}"
                }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["ollama"] = {
            "status": "unhealthy",
            "message": str(e)
        }
    
    # Add system information
    health_status["system"] = HardwareDetector.get_system_info()
    
    # Determine overall status code
    if health_status["status"] == "unhealthy":
        return JSONResponse(status_code=503, content=health_status)
    elif health_status["status"] == "degraded":
        return JSONResponse(status_code=200, content=health_status)
    else:
        return health_status


@router.get("/health/db")
async def database_health(db: AsyncSession = Depends(get_db)):
    """Database-specific health check"""
    try:
        # Test basic query
        await db.execute(text("SELECT 1"))
        
        # Check table existence
        result = await db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """))
        tables = [row[0] for row in result.fetchall()]
        
        # Check pgvector extension
        result = await db.execute(text("""
            SELECT * FROM pg_extension WHERE extname = 'vector'
        """))
        has_pgvector = result.fetchone() is not None
        
        return {
            "status": "healthy",
            "database": "connected",
            "tables": tables,
            "pgvector_enabled": has_pgvector
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e)
            }
        )


@router.get("/health/ollama")
async def ollama_health():
    """Ollama service health check"""
    try:
        async with httpx.AsyncClient() as client:
            # Check if Ollama is running
            response = await client.get(
                f"{settings.OLLAMA_BASE_URL}/api/tags",
                timeout=5.0
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                
                # Check if required model is available
                model_names = [m["name"] for m in models]
                has_required_model = any(
                    settings.CHAT_MODEL in name 
                    for name in model_names
                )
                
                return {
                    "status": "healthy",
                    "ollama": "connected",
                    "base_url": settings.OLLAMA_BASE_URL,
                    "models": model_names,
                    "required_model": settings.CHAT_MODEL,
                    "required_model_available": has_required_model
                }
            else:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "ollama": "error",
                        "http_status": response.status_code
                    }
                )
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "ollama": "disconnected",
                "error": str(e)
            }
        )


@router.get("/health/redis")
async def redis_health():
    """Redis health check"""
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL)
        
        # Ping Redis
        r.ping()
        
        # Get info
        info = r.info()
        
        return {
            "status": "healthy",
            "redis": "connected",
            "version": info.get("redis_version"),
            "used_memory_human": info.get("used_memory_human"),
            "connected_clients": info.get("connected_clients")
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "redis": "disconnected",
                "error": str(e)
            }
        )


@router.get("/health/embedding")
async def embedding_service_health():
    """Embedding service health check"""
    try:
        from app.services.embedding import embedding_service
        
        # Get model info
        model_info = embedding_service.get_model_info()
        
        # Test embedding generation
        test_embedding = embedding_service.embed_text("test")
        
        return {
            "status": "healthy",
            "embedding_service": "operational",
            "model": model_info,
            "test_embedding_dimension": len(test_embedding)
        }
    except Exception as e:
        logger.error(f"Embedding service health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "embedding_service": "error",
                "error": str(e)
            }
        )


@router.get("/health/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """
    Readiness probe for Kubernetes/container orchestration
    Returns 200 only if all critical services are ready
    """
    try:
        # Check database
        await db.execute(text("SELECT 1"))
        
        # Check Redis
        import redis
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        
        # Check embedding service
        from app.services.embedding import embedding_service
        embedding_service.get_model_info()
        
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": str(e)}
        )


@router.get("/health/live")
async def liveness_check():
    """
    Liveness probe for Kubernetes/container orchestration
    Returns 200 if application is running (doesn't check dependencies)
    """
    return {"status": "alive"}