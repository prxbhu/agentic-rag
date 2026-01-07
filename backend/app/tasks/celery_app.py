"""
Celery application configuration
"""
import logging
from celery import Celery
from celery.schedules import crontab

from app.config import settings

logger = logging.getLogger(__name__)

# Create Celery instance
celery_app = Celery(
    "rag_tasks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.embedding_tasks"]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution settings
    task_acks_late=True,  # Acknowledge tasks after completion
    task_reject_on_worker_lost=True,
    task_track_started=True,
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Prevent worker from prefetching too many tasks
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks to prevent memory leaks
    
    # Task routing
    task_routes={
        "app.tasks.embedding_tasks.*": {"queue": "embeddings"},
    },
    
    # Rate limiting
    task_annotations={
        "app.tasks.embedding_tasks.generate_embeddings_task": {
            "rate_limit": "10/m"  # Max 10 embedding tasks per minute
        }
    },
    
    # Scheduled tasks (Celery Beat)
    beat_schedule={
        "cleanup-old-tasks": {
            "task": "app.tasks.embedding_tasks.cleanup_old_tasks",
            "schedule": crontab(hour=2, minute=0),  # Run daily at 2 AM
        },
        "reindex-chunks": {
            "task": "app.tasks.embedding_tasks.reindex_chunks",
            "schedule": crontab(day_of_week=0, hour=3, minute=0),  # Run weekly on Sunday at 3 AM
        }
    }
)

# Event handlers
@celery_app.task(bind=True)
def debug_task(self):
    """Debug task to verify Celery is working"""
    logger.info(f"Request: {self.request!r}")
    return "Celery is working!"


# Configure logging for Celery
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Setup periodic tasks after Celery configuration"""
    logger.info("Celery configured with beat schedule")


# Error handling
@celery_app.task(bind=True, max_retries=3)
def error_handler(self, uuid):
    """Handle task errors"""
    try:
        # Task-specific error handling logic
        logger.error(f"Task {uuid} failed")
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)