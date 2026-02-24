#!/bin/bash
# start.sh - Script to run both backend and frontend services

echo "Starting Backend (FastAPI)..."
cd /app/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

echo "Starting Celery Worker (FastAPI)..."
cd /app/backend
celery -A app.tasks.celery_app worker --loglevel=info --pool=solo -Q embeddings&

echo "Starting Frontend (Vite)..."
cd /app/frontend
npm run dev -- --host 0.0.0.0 --port 3000 &

# Wait for any process to exit. If one fails, the container stops.
wait -n

# Exit with the status of the process that failed
exit $?