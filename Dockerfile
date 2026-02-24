FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libpq-dev \
    libmagic1 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY backend/requirements.txt ./backend/
RUN uv pip install --system --no-cache-dir -r backend/requirements.txt

COPY frontend/package.json frontend/package-lock.json* ./frontend/
WORKDIR /app/frontend
RUN npm install

WORKDIR /app
COPY backend ./backend
COPY frontend ./frontend

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000 3000

CMD ["/app/start.sh"]