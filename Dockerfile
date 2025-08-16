# Use a slim Python base image
FROM python:3.11-slim

# System deps for matplotlib, sqlite, and fonts
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy files
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Expose port (Hugging Face Spaces expects 7860 or 8000; we’ll use 7860)
ENV PORT=7860
ENV HOST=0.0.0.0

# Create a writable directory for SQLite & generated artifacts
RUN mkdir -p /app/store && chmod -R 777 /app/store

# Start the Flask app
CMD ["python", "app.py"]
