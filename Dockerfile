# Use lightweight Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=10000 \
    KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

# Install system audio libraries and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install (requirements.txt contains --extra-index-url for CPU torch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port (Render uses $PORT, defaults to 10000)
EXPOSE 10000

# Start FastAPI application with Uvicorn
CMD ["sh", "-c", "uvicorn src.backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
