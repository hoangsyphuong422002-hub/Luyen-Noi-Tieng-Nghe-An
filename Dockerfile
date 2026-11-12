# Use lightweight Python 3.10 image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

# Install system audio libraries and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only (drastically reduces image size from 2GB to ~180MB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port (7860 is default for Hugging Face Spaces; Render/Railway use $PORT)
EXPOSE 7860

# Start FastAPI application with Uvicorn
CMD ["sh", "-c", "uvicorn src.backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
