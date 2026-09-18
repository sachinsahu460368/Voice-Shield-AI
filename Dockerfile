FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies for audio libraries (no apt FFmpeg — using imageio-ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend code (includes models/ and debug.wav)
COPY backend/ .

# ---------- Environment ----------
# Railway injects $PORT at runtime; default to 8000 for local testing
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# PyTorch CPU thread limit (Railway containers are typically 1-2 vCPU)
ENV TORCH_THREADS=2

# Expose port (documentation only; Railway uses $PORT)
EXPOSE 8000

# Run with Gunicorn + UvicornWorker, binding to 0.0.0.0:$PORT
CMD exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT main:app --timeout 120
