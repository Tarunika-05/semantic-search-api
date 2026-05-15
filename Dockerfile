# ─────────────────────────────────────────────────────────────────────
# Base image: Python 3.10 slim
# We use slim over alpine because sentence-transformers and faiss
# require compiled C extensions that are painful to build on alpine.
# slim gives us a small image without the build complexity.
# ─────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# ─────────────────────────────────────────────────────────────────────
# Create a non-root user for security (Production Best Practice)
# Running as root inside a container is a security risk.
# We create 'appuser' and will switch to it before running the app.
# ─────────────────────────────────────────────────────────────────────
RUN useradd -m -s /bin/bash appuser

# ─────────────────────────────────────────────────────────────────────
# Install system dependencies
# libgomp1 is required by faiss-cpu for OpenMP parallelism.
# Without it, faiss silently falls back to single-threaded mode.
# ─────────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
# This means pip install only reruns if requirements.txt changes,
# not on every code change — significantly speeds up rebuilds.
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of project and set ownership
COPY --chown=appuser:appuser . .

# Switch to the non-root user
USER appuser

# Expose port 8000
EXPOSE 8000

# ─────────────────────────────────────────────────────────────────────
# Start the FastAPI service
# --host 0.0.0.0 is required inside Docker so the port is accessible
# from outside the container. Default 127.0.0.1 would be unreachable.
# ─────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]