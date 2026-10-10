FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/models

# Install system dependencies and Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pinned Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source tree and data
COPY app ./app
COPY scripts ./scripts
COPY frontend ./frontend
COPY data ./data
COPY eval ./eval
COPY docker/entrypoint.sh /entrypoint.sh

# Fix line endings and permissions on entrypoint
RUN chmod +x /entrypoint.sh

# Pre-download and bake embedding/reranker models into Docker layer (Rule R2)
RUN python -m app.cli download-models

# Create non-root application user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /opt/models

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

ENTRYPOINT ["/entrypoint.sh"]
