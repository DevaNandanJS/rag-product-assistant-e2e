#!/bin/sh
set -e
echo "Checking Qdrant connectivity..."
python -m app.cli check
echo "Running idempotent ingestion check..."
python -m app.cli ingest --if-empty
echo "Starting Filumart Assistant server..."
exec python -m app.cli serve --host 0.0.0.0 --port 8000
