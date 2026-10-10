"""Application module entry point for Filumart RAG Product Assistant.

Provides:
    from app.main import app
or
    uvicorn app.main:app
"""

from app.api.app import create_app

app = create_app()
