"""Main entry point for Filumart RAG Product Assistant.

Allows starting the application via:
    python main.py
or
    uvicorn main:app --reload
"""

import uvicorn

from app.api.app import create_app
from app.core.config import get_settings

app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    display_host = "localhost" if settings.HOST == "0.0.0.0" else settings.HOST
    print("\n" + "=" * 60)
    print(f"  Filumart RAG Assistant UI: http://{display_host}:{settings.PORT}")
    print(f"  API Docs (Swagger UI):     http://{display_host}:{settings.PORT}/docs")
    print("=" * 60 + "\n")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.APP_ENV == "dev",
    )
