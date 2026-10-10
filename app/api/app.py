"""FastAPI application factory, lifespan management, and global error handlers.

Provides:
- create_app(): Factory function initializing routes, static files, and exception handlers.
- lifespan(): Async context manager handling startup/shutdown for models and stores.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.generation.cache import GenerationCache
from app.generation.router import build_router_from_settings
from app.generation.service import GenerationService
from app.retrieval.embedder import FastEmbedEmbedder
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.rerank import CrossEncoderReranker
from app.retrieval.store import QdrantStore

logger = logging.getLogger("app.api.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    Initializes FastEmbed embedders, Qdrant store, cross-encoder reranker,
    multi-provider LLM router, and the integrated GenerationService.
    Preserves any pre-existing dependencies on app.state (e.g., test mocks).
    """
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = settings

    if not hasattr(app.state, "store") or app.state.store is None:
        try:
            embedder = getattr(app.state, "embedder", None)
            if embedder is None:
                embedder = FastEmbedEmbedder(
                    model_name=settings.EMBEDDING_MODEL,
                    cache_dir=settings.FASTEMBED_CACHE_PATH,
                )
                app.state.embedder = embedder

            store = QdrantStore(settings=settings, embedder=embedder)
            app.state.store = store
        except Exception as exc:
            logger.warning("[lifespan] Vector store initialization notice: %s", exc)
            app.state.store = None

    if not hasattr(app.state, "router") or app.state.router is None:
        try:
            router_instance = build_router_from_settings(settings)
            app.state.router = router_instance
        except Exception as exc:
            logger.warning("[lifespan] LLM router initialization notice: %s", exc)
            app.state.router = None

    if not hasattr(app.state, "service") or app.state.service is None:
        try:
            embedder = getattr(app.state, "embedder", None)
            store = getattr(app.state, "store", None)
            reranker = getattr(app.state, "reranker", None)
            if reranker is None and settings.RERANKER_MODEL:
                try:
                    reranker = CrossEncoderReranker(model_name=settings.RERANKER_MODEL)
                except Exception as exc:
                    logger.warning("[lifespan] Reranker model load notice: %s", exc)
                    reranker = None
            app.state.reranker = reranker

            pipeline = getattr(app.state, "pipeline", None)
            if pipeline is None and store is not None and embedder is not None:
                pipeline = RetrievalPipeline(
                    settings=settings,
                    embedder=embedder,
                    store=store,
                    reranker=reranker,
                )
                app.state.pipeline = pipeline

            cache = getattr(app.state, "cache", None)
            if cache is None:
                cache = (
                    GenerationCache(
                        cache_path=settings.LLM_CACHE_PATH,
                        enabled=settings.LLM_CACHE_WRITE,
                    )
                    if settings.LLM_CACHE_PATH
                    else None
                )
                app.state.cache = cache

            if pipeline is not None and app.state.router is not None:
                service = GenerationService(
                    settings=settings,
                    pipeline=pipeline,
                    router=app.state.router,
                    cache=cache,
                )
                app.state.service = service
            else:
                logger.warning(
                    "[lifespan] Pipeline or Router not available; GenerationService not built."
                )
                app.state.service = None
        except Exception as exc:
            logger.exception("[lifespan] Generation service setup failed: %s", exc)
            app.state.service = None

    yield

    logger.info("[lifespan] Application shutdown complete.")


def create_app(custom_settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        custom_settings: Optional settings instance (used in tests).

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Filumart RAG Assistant",
        description="Grounded product catalog assistant with hybrid search and true SSE streaming",
        version="1.0.0",
        lifespan=lifespan,
    )

    if custom_settings is not None:
        app.state.settings = custom_settings

    # 1. Global Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning(
            "[validation] 422 error on %s %s: %s", request.method, request.url.path, exc.errors()
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid request query or payload parameters.",
                    "details": exc.errors(),
                }
            },
        )

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("[app_error] %s: %s", exc.__class__.__name__, exc)
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": exc.__class__.__name__,
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("[internal_error] Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An internal server error occurred.",
                }
            },
        )

    # 2. Register API routes first so they take precedence over static files
    app.include_router(api_router)

    # 3. Mount static frontend directory at root
    frontend_dir = Path("frontend")
    if frontend_dir.exists() and frontend_dir.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(frontend_dir), html=True),
            name="frontend",
        )
    else:
        logger.warning("[app] Static directory '%s' not found.", frontend_dir)

    return app
