"""
KYC Document Extraction API - Application Entry Point

FastAPI application with Uvicorn server configuration for the
stateless OCR-based KYC document extraction system.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.endpoints import router
from app.api.exception_handlers import (
    security_validation_exception_handler,
    unhandled_exception_handler,
)
from app.core.config import get_settings
from app.middleware.request_logging import RequestLoggingMiddleware
from app.validators.security import SecurityValidationError

# Environment-configurable server settings
HOST = os.environ.get("KYC_HOST", "0.0.0.0")
PORT = int(os.environ.get("KYC_PORT", "8000"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup: Initialize the OCR engine singleton.
    Shutdown: Clean up resources.
    """
    # Startup: OCR engine initialization
    from app.ocr.engine import OCREngine

    OCREngine()
    yield
    # Shutdown: Resource cleanup placeholder


app = FastAPI(
    title="KYC Document Extraction API",
    version="1.0.0",
    description=(
        "Production-ready, stateless OCR-based KYC document extraction "
        "service for Indian identity documents (Aadhaar, PAN, Driving License)."
    ),
    lifespan=lifespan,
)

# --- CORS Middleware ---
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Logging Middleware ---
app.add_middleware(RequestLoggingMiddleware)

# --- Exception Handlers ---
app.add_exception_handler(SecurityValidationError, security_validation_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, unhandled_exception_handler)  # type: ignore[arg-type]

# --- Router Registration ---
app.include_router(router)

# --- Static Files ---
static_dir = Path(__file__).parent.parent / "static"


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the frontend HTML page at the root path."""
    index_path = static_dir / "index.html"
    return FileResponse(str(index_path), media_type="text/html")


if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def run() -> None:
    """
    Start the application using Uvicorn programmatically.

    This function is the entry point for the `kyc-server` script
    defined in pyproject.toml. Server host and port are configurable
    via KYC_HOST and KYC_PORT environment variables.
    """
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=False,
    )
