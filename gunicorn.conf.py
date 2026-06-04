"""
Gunicorn configuration for KYC Document Extraction API.

Optimized for OCR workloads with longer timeouts and preloaded models.
"""

import multiprocessing
import os

# Server socket
bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:8000")

# Worker processes
# Default: 2 * CPU cores + 1 (suitable for CPU-bound OCR workloads)
workers = int(os.environ.get("GUNICORN_WORKERS", (2 * multiprocessing.cpu_count()) + 1))

# Uvicorn worker class for async support
worker_class = "uvicorn.workers.UvicornWorker"

# Timeout: 120 seconds to accommodate slow OCR processing on large documents
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))

# Graceful timeout for worker shutdown
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", "30"))

# Keep-alive connections
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", "5"))

# Preload app so OCR model is loaded once in master process
# and shared across workers via copy-on-write memory
preload_app = True

# Logging
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")

# Process naming
proc_name = "kyc-extraction-api"

# Worker tmp dir (use RAM-backed tmpfs for better performance)
worker_tmp_dir = "/dev/shm"

# Max requests per worker before recycling (prevents memory leaks)
max_requests = int(os.environ.get("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.environ.get("GUNICORN_MAX_REQUESTS_JITTER", "50"))
