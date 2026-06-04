# ==============================================================================
# Stage 1: Build stage - install dependencies
# ==============================================================================
FROM python:3.12-slim AS build

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# ==============================================================================
# Stage 2: Runtime stage - minimal image
# ==============================================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install runtime dependencies:
# - poppler-utils: required by pdf2image for PDF to image conversion
# - libgl1: required by OpenCV for image processing
# - libglib2.0-0: required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from build stage
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY app/ ./app/
COPY static/ ./static/
COPY gunicorn.conf.py ./

# Create non-root user for security
RUN groupadd -r kyc && useradd -r -g kyc -d /app -s /sbin/nologin kyc && \
    chown -R kyc:kyc /app

USER kyc

# Expose the application port
EXPOSE 8000

# Health check using the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with Gunicorn + Uvicorn workers
CMD ["gunicorn", "app.main:app", "-c", "gunicorn.conf.py"]
