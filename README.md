# KYC Document Extraction API

A stateless, OCR-based REST API for extracting structured data from Indian KYC documents. Upload an image or PDF of an Aadhaar Card, PAN Card, or Driving License and get back structured JSON with extracted fields.

## Features

- **Document Classification** — Automatically identifies document type (Aadhaar/PAN/DL)
- **Field Extraction** — Extracts name, DOB, document number, address, and more
- **Multi-format Support** — Accepts JPEG, PNG, TIFF, WEBP, and PDF uploads
- **Multi-page PDF** — Processes all pages (e.g., Aadhaar front + back)
- **Security Hardened** — Magic-bytes validation, PDF bomb detection, filename sanitization
- **PII Protection** — Structured logging with automatic Aadhaar/PAN redaction
- **Browser UI** — Drag-and-drop upload interface included
- **Stateless** — No database, no file storage, horizontally scalable
- **Docker Ready** — Multi-stage Dockerfile with health checks

## Supported Documents

| Document | Fields Extracted |
|----------|-----------------|
| Aadhaar Card (Front) | Aadhaar number, name, DOB, gender |
| Aadhaar Card (Back) | Address, pincode, state |
| PAN Card | PAN number, name, father's name, DOB |
| Driving License | License number, name, DOB, issue date, expiry date, address, issuing authority |

## Quick Start

### Prerequisites

- Python 3.11+
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) (for PDF support)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/kyc-document-extraction.git
cd kyc-document-extraction

# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -e ".[dev]"
```

### Run the Server

```bash
# Development
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production
gunicorn app.main:app -c gunicorn.conf.py
```

### Access

- **UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Usage

### Extract Document Data

```bash
curl -X POST http://localhost:8000/api/v1/kyc/extract \
  -F "file=@/path/to/document.jpg"
```

### Response

```json
{
  "request_id": "req_a1b2c3d4e5f6...",
  "document_type": "pan_card",
  "classification_confidence": 0.85,
  "extracted_data": {
    "pan_number": "ABCDE1234F",
    "name": "Rajesh Kumar",
    "father_name": "Suresh Kumar",
    "dob": "15/08/1990"
  },
  "processing_time_ms": 1850,
  "ocr_confidence": 0.91,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Error Response

```json
{
  "request_id": "req_xyz789...",
  "error_code": "INVALID_FILE_TYPE",
  "error_message": "File type is not supported",
  "timestamp": "2024-01-15T10:31:00Z"
}
```

## Docker

```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t kyc-extraction .
docker run -p 8000:8000 kyc-extraction
```

## Configuration

All settings are configurable via environment variables (prefixed with `KYC_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `KYC_PORT` | 8000 | Server port |
| `KYC_DEBUG` | false | Include OCR text in response |
| `KYC_MAX_FILE_SIZE_MB` | 10 | Max upload size |
| `KYC_MAX_PDF_PAGES` | 10 | Max PDF pages |
| `KYC_LOG_LEVEL` | INFO | Log verbosity |
| `KYC_OCR_USE_GPU` | false | Enable GPU acceleration |
| `KYC_CORS_ORIGINS` | ["*"] | Allowed CORS origins |

## Architecture

```
Upload → Security Validation → Preprocessing → OCR → Classification → Extraction → Validation → Response
```

- **Security**: Magic-bytes file type detection, PDF bomb checks, filename sanitization
- **Preprocessing**: Grayscale, contrast enhancement, smart DPI normalization
- **OCR**: PaddleOCR with CPU optimization (MKL-DNN)
- **Classification**: Weighted pattern matching across document types
- **Extraction**: Document-specific regex + heuristic field extraction
- **Validation**: Verhoeff checksum (Aadhaar), format validation, date normalization

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# Skip slow property tests
pytest -m "not slow"
```

## Project Structure

```
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── api/                    # REST endpoints & error handlers
│   ├── core/                   # Config & logging
│   ├── extraction/             # Document field extractors
│   ├── middleware/             # PII filter & request logging
│   ├── models/                 # Data models
│   ├── ocr/                    # PaddleOCR engine wrapper
│   ├── schemas/                # Pydantic response models
│   ├── services/               # Classifier & preprocessor
│   ├── utils/                  # Utilities
│   └── validators/             # Security & field validation
├── tests/                      # Unit & property-based tests
├── static/                     # Browser UI
├── Dockerfile                  # Production container
├── docker-compose.yml          # Local development
└── pyproject.toml              # Dependencies
```

## Tech Stack

- **Python 3.11+** with FastAPI
- **PaddleOCR** for text recognition (10 Indian languages)
- **OpenCV** for image preprocessing
- **Poppler** for PDF conversion
- **Hypothesis** for property-based testing
- **Docker** for deployment

## Limitations

- OCR accuracy depends on image quality — clear, well-lit photos produce better results
- Hindi/regional text recognition is less reliable than English
- e-Aadhaar letter PDFs have complex layouts that may affect extraction
- No GPU required but processing is CPU-intensive (~2-3 seconds per document)

## License

MIT
