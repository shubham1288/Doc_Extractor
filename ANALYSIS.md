# KYC Document Extraction System — Technical Analysis

## 1. System Overview

This is a **stateless, OCR-based REST API** that accepts Indian KYC documents (Aadhaar Card, PAN Card, Driving License) as images or PDFs, extracts structured data fields, and returns them as JSON.

**Key architectural decision**: Fully stateless — no database, no session storage, no file persistence. Every request is self-contained. This enables horizontal scaling behind a load balancer.

---

## 2. Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Web Framework | FastAPI (Python) | Async support, auto-docs, Pydantic validation |
| OCR Engine | PaddleOCR | Best open-source OCR for Indian languages, CPU-friendly |
| Image Processing | OpenCV (headless) | Industry standard, fast, no GUI deps |
| PDF Conversion | pdf2image + Poppler | Converts PDF pages to images at 300 DPI |
| Configuration | Pydantic Settings | Type-safe env var config with validation |
| Logging | structlog | Structured JSON logs, processor pipeline |
| Testing | pytest + Hypothesis | Property-based testing for correctness |
| Deployment | Docker + Gunicorn/Uvicorn | Production-ready, multi-worker |

---

## 3. Request Processing Pipeline (Step by Step)

```
Upload → Security Validation → Preprocessing → OCR → Classification → Extraction → Validation → Response
```

### Step 1: File Upload & Security Validation
**File**: `app/validators/security.py`

- Checks file size (max 10 MB)
- Detects REAL file type via magic bytes (ignores declared content-type)
- Validates extension is in allowed set (.jpg, .jpeg, .png, .tiff, .webp, .pdf)
- Sanitizes filename (strips path traversal, null bytes, special chars)
- For PDFs: checks page count, decompression ratio, nested object depth (bomb detection)

**Key point**: We NEVER trust the client-declared content-type. We always inspect the first 12 bytes to determine true file type.

### Step 2: Image Preprocessing (for regular images only)
**File**: `app/services/preprocessor.py`

For uploaded images (not PDFs):
1. Convert to grayscale
2. Normalize resolution to 300 DPI (assuming 72 DPI input)
3. Auto-rotate (detect 90/270 degree rotation via edge analysis)
4. Deskew (Hough transform, correct if angle > 0.5°)
5. Denoise (bilateral filter if blur score < 100)
6. CLAHE contrast enhancement
7. Adaptive thresholding (if std dev < 40)

For PDF pages: Only grayscale + CLAHE (PDF is already 300 DPI from Poppler).

### Step 3: OCR Text Extraction
**File**: `app/ocr/engine.py`

- Thread-safe singleton (double-checked locking with `threading.Lock`)
- PaddleOCR with CPU optimization (MKL-DNN enabled)
- Supports 10 languages: English + Hindi, Marathi, Gujarati, Bengali, Tamil, Telugu, Kannada, Malayalam, Punjabi
- Returns `OCRResult`: text, bounding boxes, per-box confidence, average confidence
- For multi-page PDFs: OCRs each page separately, then merges results (deduplication + confidence averaging)

### Step 4: Document Classification
**File**: `app/services/classifier.py`

Pattern-matching classifier that scores text against 4 document types:
- **Aadhaar Front**: UIDAI keyword (0.2), 12-digit pattern (0.3), gender (0.1), DOB (0.1), Govt of India (0.15)
- **Aadhaar Back**: 12-digit pattern (0.25), address keyword (0.25), pincode (0.15), UIDAI (0.15)
- **PAN Card**: PAN format ABCDE1234F (0.35), Income Tax (0.25), Father's name (0.15), Govt of India (0.1)
- **Driving License**: Driving/licence (0.25), transport/RTO (0.2), validity dates (0.15), vehicle class (0.15), DL number (0.15)

**Threshold**: If best score < 0.5 → returns UNKNOWN.

### Step 5: Field Extraction
**Files**: `app/extraction/aadhaar.py`, `pan.py`, `driving_license.py`

Each extractor uses regex patterns and heuristics to pull fields from OCR text:

**Aadhaar**:
- Number: 12-digit with Verhoeff checksum validation, distinguishes VID from Aadhaar
- Name: Multi-strategy (label search → capitalized English line scoring → S/O inference)
- DOB: Finds dates near "DOB"/"Birth" keywords, penalizes enrolment/print dates
- Gender: Male/Female/Transgender including Hindi equivalents
- Address: Looks for "Address:" keyword or S/O, D/O patterns
- Pincode: 6 digits, first digit 1-9
- State: Pincode-to-state mapping → keyword search → word-boundary match

**PAN**:
- Number: Regex `[A-Z]{5}[0-9]{4}[A-Z]`
- Name, Father's name, DOB

**Driving License**:
- Number: Multiple state formats (KA01..., DL-04..., MH12...)
- Name, DOB, issue date, expiry date, address, issuing authority

### Step 6: Field Validation
**File**: `app/validators/field_validator.py`

- Dates: Normalized to DD/MM/YYYY from any format
- Pincode: 6 digits, first digit 1-9
- DL Number: Valid state code + digits
- Aadhaar: Verhoeff checksum

### Step 7: Response Building
Returns JSON with: request_id, document_type, classification_confidence, extracted_data, processing_time_ms, ocr_confidence, timestamp.

---

## 4. Architecture Diagram

```
┌─────────────┐     POST /api/v1/kyc/extract
│   Client    │────────────────────────────────┐
└─────────────┘                                │
                                               ▼
┌──────────────────────────────────────────────────────┐
│                    FastAPI App                         │
│                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │  Security   │→ │ Preprocessor │→ │ OCR Engine │  │
│  │  Validator  │  │  (OpenCV)    │  │ (PaddleOCR)│  │
│  └─────────────┘  └──────────────┘  └────────────┘  │
│         │                                    │        │
│         │         ┌──────────────┐           │        │
│         │         │  Classifier  │←──────────┘        │
│         │         └──────┬───────┘                    │
│         │                │                            │
│         │         ┌──────▼───────┐                    │
│         │         │  Extractors  │                    │
│         │         │ (Aadhaar/PAN/│                    │
│         │         │     DL)      │                    │
│         │         └──────┬───────┘                    │
│         │                │                            │
│         │         ┌──────▼───────┐                    │
│         │         │  Validators  │                    │
│         │         └──────────────┘                    │
└──────────────────────────────────────────────────────┘
```

---

## 5. Key Design Decisions

### Why PaddleOCR over Tesseract?
- Better accuracy on Indian scripts (Hindi, Telugu, Tamil)
- Better at handling mixed English+Hindi text on Aadhaar cards
- Angle classification built-in
- Faster on CPU with MKL-DNN

### Why Stateless?
- Horizontal scaling: any instance can handle any request
- No database means no connection pooling, no migrations, no backups
- Memory cleanup on every request (gc.collect in finally block)
- Perfect for Kubernetes/container deployments

### Why Singleton for OCR Engine?
- PaddleOCR model loading takes 5-10 seconds and uses ~500MB RAM
- Loading once and reusing across requests is essential
- Thread-safe via `threading.Lock` on inference calls
- Gunicorn with `preload_app=True` shares model across workers via copy-on-write

### Why Magic Bytes over Content-Type?
- Attackers can send malware with `Content-Type: image/jpeg`
- Magic bytes are the ACTUAL file format indicator (first N bytes)
- This prevents disguised executable uploads

---

## 6. Security Measures

| Threat | Mitigation |
|--------|-----------|
| Malicious file upload | Magic bytes validation, extension whitelist |
| PDF bomb | Page count limit, decompression ratio check, nesting depth check |
| Path traversal | Filename sanitization (strip /, \, .., null bytes) |
| PII in logs | structlog processor that redacts Aadhaar/PAN patterns |
| Memory leaks | gc.collect() in finally blocks, file_bytes set to None |
| File too large | 10MB hard limit checked before any processing |
| OCR not ready | HTTP 503 with OCR_NOT_READY error code |

---

## 7. Minimum System Requirements

### Development
- Python 3.11+
- 4 GB RAM (PaddleOCR model ~500MB + processing headroom)
- Poppler utilities (for PDF processing)
- ~500MB disk for PaddleOCR model files (auto-downloaded on first run)

### Production (Docker)
- 2-4 GB RAM per container (recommended 4GB)
- 2+ CPU cores (OCR is CPU-intensive)
- Poppler-utils (included in Docker image)
- No GPU required (CPU-optimized with MKL-DNN)

### Supported Input
- Image formats: JPEG, PNG, TIFF, WEBP
- PDF (max 10 pages)
- Max file size: 10 MB
- Documents: Aadhaar Card (front/back), PAN Card, Driving License

---

## 8. Performance Targets

| Metric | Target |
|--------|--------|
| Single image processing | < 2 seconds |
| Multi-page PDF (2 pages) | < 5 seconds |
| OCR model load time | ~5-10 seconds (one-time at startup) |
| Concurrent requests | Thread-safe, limited by CPU cores |
| Memory per request | ~50-200MB (cleaned up after) |

---

## 9. Constraints & Limitations

1. **OCR Accuracy**: Depends entirely on image quality. Blurry/low-resolution scans will produce garbled text.
2. **e-Aadhaar PDFs**: Complex layout with instructional text, QR codes, multiple pages — extraction quality varies.
3. **Language**: OCR works best with English text. Hindi/regional language text is supported but less reliable.
4. **No ML Classification**: Uses rule-based pattern matching, not trained ML model. Works well for standard documents but may fail on unusual layouts.
5. **No Database**: Cannot store/compare results across requests. Each request is independent.
6. **CPU-only**: No GPU acceleration. Processing is fast enough for single requests but not for high-throughput batch processing.
7. **Verhoeff Checksum**: Only validates Aadhaar numbers — cannot generate them. Invalid numbers are still extracted but with lower confidence.

---

## 10. Project Structure

```
Doc-extractor/
├── app/
│   ├── main.py              # FastAPI app entry point, middleware, routes
│   ├── api/
│   │   ├── endpoints.py     # REST endpoints (extract, health, ready)
│   │   └── exception_handlers.py  # Global error handling
│   ├── core/
│   │   ├── config.py        # Pydantic settings (env vars)
│   │   └── logging.py       # structlog configuration
│   ├── extraction/
│   │   ├── base.py          # Abstract extractor + helpers
│   │   ├── aadhaar.py       # Aadhaar field extraction
│   │   ├── pan.py           # PAN field extraction
│   │   ├── driving_license.py  # DL field extraction
│   │   ├── verhoeff.py      # Aadhaar checksum algorithm
│   │   └── router.py        # Maps document type → extractor
│   ├── middleware/
│   │   ├── pii_filter.py    # Redacts PII from logs
│   │   └── request_logging.py  # Request timing + ID
│   ├── models/
│   │   ├── document.py      # DocumentType enum, ClassificationResult
│   │   ├── ocr.py           # OCRResult dataclass
│   │   └── validation.py    # ValidationResult dataclass
│   ├── ocr/
│   │   └── engine.py        # Thread-safe PaddleOCR singleton
│   ├── schemas/
│   │   └── responses.py     # Pydantic response models
│   ├── services/
│   │   ├── classifier.py    # Document type classification
│   │   ├── preprocessor.py  # Image preprocessing pipeline
│   │   └── pdf_processor.py # PDF → images + OCR merging
│   ├── utils/
│   │   └── request_id.py    # UUID-based request ID generation
│   └── validators/
│       ├── security.py      # File upload security checks
│       └── field_validator.py  # Date/pincode/DL validation
├── tests/                   # Unit + property-based tests
├── static/
│   └── index.html           # Browser upload UI
├── pyproject.toml           # Dependencies + config
├── Dockerfile               # Multi-stage production build
├── docker-compose.yml       # Local development
└── gunicorn.conf.py         # Production server config
```

---

## 11. API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/kyc/extract` | Upload document, get extracted fields |
| GET | `/health` | System health + OCR readiness |
| GET | `/ready` | Detailed readiness probes |
| GET | `/` | Browser UI for file upload |
| GET | `/docs` | Auto-generated Swagger docs |

---

## 12. Environment Variables

All prefixed with `KYC_`:

| Variable | Default | Description |
|----------|---------|-------------|
| KYC_HOST | 0.0.0.0 | Server bind address |
| KYC_PORT | 8000 | Server port |
| KYC_DEBUG | false | Include raw OCR text in response |
| KYC_LOG_LEVEL | INFO | Logging verbosity |
| KYC_MAX_FILE_SIZE_MB | 10 | Upload size limit |
| KYC_MAX_PDF_PAGES | 10 | PDF page limit |
| KYC_LOW_CONFIDENCE_THRESHOLD | 0.3 | Below this → return UNKNOWN |
| KYC_CLASSIFICATION_THRESHOLD | 0.5 | Below this → document type UNKNOWN |
| KYC_OCR_USE_GPU | false | Enable GPU for OCR |
| KYC_CORS_ORIGINS | ["*"] | Allowed CORS origins |

---

## 13. Error Handling Strategy

All errors return a consistent structure:
```json
{
  "request_id": "req_abc123...",
  "error_code": "ERROR_CODE_HERE",
  "error_message": "Human-readable message (no PII)",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

| HTTP Code | Error Code | When |
|-----------|-----------|------|
| 400 | INVALID_FILE_TYPE | Magic bytes don't match allowed types |
| 400 | IMAGE_DECODE_ERROR | OpenCV can't decode the image |
| 413 | FILE_TOO_LARGE | File > 10 MB |
| 422 | (FastAPI default) | No file provided |
| 500 | INTERNAL_ERROR | Unhandled exception |
| 500 | PDF_PROCESSING_ERROR | Poppler can't convert PDF |
| 503 | OCR_NOT_READY | PaddleOCR model not loaded |

---

## 14. Testing Strategy

- **Unit tests**: Each module tested in isolation (160+ tests)
- **Property-based tests**: 21 Hypothesis properties verifying invariants:
  - File size enforcement always works for ANY file
  - Magic bytes ALWAYS override content-type
  - Confidence scores ALWAYS in [0.0, 1.0]
  - Below threshold ALWAYS returns UNKNOWN
  - Verhoeff detects ALL single-digit mutations
  - No PII EVER appears in logs
  - Request IDs are ALWAYS unique
- **Integration tests**: E2E pipeline with mocked OCR, concurrent requests, error structure consistency

---

## 15. Common Questions Your Engineer Might Ask

**Q: Why not use Tesseract?**
A: PaddleOCR has significantly better accuracy on Indian scripts and mixed-language documents. It also handles rotated/skewed text better out of the box.

**Q: How does it handle concurrent requests?**
A: The OCR engine is a thread-safe singleton with a `threading.Lock` around inference. FastAPI handles async I/O. Gunicorn runs multiple worker processes, each with their own OCR engine instance (shared via preload_app + copy-on-write memory).

**Q: What if the document type can't be determined?**
A: Returns `document_type: "unknown"` with `extracted_data: {}`. The classification threshold is 0.5 — if no pattern scores above that, we don't guess.

**Q: How is PII protected?**
A: A structlog processor scans ALL log values and redacts Aadhaar/PAN patterns before they reach the log output. Error responses never include document content. Memory is explicitly freed after each request.

**Q: Can it handle password-protected PDFs?**
A: No — pypdf will fail to parse them. But the system gracefully falls back to basic safety checks and attempts conversion via Poppler. If Poppler can't open it either, you get a 500 PDF_PROCESSING_ERROR.

**Q: What's the Verhoeff algorithm?**
A: A checksum used by UIDAI to validate Aadhaar numbers. It detects all single-digit errors and all adjacent transposition errors. We use it to verify extracted Aadhaar numbers — valid numbers get higher confidence scores.

**Q: Why no database?**
A: The system is designed for stateless horizontal scaling. Each request is self-contained. If you need to store results, the consuming application should save the API response.

**Q: What happens if OCR produces garbage?**
A: If average OCR confidence is below 0.3 (configurable), we short-circuit and return UNKNOWN without attempting extraction. This prevents garbage-in-garbage-out results.
