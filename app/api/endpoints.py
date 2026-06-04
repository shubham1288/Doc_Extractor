"""
REST API endpoints for the KYC Document Extraction service.

Provides:
- POST /api/v1/kyc/extract — Full KYC extraction pipeline
- GET /health — System health status
- GET /ready — Readiness probes (OCR + memory)
"""

import gc
import logging
import time
from datetime import datetime, timezone

import psutil
from fastapi import APIRouter, File, UploadFile

from app.extraction.router import get_extractor
from app.models.document import DocumentType
from app.ocr.engine import OCREngine
from app.schemas.responses import ErrorResponse, KYCExtractionResponse
from app.services.classifier import DocumentClassifier
from app.services.preprocessor import ImagePreprocessor
from app.utils.request_id import generate_request_id
from app.validators.field_validator import FieldValidator
from app.validators.security import validate_upload

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/api/v1/kyc/extract",
    response_model=KYCExtractionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Extract KYC data from an uploaded document",
    description="Accepts an image or PDF of an Indian KYC document and returns structured extraction results.",
)
async def extract_kyc(file: UploadFile = File(...)) -> KYCExtractionResponse | ErrorResponse:
    """Run the full KYC extraction pipeline on an uploaded file.

    Pipeline steps:
    1. Generate request_id
    2. Validate upload (security checks)
    3. Preprocess image
    4. Run OCR
    5. Classify document type
    6. Extract fields
    7. Validate fields
    8. Build and return response

    All processing is in-memory with cleanup on completion.
    """
    request_id = generate_request_id()
    start_time = time.perf_counter()
    file_bytes: bytes | None = None

    try:
        # Step 1: Read file bytes
        file_bytes = await file.read()
        filename = file.filename or "unnamed"
        content_type = file.content_type or "application/octet-stream"

        logger.info(
            "Processing KYC extraction request",
            extra={"request_id": request_id, "filename_length": len(filename)},
        )

        # Step 2: Validate upload (security)
        validated = validate_upload(file_bytes, filename, content_type)

        # Step 3: Check OCR engine readiness
        ocr_engine = OCREngine()
        if not ocr_engine.is_ready:
            from fastapi.responses import JSONResponse

            error_response = ErrorResponse(
                request_id=request_id,
                error_code="OCR_NOT_READY",
                error_message="OCR engine is not ready for processing",
            )
            return JSONResponse(  # type: ignore[return-value]
                status_code=503,
                content=error_response.model_dump(mode="json"),
            )

        # Step 4: Preprocess image
        import numpy as np
        import cv2

        if validated.is_pdf:
            # PDF processing: convert PDF pages to images using poppler
            try:
                from pdf2image import convert_from_bytes
                import os

                # Find poppler path - check common Windows installation locations
                poppler_path = None
                possible_paths = [
                    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages"),
                ]
                for base in possible_paths:
                    if os.path.isdir(base):
                        for d in os.listdir(base):
                            if "Poppler" in d or "poppler" in d:
                                bin_path = os.path.join(base, d)
                                # Walk to find the bin directory
                                for root, dirs, files in os.walk(bin_path):
                                    if "pdftoppm.exe" in files:
                                        poppler_path = root
                                        break
                                if poppler_path:
                                    break

                convert_kwargs = {"pdf_file": validated.file_bytes, "dpi": 300, "first_page": 1, "last_page": 10}
                if poppler_path:
                    images = convert_from_bytes(poppler_path=poppler_path, **convert_kwargs)
                else:
                    images = convert_from_bytes(**convert_kwargs)

                # Process ALL pages: convert each to BGR numpy array
                pdf_page_arrays = []
                for pil_img in images:
                    page_array = np.array(pil_img)
                    if page_array.ndim == 3 and page_array.shape[2] == 3:
                        page_array = cv2.cvtColor(page_array, cv2.COLOR_RGB2BGR)
                    pdf_page_arrays.append(page_array)

                img_array = pdf_page_arrays[0]  # First page for preprocessing
                is_multi_page_pdf = len(pdf_page_arrays) > 1
            except Exception as e:
                logger.error(
                    "PDF processing failed",
                    extra={"request_id": request_id, "error": str(e)},
                )
                error_response = ErrorResponse(
                    request_id=request_id,
                    error_code="PDF_PROCESSING_ERROR",
                    error_message=f"Failed to process PDF document: {str(e)}",
                )
                from fastapi.responses import JSONResponse

                return JSONResponse(  # type: ignore[return-value]
                    status_code=500,
                    content=error_response.model_dump(mode="json"),
                )
        else:
            # Decode image from bytes
            img_array = np.frombuffer(validated.file_bytes, dtype=np.uint8)
            img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            is_multi_page_pdf = False

            if img_array is None:
                error_response = ErrorResponse(
                    request_id=request_id,
                    error_code="IMAGE_DECODE_ERROR",
                    error_message="Failed to decode the uploaded image",
                )
                from fastapi.responses import JSONResponse

                return JSONResponse(  # type: ignore[return-value]
                    status_code=400,
                    content=error_response.model_dump(mode="json"),
                )

        preprocessor = ImagePreprocessor()

        # Step 5: Run OCR (process all pages for multi-page PDFs)
        if validated.is_pdf and is_multi_page_pdf:
            from app.services.pdf_processor import merge_ocr_results

            page_ocr_results = []
            for page_array in pdf_page_arrays:
                # For PDF pages rendered at 300 DPI, skip heavy preprocessing
                # Just convert to grayscale and enhance contrast (no DPI upscaling)
                gray = preprocessor.to_grayscale(page_array)
                enhanced = preprocessor.enhance_contrast(gray)
                page_ocr = ocr_engine.extract_text(enhanced)
                page_ocr_results.append(page_ocr)

            ocr_result = merge_ocr_results(page_ocr_results)
        elif validated.is_pdf:
            # Single page PDF — also skip DPI upscaling
            gray = preprocessor.to_grayscale(img_array)
            enhanced = preprocessor.enhance_contrast(gray)
            ocr_result = ocr_engine.extract_text(enhanced)
        else:
            preprocessed = preprocessor.preprocess(img_array)
            ocr_result = ocr_engine.extract_text(preprocessed)

        logger.debug(
            "OCR extraction complete",
            extra={
                "request_id": request_id,
                "ocr_text_length": len(ocr_result.text),
                "ocr_confidence": ocr_result.avg_confidence,
                "ocr_text_preview": ocr_result.text[:200] if ocr_result.text else "",
            },
        )

        # Step 6: Check OCR confidence threshold
        from app.core.config import get_settings

        settings = get_settings()
        if ocr_result.avg_confidence < settings.low_confidence_threshold:
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            return KYCExtractionResponse(
                request_id=request_id,
                document_type=DocumentType.UNKNOWN,
                classification_confidence=0.0,
                extracted_data={},
                processing_time_ms=processing_time_ms,
                ocr_confidence=ocr_result.avg_confidence,
            )

        # Step 7: Classify document type
        classifier = DocumentClassifier()
        classification = classifier.classify(ocr_result)

        # Step 8: Extract fields (if document type is known)
        extracted_data: dict = {}
        if classification.document_type != DocumentType.UNKNOWN:
            extractor = get_extractor(classification.document_type)
            extraction_fields = extractor.extract(ocr_result)

            # Convert ExtractionField objects to simple dict
            extracted_data = {
                key: field.value
                for key, field in extraction_fields.items()
                if field.value is not None
            }

            # Step 9: Validate extracted fields
            field_validator = FieldValidator()
            validation_results = field_validator.validate_all(
                classification.document_type, extracted_data
            )

            # Apply normalized values where validation passed
            for field_name, result in validation_results.items():
                if result.is_valid and result.normalized_value is not None:
                    extracted_data[field_name] = result.normalized_value

        # Step 10: Build response
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        response = KYCExtractionResponse(
            request_id=request_id,
            document_type=classification.document_type,
            classification_confidence=classification.confidence,
            extracted_data=extracted_data,
            processing_time_ms=processing_time_ms,
            ocr_confidence=ocr_result.avg_confidence,
            ocr_text=ocr_result.text if settings.debug else None,
        )

        logger.info(
            "KYC extraction completed",
            extra={
                "request_id": request_id,
                "document_type": classification.document_type.value,
                "confidence": classification.confidence,
                "processing_time_ms": processing_time_ms,
            },
        )

        return response

    finally:
        # Ensure cleanup of file bytes and intermediate data
        file_bytes = None
        gc.collect()


@router.get(
    "/health",
    summary="System health check",
    description="Returns system health status including OCR engine readiness.",
)
async def health_check() -> dict:
    """Return system health status.

    Checks:
    - Overall system status
    - OCR engine readiness
    """
    ocr_engine = OCREngine()

    return {
        "status": "healthy",
        "ocr_ready": ocr_engine.is_ready,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/ready",
    summary="Readiness probe",
    description="Returns readiness checks for OCR model and memory status.",
)
async def readiness_check() -> dict:
    """Return readiness probe results.

    Checks:
    - OCR model loaded and ready
    - Memory usage within acceptable bounds (< 90%)
    """
    ocr_engine = OCREngine()
    ocr_ready = ocr_engine.is_ready

    # Memory check: ensure less than 90% memory usage
    memory = psutil.virtual_memory()
    memory_ok = memory.percent < 90.0

    all_ready = ocr_ready and memory_ok

    return {
        "ready": all_ready,
        "checks": {
            "ocr_model": {
                "ready": ocr_ready,
                "detail": "OCR model loaded" if ocr_ready else "OCR model not loaded",
            },
            "memory": {
                "ready": memory_ok,
                "detail": f"Memory usage: {memory.percent:.1f}%",
                "percent_used": memory.percent,
            },
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
