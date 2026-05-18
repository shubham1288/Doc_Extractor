"""
Document Extractor - FastAPI OCR Application
Extracts text from document images (Aadhaar, PAN, Passport, etc.)
using Tesseract OCR with OpenCV preprocessing.
"""

import os
import re
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import pytesseract
import uvicorn
import fitz  # PyMuPDF - for PDF handling

# ============================================================
# Configuration
# ============================================================

app = FastAPI(title="Document Extractor")

# Folder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

# Max file size (16 MB)
MAX_FILE_SIZE = 16 * 1024 * 1024

# Configure Tesseract path (auto-detect based on environment)
import shutil
tesseract_path = shutil.which('tesseract')
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
elif os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# ============================================================
# Helper Functions
# ============================================================

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """
    Preprocess the image using OpenCV to improve OCR accuracy.
    Uses multiple preprocessing strategies and returns the best result.
    """
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read the image file.")

    # Resize if image is too small (improves OCR on low-res images)
    height, width = image.shape[:2]
    if width < 1000:
        scale = 1000 / width
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Strategy 1: Simple thresholding with Otsu's method (works well for clean docs)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Strategy 2: Light denoise + sharpen (best for Aadhaar/PAN cards)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, sharpening_kernel)

    # Strategy 3: Adaptive threshold with larger block size (for uneven lighting)
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    # Try all strategies and pick the one with most text
    strategies = {
        'sharpened': sharpened,
        'otsu': otsu,
        'adaptive': adaptive,
        'gray': gray,  # Sometimes raw grayscale works best
    }

    best_text = ""
    best_path = None

    for name, processed in strategies.items():
        temp_path = image_path.replace('.', f'_prep_{name}.', 1)
        cv2.imwrite(temp_path, processed)

        # OCR with optimized config for document cards
        text = pytesseract.image_to_string(
            temp_path,
            lang='eng',
            config='--oem 3 --psm 6'
        )

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Pick the result with the most readable characters
        readable_chars = sum(1 for c in text if c.isalnum() or c.isspace())
        if readable_chars > len(best_text):
            best_text = text

    return best_text.strip() if best_text else ""


def extract_text(image_path):
    """
    Extract text from the image or PDF using Tesseract OCR.
    Uses multiple preprocessing strategies for best accuracy.
    For PDFs, converts each page to an image first.
    """
    file_ext = os.path.splitext(image_path)[1].lower()

    if file_ext == '.pdf':
        return extract_text_from_pdf(image_path)

    # Use the multi-strategy preprocessing pipeline
    return preprocess_image(image_path)


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    First tries to extract embedded text directly.
    If no text found, converts pages to images and runs OCR.
    """
    all_text = []

    # Open the PDF
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]

        # First try to extract embedded text directly (faster, more accurate)
        page_text = page.get_text().strip()

        if page_text:
            all_text.append(page_text)
        else:
            # No embedded text - convert page to image and OCR it
            # Render page at 300 DPI for good OCR quality
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat)

            # Save as temporary image
            temp_image_path = pdf_path.replace('.pdf', f'_page_{page_num}.png')
            pix.save(temp_image_path)

            # Use multi-strategy preprocessing pipeline
            page_text = preprocess_image(temp_image_path)

            if page_text:
                all_text.append(page_text)

            # Clean up temp image
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    doc.close()
    return "\n\n".join(all_text)


def extract_fields(text):
    """
    Extract important fields from the OCR text using regex patterns.
    Supports: Aadhaar Number, Name, Father's Name, DOB, Gender,
    PAN Number, Address, PIN Code, Mobile, VID, Enrolment No.
    """
    fields = {}

    # Extract Aadhaar Number (format: XXXX XXXX XXXX)
    aadhaar_pattern = r'\b\d{4}\s\d{4}\s\d{4}\b'
    aadhaar_match = re.search(aadhaar_pattern, text)
    if aadhaar_match:
        fields['Aadhaar Number'] = aadhaar_match.group()

    # Extract Name (English name in CAPS, typically after a Telugu/Hindi name)
    name_patterns = [
        r'(?:DOB|MALE|FEMALE).*?\n\s*([A-Z][A-Z\s]+)\n',  # Name line after DOB/gender
        r'\n([A-Z][A-Z\s]{3,40})\n\s*(?:S/O|D/O|W/O|C/O)',  # Name before S/O, D/O
        r'(?:Name|NAME)\s*[:\-]?\s*([A-Z][A-Z\s]{3,40})',
        r'\n([A-Z]{2,}\s[A-Z]{2,}(?:\s[A-Z]{2,})?)\s*\n',  # Two or three capitalized words
    ]
    for pattern in name_patterns:
        name_match = re.search(pattern, text)
        if name_match:
            name = name_match.group(1).strip()
            # Filter out common non-name strings
            skip_words = ['MALE', 'FEMALE', 'GOVERNMENT', 'INDIA', 'AADHAAR', 'ADDRESS', 'UNIQUE', 'AUTHORITY']
            if len(name) > 3 and len(name) < 50 and not any(w in name for w in skip_words):
                fields['Name'] = name.title()
                break

    # If name not found from patterns, try specific Aadhaar format
    if 'Name' not in fields:
        # Look for pattern like "NANDYALA NAVEEN" (all caps name on its own line)
        all_caps_name = re.search(r'\n([A-Z][A-Z]+\s[A-Z][A-Z]+(?:\s[A-Z]+)?)\s*\n', text)
        if all_caps_name:
            candidate = all_caps_name.group(1).strip()
            skip_words = ['MALE', 'FEMALE', 'GOVERNMENT', 'INDIA', 'AADHAAR', 'UNIQUE', 'AUTHORITY', 'IDENTIFICATION']
            if not any(w in candidate for w in skip_words):
                fields['Name'] = candidate.title()

    # Extract Father's/Spouse Name (S/O, D/O, W/O, C/O)
    relation_pattern = r'(?:S/O|D/O|W/O|C/O)\s+([A-Za-z][A-Za-z\s,]+?)(?:,|\n|\d)'
    relation_match = re.search(relation_pattern, text)
    if relation_match:
        relation_name = relation_match.group(1).strip().rstrip(',')
        if len(relation_name) > 3:
            fields["Father's/Guardian's Name"] = relation_name.title()

    # Extract Date of Birth (look for dates near DOB/birth labels first)
    dob_labeled_patterns = [
        r'(?:DOB|D\.O\.B|Date of Birth|Birth|Year of Birth)\s*[:\-]?\s*(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})',
        r'(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\s*(?:DOB|Date of Birth)',
    ]
    for pattern in dob_labeled_patterns:
        dob_match = re.search(pattern, text, re.IGNORECASE)
        if dob_match:
            fields['Date of Birth'] = dob_match.group(1)
            break

    # If no labeled DOB found, pick a date that's clearly in the past
    if 'Date of Birth' not in fields:
        dob_pattern = r'\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b'
        all_dates = re.findall(dob_pattern, text)
        from datetime import datetime as dt
        today = dt.now()
        for date_str in all_dates:
            try:
                sep = '/' if '/' in date_str else ('-' if '-' in date_str else '.')
                parts = date_str.split(sep)
                day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                if 1900 <= year <= (today.year - 1) and 1 <= month <= 12 and 1 <= day <= 31:
                    fields['Date of Birth'] = date_str
                    break
            except (ValueError, IndexError):
                continue

    # Extract Gender
    gender_pattern = r'\b(MALE|FEMALE|Male|Female|male|female|Transgender)\b'
    gender_match = re.search(gender_pattern, text)
    if gender_match:
        fields['Gender'] = gender_match.group().capitalize()

    # Extract Address
    address_pattern = r'(?:Address|చిరునామా)\s*[:\-]?\s*(S/O.*?(?:\d{6}))'
    address_match = re.search(address_pattern, text, re.DOTALL | re.IGNORECASE)
    if address_match:
        address = address_match.group(1).strip()
        # Clean up the address - remove extra whitespace and newlines
        address = re.sub(r'\s+', ' ', address)
        if len(address) > 10:
            fields['Address'] = address

    # Extract PIN Code (6-digit Indian PIN)
    pin_pattern = r'\b(?:PIN\s*(?:Code)?[:\-]?\s*)?(\d{6})\b'
    pin_match = re.search(pin_pattern, text)
    if pin_match:
        pin = pin_match.group(1)
        # Valid Indian PIN codes start with 1-9
        if pin[0] != '0':
            fields['PIN Code'] = pin

    # Extract Mobile Number (10-digit Indian mobile)
    mobile_pattern = r'(?:Mobile|Phone|Mob)\s*[:\-]?\s*(\d{10})'
    mobile_match = re.search(mobile_pattern, text, re.IGNORECASE)
    if mobile_match:
        fields['Mobile'] = mobile_match.group(1)

    # Extract VID (Virtual ID: XXXX XXXX XXXX XXXX)
    vid_pattern = r'VID\s*[:\-]?\s*(\d{4}\s\d{4}\s\d{4}\s\d{4})'
    vid_match = re.search(vid_pattern, text)
    if vid_match:
        fields['VID'] = vid_match.group(1)

    # Extract Enrolment Number
    enrol_pattern = r'(?:Enrolment|Enrollment|Registration)\s*(?:No\.?|Number)?\s*[:\-]?\s*([\d/]+)'
    enrol_match = re.search(enrol_pattern, text, re.IGNORECASE)
    if enrol_match:
        fields['Enrolment No.'] = enrol_match.group(1)

    # Extract PAN Number (format: ABCDE1234F)
    pan_pattern = r'\b[A-Z]{5}\d{4}[A-Z]\b'
    pan_match = re.search(pan_pattern, text)
    if pan_match:
        fields['PAN Number'] = pan_match.group()

    return fields


def save_output(text, fields, filename):
    """Save extracted text and fields to a .txt file in the outputs folder."""
    output_filename = f"extracted_{os.path.splitext(filename)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  DOCUMENT EXTRACTION RESULTS\n")
        f.write(f"  File: {filename}\n")
        f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        if fields:
            f.write("--- Extracted Fields ---\n\n")
            for key, value in fields.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

        f.write("--- Full Extracted Text ---\n\n")
        f.write(text)
        f.write("\n\n" + "=" * 60 + "\n")

    return output_filename


# ============================================================
# FastAPI Routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload and OCR extraction."""

    # Check if a file was actually selected
    if not file.filename:
        return JSONResponse(
            status_code=400,
            content={'error': 'No file selected. Please choose an image to upload.'}
        )

    # Validate file extension
    if not allowed_file(file.filename):
        return JSONResponse(
            status_code=400,
            content={'error': 'Invalid file type. Only JPG, JPEG, PNG, and PDF files are allowed.'}
        )

    try:
        # Read file content
        contents = await file.read()

        # Check file size
        if len(contents) > MAX_FILE_SIZE:
            return JSONResponse(
                status_code=400,
                content={'error': 'File is too large. Maximum size is 16MB.'}
            )

        # Create safe filename with timestamp
        safe_filename = re.sub(r'[^\w\-.]', '_', file.filename)
        name, ext = os.path.splitext(safe_filename)
        filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Save the file
        with open(filepath, 'wb') as f:
            f.write(contents)

        # Extract text using OCR
        extracted_text = extract_text(filepath)

        if not extracted_text:
            return JSONResponse(
                status_code=400,
                content={'error': 'No text could be extracted from the image. Please try a clearer image.'}
            )

        # Extract important fields using regex
        fields = extract_fields(extracted_text)

        # Save results to output file
        output_filename = save_output(extracted_text, fields, filename)

        return JSONResponse(content={
            'success': True,
            'text': extracted_text,
            'fields': fields,
            'image_url': f'/uploads/{filename}',
            'output_file': output_filename
        })

    except ValueError as e:
        return JSONResponse(status_code=400, content={'error': str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'error': f'An error occurred during processing: {str(e)}'}
        )


@app.get("/uploads/{filename}")
async def uploaded_file(filename: str):
    """Serve uploaded images for preview."""
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    return JSONResponse(status_code=404, content={'error': 'File not found.'})


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download the extracted text file."""
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, filename=filename, media_type='application/octet-stream')
    return JSONResponse(status_code=404, content={'error': 'File not found.'})


# ============================================================
# Run the Application
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    print("\n" + "=" * 50)
    print("  Document Extractor - OCR Application")
    print(f"  Running at: http://127.0.0.1:{port}")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
