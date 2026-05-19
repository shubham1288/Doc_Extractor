"""
Document Extractor - FastAPI OCR Application
Extracts text from document images (Aadhaar, PAN, Passport, etc.)
using Tesseract OCR with OpenCV preprocessing.
"""

import os
import re
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Request, Form
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


def preprocess_image(image_path, lang='eng+hin+tel'):
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
            lang=lang,
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


def extract_text(image_path, lang='eng+hin+tel'):
    """
    Extract text from the image or PDF using Tesseract OCR.
    Uses multiple preprocessing strategies for best accuracy.
    For PDFs, converts each page to an image first.
    """
    file_ext = os.path.splitext(image_path)[1].lower()

    if file_ext == '.pdf':
        return extract_text_from_pdf(image_path, lang)

    # Use the multi-strategy preprocessing pipeline
    return preprocess_image(image_path, lang)


def extract_text_from_pdf(pdf_path, lang='eng+hin+tel'):
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
            page_text = preprocess_image(temp_image_path, lang)

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

    # Detect Document Type
    text_upper = text.upper()
    if re.search(r'\b\d{4}\s\d{4}\s\d{4}\b', text) and ('AADHAAR' in text_upper or 'UIDAI' in text_upper or 'UID' in text_upper or re.search(r'VID\s*:', text)):
        fields['Document Type'] = 'Aadhaar Card'
    elif re.search(r'\b[A-Z]{5}\d{4}[A-Z]\b', text):
        fields['Document Type'] = 'PAN Card'
    elif 'PASSPORT' in text_upper or 'REPUBLIC OF INDIA' in text_upper:
        fields['Document Type'] = 'Passport'
    elif 'DRIVING' in text_upper or 'LICENCE' in text_upper or 'LICENSE' in text_upper:
        fields['Document Type'] = 'Driving License'
    else:
        fields['Document Type'] = 'Document'

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


def detect_languages(text):
    """Detect which languages are present in the text using Unicode ranges.
    Only returns a language if there's meaningful content (more than just a few chars)."""
    languages = []

    # English (Latin) - check for at least a few words
    english_chars = len(re.findall(r'[A-Za-z]', text))
    if english_chars > 10:
        languages.append('English')

    # Telugu: \u0C00-\u0C7F - need meaningful Telugu content
    telugu_chars = len(re.findall(r'[\u0C00-\u0C7F]', text))
    if telugu_chars > 10:
        languages.append('Telugu')

    # Hindi/Devanagari: \u0900-\u097F - need meaningful Hindi content
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    if hindi_chars > 10:
        languages.append('Hindi')

    return languages


def split_text_by_language(text):
    """Split extracted text into language-specific sections."""
    lines = text.split('\n')
    result = {}

    english_lines = []
    telugu_lines = []
    hindi_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        has_telugu = bool(re.search(r'[\u0C00-\u0C7F]', stripped))
        has_hindi = bool(re.search(r'[\u0900-\u097F]', stripped))
        has_english = bool(re.search(r'[A-Za-z]', stripped))

        if has_telugu:
            telugu_lines.append(stripped)
        elif has_hindi:
            hindi_lines.append(stripped)
        elif has_english:
            english_lines.append(stripped)

    if english_lines:
        result['English'] = '\n'.join(english_lines)
    if telugu_lines:
        result['Telugu'] = '\n'.join(telugu_lines)
    if hindi_lines:
        result['Hindi'] = '\n'.join(hindi_lines)

    return result


def extract_fields_by_language(text):
    """Extract fields separately for each detected language section.
    All languages get the same fields - just with localized values where available."""
    lines = text.split('\n')
    result = {}

    # First extract all fields from the full text (English base)
    english_fields = extract_fields(text)

    # Separate lines by language
    telugu_text = '\n'.join(l for l in lines if re.search(r'[\u0C00-\u0C7F]', l))
    hindi_text = '\n'.join(l for l in lines if re.search(r'[\u0900-\u097F]', l))

    # English always gets the full extracted fields
    if re.search(r'[A-Za-z]', text):
        result['English'] = english_fields.copy()

    # Telugu - start with English fields, then override with Telugu values where available
    if telugu_text and len(re.findall(r'[\u0C00-\u0C7F]', telugu_text)) > 10:
        telugu_fields = english_fields.copy()

        # Override Name with Telugu name
        telugu_lines_list = [l.strip() for l in telugu_text.split('\n') if l.strip()]
        telugu_name_candidates = []
        for l in telugu_lines_list:
            if re.search(r'(పుట్టిన\s*తేదీ|చిరునామా|పురుషుడు|స్త్రీ|S/O|D/O|W/O|మండల|జిల్లా|రాష్ట్ర|పిన్)', l, re.IGNORECASE):
                continue
            telugu_char_count = len(re.findall(r'[\u0C00-\u0C7F]', l))
            if telugu_char_count > 3 and len(l) < 40:
                telugu_name_candidates.append(l)

        if telugu_name_candidates:
            telugu_fields['Name'] = telugu_name_candidates[0]

        # Override Father's name with Telugu
        father_match = re.search(r'(?:S/O|D/O|W/O)\s+([\u0C00-\u0C7F\s,]+)', telugu_text)
        if father_match:
            fname = father_match.group(1).strip().rstrip(',')
            if len(fname) > 2:
                telugu_fields["Father's/Guardian's Name"] = fname

        # Override Gender with Telugu (fully in Telugu)
        if 'పురుషుడు' in telugu_text or telugu_fields.get('Gender', '').lower() == 'male':
            telugu_fields['Gender'] = 'పురుషుడు'
        elif 'స్త్రీ' in telugu_text or telugu_fields.get('Gender', '').lower() == 'female':
            telugu_fields['Gender'] = 'స్త్రీ'

        # Override Document Type in Telugu
        doc_type = telugu_fields.get('Document Type', '')
        if doc_type == 'Aadhaar Card':
            telugu_fields['Document Type'] = 'ఆధార్ కార్డ్'
        elif doc_type == 'PAN Card':
            telugu_fields['Document Type'] = 'పాన్ కార్డ్'
        elif doc_type == 'Passport':
            telugu_fields['Document Type'] = 'పాస్‌పోర్ట్'
        elif doc_type == 'Driving License':
            telugu_fields['Document Type'] = 'డ్రైవింగ్ లైసెన్స్'
        elif doc_type:
            telugu_fields['Document Type'] = 'పత్రం'

        # Override Address with Telugu address
        addr_match = re.search(r'చిరునామా\s*[:\-]?\s*(.+?)(?:\d{4}\s\d{4}\s\d{4}|$)', telugu_text, re.DOTALL)
        if addr_match:
            addr = re.sub(r'\s+', ' ', addr_match.group(1).strip())
            addr = re.sub(r'\d{4}\s\d{4}\s\d{4}.*', '', addr).strip()
            if len(addr) > 5:
                telugu_fields['Address'] = addr

        result['Telugu'] = telugu_fields

    # Hindi - start with English fields, then override with Hindi values where available
    if hindi_text and len(re.findall(r'[\u0900-\u097F]', hindi_text)) > 10:
        hindi_fields = english_fields.copy()

        # Override Name with Hindi
        hindi_name_lines = [l.strip() for l in hindi_text.split('\n')
                           if len(l.strip()) > 3
                           and not re.search(r'(जन्म|तिथि|पता|पुरुष|महिला|S/O|D/O|W/O)', l)]
        if hindi_name_lines:
            hindi_fields['Name'] = hindi_name_lines[0]

        # Override Father's name with Hindi
        father_match = re.search(r'(?:S/O|D/O|W/O)\s+([\u0900-\u097F\s,]+)', hindi_text)
        if father_match:
            fname = father_match.group(1).strip().rstrip(',')
            if len(fname) > 2:
                hindi_fields["Father's/Guardian's Name"] = fname

        # Override Gender with Hindi (fully in Hindi)
        if 'पुरुष' in hindi_text or hindi_fields.get('Gender', '').lower() == 'male':
            hindi_fields['Gender'] = 'पुरुष'
        elif 'महिला' in hindi_text or hindi_fields.get('Gender', '').lower() == 'female':
            hindi_fields['Gender'] = 'महिला'

        # Override Document Type in Hindi
        doc_type = hindi_fields.get('Document Type', '')
        if doc_type == 'Aadhaar Card':
            hindi_fields['Document Type'] = 'आधार कार्ड'
        elif doc_type == 'PAN Card':
            hindi_fields['Document Type'] = 'पैन कार्ड'
        elif doc_type == 'Passport':
            hindi_fields['Document Type'] = 'पासपोर्ट'
        elif doc_type == 'Driving License':
            hindi_fields['Document Type'] = 'ड्राइविंग लाइसेंस'
        elif doc_type:
            hindi_fields['Document Type'] = 'दस्तावेज़'

        result['Hindi'] = hindi_fields

    return result


def format_extracted_text(fields, lang='English'):
    """Format extracted fields into a clean readable text output.
    Labels are shown in the selected language."""

    # Field labels in different languages
    labels = {
        'English': {
            'Document Type': 'Document Type',
            'Name': 'Name',
            "Father's/Guardian's Name": "Father's/Guardian's Name",
            'Date of Birth': 'Date of Birth',
            'Gender': 'Gender',
            'Aadhaar Number': 'Aadhaar Number',
            'PAN Number': 'PAN Number',
            'VID': 'VID',
            'Mobile': 'Mobile',
            'Enrolment No.': 'Enrolment No.',
            'Address': 'Address',
            'PIN Code': 'PIN Code',
        },
        'Telugu': {
            'Document Type': 'పత్రం రకం',
            'Name': 'పేరు',
            "Father's/Guardian's Name": "తండ్రి/సంరక్షకుడి పేరు",
            'Date of Birth': 'పుట్టిన తేదీ',
            'Gender': 'లింగం',
            'Aadhaar Number': 'ఆధార్ నంబర్',
            'PAN Number': 'పాన్ నంబర్',
            'VID': 'VID',
            'Mobile': 'మొబైల్',
            'Enrolment No.': 'నమోదు సంఖ్య',
            'Address': 'చిరునామా',
            'PIN Code': 'పిన్ కోడ్',
        },
        'Hindi': {
            'Document Type': 'दस्तावेज़ प्रकार',
            'Name': 'नाम',
            "Father's/Guardian's Name": "पिता/अभिभावक का नाम",
            'Date of Birth': 'जन्म तिथि',
            'Gender': 'लिंग',
            'Aadhaar Number': 'आधार नंबर',
            'PAN Number': 'पैन नंबर',
            'VID': 'VID',
            'Mobile': 'मोबाइल',
            'Enrolment No.': 'नामांकन संख्या',
            'Address': 'पता',
            'PIN Code': 'पिन कोड',
        }
    }

    lang_labels = labels.get(lang, labels['English'])

    if not fields:
        return ""

    lines = []
    field_order = [
        'Document Type', 'Name', "Father's/Guardian's Name",
        'Date of Birth', 'Gender', 'Aadhaar Number', 'PAN Number',
        'VID', 'Mobile', 'Enrolment No.', 'Address', 'PIN Code'
    ]

    for key in field_order:
        if key in fields:
            label = lang_labels.get(key, key)
            lines.append(f"{label}: {fields[key]}")

    # Add any remaining fields not in the order
    for key, value in fields.items():
        if key not in field_order:
            label = lang_labels.get(key, key)
            lines.append(f"{label}: {value}")

    return '\n'.join(lines)


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
async def upload_file(file: UploadFile = File(...), language: str = Form(default="eng+hin+tel")):
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
        extracted_text = extract_text(filepath, language)

        if not extracted_text:
            return JSONResponse(
                status_code=400,
                content={'error': 'No text could be extracted from the image. Please try a clearer image.'}
            )

        # Extract important fields using regex
        fields = extract_fields(extracted_text)

        # Extract fields by language
        fields_by_language = extract_fields_by_language(extracted_text)

        # Build formatted text per language (structured output)
        text_by_language = {}
        for lang, lang_fields in fields_by_language.items():
            text_by_language[lang] = format_extracted_text(lang_fields, lang)

        # Detect languages present
        detected_languages = detect_languages(extracted_text)

        # Save results to output file
        output_filename = save_output(extracted_text, fields, filename)

        return JSONResponse(content={
            'success': True,
            'text': extracted_text,
            'fields': fields,
            'fields_by_language': fields_by_language,
            'text_by_language': text_by_language,
            'detected_languages': detected_languages,
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
