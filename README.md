# Document Extractor - OCR Web Application

A Python-based OCR Document Extraction web application built with **FastAPI**, OpenCV, and Tesseract OCR. Upload document images (Aadhaar Card, PAN Card, Passport, or any text document) and automatically extract text using OCR.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-orange.svg)

## Features

- **Upload Document Images** - Drag-and-drop or click to upload JPG/JPEG/PNG files
- **Image Preprocessing** - Grayscale conversion, noise removal, and thresholding using OpenCV
- **OCR Text Extraction** - Extract text using Tesseract OCR engine
- **Smart Field Detection** - Regex-based extraction of:
  - Aadhaar Number (XXXX XXXX XXXX format)
  - PAN Number (ABCDE1234F format)
  - Date of Birth
  - Gender
  - Name
- **Download Results** - Save extracted text as a .txt file
- **Copy to Clipboard** - One-click copy of extracted text
- **Dark Mode UI** - Modern, responsive Bootstrap-based interface
- **Loading Spinner** - Visual feedback during extraction
- **Error Handling** - Proper validation for file types and sizes

## Project Structure

```
document_extractor/
│
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── uploads/            # Uploaded images (auto-created)
├── outputs/            # Extracted text files (auto-created)
├── static/
│   └── style.css       # Custom dark mode styles
└── templates/
    └── index.html      # Main HTML template
```

## Prerequisites

1. **Python 3.8+** - Download from [python.org](https://www.python.org/downloads/)
2. **Tesseract OCR** - Must be installed on your system

### Installing Tesseract OCR

#### Windows
1. Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (default path: `C:\Program Files\Tesseract-OCR\`)
3. The path is already configured in `app.py`. If you installed elsewhere, update:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install tesseract-ocr
```

#### macOS
```bash
brew install tesseract
```

## Setup Instructions

### 1. Navigate to the Project

```bash
cd document_extractor
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

### 5. Open in Browser

Navigate to: **http://127.0.0.1:8000**

## Usage

1. Open the application in your browser
2. Drag and drop a document image onto the upload area (or click to browse)
3. Click the **"Extract Text"** button
4. View the extracted fields and full text
5. Use **Copy** to copy text to clipboard or **Download** to save as .txt file

## Supported Document Types

- Aadhaar Card
- PAN Card
- Passport
- Driving License
- Any text-based document image

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.8+ | Backend language |
| FastAPI | Web framework |
| Uvicorn | ASGI server |
| OpenCV | Image preprocessing |
| pytesseract | OCR engine wrapper |
| Tesseract OCR | Text recognition engine |
| Bootstrap 5 | Frontend UI framework |
| Font Awesome | Icons |
| HTML/CSS/JS | Frontend |

## Tips for Better OCR Results

- Use high-resolution images (300 DPI or higher)
- Ensure good lighting and contrast
- Avoid blurry or skewed images
- Crop the document area if there's too much background
- Use scanned images rather than photos when possible

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "tesseract is not installed" | Install Tesseract OCR and set the path in app.py |
| Poor text extraction | Use a clearer, higher-resolution image |
| File upload fails | Check file is JPG/JPEG/PNG and under 16MB |
| Module not found | Run `pip install -r requirements.txt` |

## License

This project is open source and available for educational purposes.
