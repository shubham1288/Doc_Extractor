---
title: Document Extractor
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Document Extractor - OCR Web Application

A Python-based OCR Document Extraction web application built with **FastAPI**, OpenCV, and Tesseract OCR. Upload document images (Aadhaar Card, PAN Card, Passport, or any text document) and automatically extract text using OCR.

## Features

- **Upload Document Images** - Drag-and-drop or click to upload JPG/JPEG/PNG/PDF files
- **Image Preprocessing** - Grayscale conversion, noise removal, and thresholding using OpenCV
- **OCR Text Extraction** - Extract text using Tesseract OCR engine
- **Smart Field Detection** - Regex-based extraction of:
  - Aadhaar Number (XXXX XXXX XXXX format)
  - PAN Number (ABCDE1234F format)
  - Date of Birth
  - Gender
  - Name
  - Father's/Guardian's Name
  - Address, PIN Code, Mobile, VID
- **Download Results** - Save extracted text as a .txt file
- **Copy to Clipboard** - One-click copy of extracted text
- **Dark Mode UI** - Modern, responsive Bootstrap-based interface

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.11 | Backend language |
| FastAPI | Web framework |
| OpenCV | Image preprocessing |
| pytesseract | OCR engine wrapper |
| Tesseract OCR | Text recognition engine |
| PyMuPDF | PDF handling |
| Bootstrap 5 | Frontend UI framework |
