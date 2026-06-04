"""Document classifier for Indian KYC documents.

Determines document type from OCR text using pattern matching and
weighted confidence scoring.
"""

import re

from app.models.document import ClassificationResult, DocumentType
from app.models.ocr import OCRResult


def _normalize_text(text: str) -> str:
    """Normalize text for classification.

    Converts to lowercase, collapses multiple whitespace characters
    into a single space, and strips leading/trailing whitespace.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class DocumentClassifier:
    """Classifies Indian KYC documents from OCR text.

    Uses keyword matching and regex pattern matching to score text
    against known document patterns. Returns the highest-scoring
    document type with its confidence score and matched patterns.
    """

    def classify(self, ocr_result: OCRResult) -> ClassificationResult:
        """Determine document type from extracted text.

        Scores the OCR text against all supported document types and
        returns the best match. If the best score is below 0.5,
        returns UNKNOWN.

        Args:
            ocr_result: The OCR extraction result containing text.

        Returns:
            ClassificationResult with document type, confidence, and
            matched patterns.
        """
        text = _normalize_text(ocr_result.text)

        scores: list[tuple[DocumentType, float, list[str]]] = [
            (DocumentType.AADHAAR_FRONT, *self._score_aadhaar_front(text)),
            (DocumentType.AADHAAR_BACK, *self._score_aadhaar_back(text)),
            (DocumentType.PAN_CARD, *self._score_pan(text)),
            (DocumentType.DRIVING_LICENSE, *self._score_driving_license(text)),
        ]

        # Select highest scoring type
        best_type, best_score, best_patterns = max(scores, key=lambda x: x[1])

        if best_score < 0.5:
            return ClassificationResult(
                document_type=DocumentType.UNKNOWN,
                confidence=best_score,
                matched_patterns=[],
            )

        # Cap confidence at 1.0
        confidence = min(best_score, 1.0)

        return ClassificationResult(
            document_type=best_type,
            confidence=confidence,
            matched_patterns=best_patterns,
        )

    def _score_aadhaar_front(self, text: str) -> tuple[float, list[str]]:
        """Score likelihood of Aadhaar front card.

        Checks for:
        - Aadhaar number pattern (4-4-4 digit format): weight 0.3
        - UIDAI / unique identification keywords: weight 0.2
        - Government of India keyword: weight 0.15
        - Gender keywords (male/female/transgender): weight 0.1
        - DOB keywords (dob/date of birth/year of birth): weight 0.1
        """
        score = 0.0
        patterns: list[str] = []

        # Aadhaar number pattern: 4 digits space/dash 4 digits space/dash 4 digits
        if re.search(r"\d{4}[\s\-]?\d{4}[\s\-]?\d{4}", text):
            score += 0.3
            patterns.append("aadhaar_number_pattern")

        # UIDAI / unique identification
        if re.search(r"uidai|unique\s*identification", text):
            score += 0.2
            patterns.append("uidai_keyword")

        # Government of India
        if re.search(r"government\s*of\s*india|govt\s*of\s*india", text):
            score += 0.15
            patterns.append("government_of_india")

        # Gender keywords
        if re.search(r"\b(male|female|transgender)\b", text):
            score += 0.1
            patterns.append("gender_keyword")

        # DOB keywords
        if re.search(r"\b(dob|date\s*of\s*birth|year\s*of\s*birth)\b", text):
            score += 0.1
            patterns.append("dob_keyword")

        return score, patterns

    def _score_aadhaar_back(self, text: str) -> tuple[float, list[str]]:
        """Score likelihood of Aadhaar back card.

        Checks for:
        - Aadhaar number pattern: weight 0.25
        - Address keyword: weight 0.25
        - Pincode pattern (6 digits): weight 0.15
        - UIDAI keyword: weight 0.15
        """
        score = 0.0
        patterns: list[str] = []

        # Aadhaar number pattern
        if re.search(r"\d{4}[\s\-]?\d{4}[\s\-]?\d{4}", text):
            score += 0.25
            patterns.append("aadhaar_number_pattern")

        # Address keyword
        if re.search(r"\b(address|s/o|d/o|w/o|c/o|house|street|lane|nagar|colony|district|village|post office)\b", text):
            score += 0.25
            patterns.append("address_keyword")

        # Pincode pattern (6-digit number)
        if re.search(r"\b\d{6}\b", text):
            score += 0.15
            patterns.append("pincode_pattern")

        # UIDAI keyword
        if re.search(r"uidai|unique\s*identification", text):
            score += 0.15
            patterns.append("uidai_keyword")

        return score, patterns

    def _score_pan(self, text: str) -> tuple[float, list[str]]:
        """Score likelihood of PAN card.

        Checks for:
        - PAN number pattern [A-Z]{5}[0-9]{4}[A-Z]: weight 0.35
        - Income Tax / Permanent Account keywords: weight 0.25
        - Father's name keyword: weight 0.15
        - Government of India keyword: weight 0.1
        """
        score = 0.0
        patterns: list[str] = []

        # PAN number pattern (case-insensitive since text is normalized)
        if re.search(r"[a-z]{5}\d{4}[a-z]", text):
            score += 0.35
            patterns.append("pan_number_pattern")

        # Income Tax / Permanent Account
        if re.search(r"income\s*tax|permanent\s*account", text):
            score += 0.25
            patterns.append("income_tax_keyword")

        # Father's name keyword
        if re.search(r"\bfather|father\'?s?\s*name\b", text):
            score += 0.15
            patterns.append("father_keyword")

        # Government of India
        if re.search(r"government\s*of\s*india|govt\s*of\s*india", text):
            score += 0.1
            patterns.append("government_of_india")

        return score, patterns

    def _score_driving_license(self, text: str) -> tuple[float, list[str]]:
        """Score likelihood of Driving License.

        Checks for:
        - Driving/licence/license keyword: weight 0.25
        - Transport/RTO keyword: weight 0.2
        - Validity/expiry/issue date keywords: weight 0.15
        - Motor vehicle keyword: weight 0.15
        - DL number pattern (state code + digits): weight 0.15
        """
        score = 0.0
        patterns: list[str] = []

        # Driving / licence / license keyword
        if re.search(r"\b(driving|licence|license)\b", text):
            score += 0.25
            patterns.append("driving_licence_keyword")

        # Transport / RTO keyword
        if re.search(r"\b(transport|rto|regional\s*transport)\b", text):
            score += 0.2
            patterns.append("transport_keyword")

        # Validity / expiry / issue date keywords
        if re.search(r"\b(validity|expiry|issue\s*date|valid\s*(from|till|upto|until))\b", text):
            score += 0.15
            patterns.append("validity_keyword")

        # Motor vehicle keyword
        if re.search(r"motor\s*vehicle|vehicle\s*class|class\s*of\s*vehicle", text):
            score += 0.15
            patterns.append("motor_vehicle_keyword")

        # DL number pattern: state code (2 letters) + optional dash + digits
        if re.search(r"\b[a-z]{2}[\s\-]?\d{2}[\s\-]?\d{4,}", text):
            score += 0.15
            patterns.append("dl_number_pattern")

        return score, patterns
