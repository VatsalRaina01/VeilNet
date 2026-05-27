"""
Document Intelligence Agent — Agent 1 in the VeilNet multi-agent pipeline.

Responsibilities:
  1. Classify document type (resume, invoice, legal, medical, financial, generic)
  2. Detect section boundaries from structural cues (bold headings, ALL-CAPS lines)
  3. Map each section to a PII-sensitivity level so downstream agents can boost
     or suppress confidence based on WHERE in the document an entity was found.

No additional ML model required — uses keyword fingerprinting and structural
heuristics that run in < 5 ms on any document.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# =============================================================================
# Document Type Fingerprints
# =============================================================================
# Each fingerprint is a set of keywords. The document type with the most keyword
# hits (weighted) wins.  Weights allow rare-but-definitive keywords to outweigh
# common ones: "whereas" alone is worth 3 generic "name:" matches.

_DOC_FINGERPRINTS: Dict[str, Dict[str, float]] = {
    "resume": {
        "experience": 2.0, "education": 2.0, "skills": 2.0,
        "resume": 3.0, "curriculum vitae": 3.0, "cv": 2.5,
        "projects": 1.5, "certifications": 1.5, "achievements": 1.5,
        "work history": 2.0, "employment": 1.5, "qualifications": 1.5,
        "objective": 1.0, "profile": 1.0, "references": 1.0,
        "internship": 1.5, "volunteer": 1.0, "extracurricular": 1.5,
        "gpa": 2.0, "cgpa": 2.5, "linkedin": 1.5, "github": 1.5,
        "portfolio": 1.0,
    },
    "invoice": {
        "invoice": 3.0, "bill to": 2.5, "ship to": 2.0,
        "subtotal": 2.5, "total": 1.5, "tax": 1.5, "gst": 2.0,
        "gstin": 3.0, "invoice no": 3.0, "invoice date": 2.5,
        "due date": 2.0, "payment terms": 2.0, "qty": 1.5,
        "unit price": 2.0, "amount due": 2.5, "balance due": 2.5,
        "purchase order": 2.0, "po number": 2.0,
    },
    "legal": {
        "whereas": 3.0, "hereby": 2.5, "herein": 2.5,
        "jurisdiction": 2.5, "governed by": 2.5, "clause": 2.0,
        "article": 1.5, "agreement": 2.0, "contract": 2.0,
        "party": 1.5, "plaintiff": 3.0, "defendant": 3.0,
        "witness": 2.0, "affidavit": 3.0, "notarized": 3.0,
        "indemnify": 2.5, "liability": 2.0, "arbitration": 2.5,
        "tribunal": 2.5, "statute": 2.0, "breach": 2.0,
        "termination": 1.5, "confidentiality": 2.0,
    },
    "medical": {
        "patient": 2.5, "diagnosis": 3.0, "prescription": 3.0,
        "icd": 2.5, "medication": 2.5, "dosage": 2.5,
        "blood pressure": 2.5, "pulse": 1.5, "allergy": 2.0,
        "symptoms": 2.0, "prognosis": 2.5, "physician": 2.0,
        "hospital": 1.5, "medical record": 3.0, "lab results": 2.5,
        "pathology": 2.5, "radiology": 2.5, "immunization": 2.0,
        "discharge summary": 3.0, "chief complaint": 2.5,
    },
    "financial": {
        "account number": 3.0, "account no": 3.0, "balance": 2.0,
        "transaction": 2.5, "ifsc": 3.0, "routing number": 3.0,
        "bank statement": 3.0, "credit": 1.5, "debit": 1.5,
        "interest rate": 2.0, "emi": 2.5, "loan": 2.0,
        "mortgage": 2.5, "investment": 1.5, "portfolio": 1.0,
        "dividend": 2.0, "mutual fund": 2.5, "neft": 2.5,
        "rtgs": 2.5, "swift": 2.0, "micr": 2.5,
    },
}

# =============================================================================
# Section Sensitivity Mapping
# =============================================================================
# Sensitivity levels:  high → PII very likely,  medium → might be PII,
#                      low  → usually NOT PII,  neutral → no adjustment

SECTION_SENSITIVITY: Dict[str, str] = {
    # HIGH — entities found here are almost certainly PII
    "contact": "high", "contact information": "high",
    "personal details": "high", "personal information": "high",
    "personal": "high", "about me": "high", "bio": "high",
    "patient information": "high", "bill to": "high", "ship to": "high",
    "address": "high", "contact details": "high",

    # MEDIUM — mix of PII and non-PII
    "experience": "medium", "work experience": "medium",
    "employment history": "medium", "work history": "medium",
    "professional experience": "medium", "education": "medium",
    "references": "medium", "emergency contact": "high",

    # LOW — entities here are rarely PII
    "skills": "low", "technical skills": "low",
    "technologies": "low", "tools": "low", "frameworks": "low",
    "projects": "low", "certifications": "low",
    "achievements": "low", "awards": "low",
    "publications": "low", "interests": "low", "hobbies": "low",
    "competencies": "low", "strengths": "low",
    "core competencies": "low", "key skills": "low",

    # NEUTRAL — no adjustment
    "summary": "neutral", "objective": "neutral",
    "profile": "neutral", "overview": "neutral",
}

# Confidence adjustments per sensitivity level
SENSITIVITY_BONUS: Dict[str, float] = {
    "high":    +0.12,
    "medium":  +0.00,
    "low":     -0.15,
    "neutral":  0.00,
}


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class Section:
    """Represents a detected section within the document."""
    name: str               # e.g. "Work Experience"
    start: int              # character offset in full text
    end: int                # character offset in full text
    sensitivity: str        # "high", "medium", "low", "neutral"


@dataclass
class DocumentProfile:
    """Complete intelligence about a document's structure."""
    doc_type: str                     # "resume", "invoice", etc.
    doc_type_confidence: float        # 0.0 – 1.0
    sections: List[Section] = field(default_factory=list)
    type_scores: Dict[str, float] = field(default_factory=dict)

    def get_section_at(self, char_offset: int) -> Optional[Section]:
        """Return the section containing a given character offset."""
        for section in self.sections:
            if section.start <= char_offset < section.end:
                return section
        return None

    def get_sensitivity_at(self, char_offset: int) -> str:
        """Return sensitivity level at a given character offset."""
        section = self.get_section_at(char_offset)
        return section.sensitivity if section else "neutral"

    def get_confidence_bonus_at(self, char_offset: int) -> float:
        """Return the confidence adjustment for a given position."""
        sensitivity = self.get_sensitivity_at(char_offset)
        return SENSITIVITY_BONUS.get(sensitivity, 0.0)


# =============================================================================
# Classification Logic
# =============================================================================

def classify_document(text: str, bold_phrases: List[str] = None) -> DocumentProfile:
    """
    Classify a document's type and map its sections.

    Args:
        text: Full extracted text of the document.
        bold_phrases: List of bold text spans (from PyMuPDF) — used to
                      identify section headings with higher confidence.

    Returns:
        DocumentProfile with type, confidence, and section map.
    """
    if bold_phrases is None:
        bold_phrases = []

    text_lower = text.lower()

    # --- Step 1: Score each document type ---
    type_scores: Dict[str, float] = {}
    for doc_type, keywords in _DOC_FINGERPRINTS.items():
        score = 0.0
        for keyword, weight in keywords.items():
            # Count occurrences (capped at 3 to prevent long docs from inflating)
            count = min(text_lower.count(keyword), 3)
            score += count * weight
        type_scores[doc_type] = score

    # Determine winner
    if not type_scores or max(type_scores.values()) == 0:
        doc_type = "generic"
        doc_confidence = 0.3
    else:
        total = sum(type_scores.values()) or 1.0
        doc_type = max(type_scores, key=type_scores.get)
        doc_confidence = min(type_scores[doc_type] / total, 0.99)

        # If the top score is very low absolute, fall back to generic
        if type_scores[doc_type] < 5.0:
            doc_type = "generic"
            doc_confidence = 0.3

    # --- Step 2: Detect section boundaries ---
    sections = _detect_sections(text, bold_phrases)

    return DocumentProfile(
        doc_type=doc_type,
        doc_type_confidence=round(doc_confidence, 4),
        sections=sections,
        type_scores={k: round(v, 2) for k, v in type_scores.items()},
    )


# =============================================================================
# Section Detection
# =============================================================================

# Pattern: A short line (1-5 words) that is either ALL CAPS, or Title Case ending with a colon
_HEADING_PATTERN = re.compile(
    r'^[ \t]*'                      # optional leading whitespace
    r'('
    r'[A-Z][A-Z\s&/\-]{2,50}'      # ALL CAPS heading (min 3 chars)
    r'|'
    r'(?:[A-Z][a-z]+(?:\s+[A-Za-z]+){0,4}):'  # Title Case heading MUST end with colon
    r')'
    r'[ \t]*$',            # optional trailing whitespace
    re.MULTILINE,
)


def _detect_sections(text: str, bold_phrases: List[str]) -> List[Section]:
    """
    Detect section boundaries using headings found via regex and bold phrases.
    """
    candidates: List[tuple] = []  # (start, end, heading_text)

    # Strategy 1: Regex-detected headings (ALL CAPS or Title Case on own line)
    for match in _HEADING_PATTERN.finditer(text):
        heading = match.group(1).strip().rstrip(":")
        if len(heading) < 3 or len(heading) > 60:
            continue
        # Skip headings that are clearly content, not section headers
        words = heading.split()
        if len(words) > 6:
            continue
        candidates.append((match.start(), match.end(), heading))

    # Strategy 2: Bold phrases that match known section names
    bold_set = set()
    for phrase in bold_phrases:
        cleaned = phrase.strip().rstrip(":")
        if len(cleaned) < 3 or len(cleaned) > 60:
            continue
        # Check if this bold phrase is a known section name
        if cleaned.lower() in SECTION_SENSITIVITY:
            # Find its position in text
            idx = text.find(phrase)
            if idx >= 0 and (idx, cleaned) not in bold_set:
                candidates.append((idx, idx + len(phrase), cleaned))
                bold_set.add((idx, cleaned))

    # Sort by position
    candidates.sort(key=lambda c: c[0])

    # Deduplicate overlapping candidates (keep the one found first)
    deduped: List[tuple] = []
    for cand in candidates:
        if not deduped or cand[0] >= deduped[-1][1]:
            deduped.append(cand)

    # Build sections: each section runs from its heading to the next heading
    sections: List[Section] = []
    for i, (start, end, heading) in enumerate(deduped):
        heading_lower = heading.lower().strip()

        # Determine sensitivity
        sensitivity = SECTION_SENSITIVITY.get(heading_lower, "neutral")

        # Also try partial matches for compound headings
        if sensitivity == "neutral":
            for key, sens in SECTION_SENSITIVITY.items():
                if key in heading_lower or heading_lower in key:
                    sensitivity = sens
                    break

        # Section extends to the start of the next heading (or end of text)
        section_end = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)

        sections.append(Section(
            name=heading,
            start=start,
            end=section_end,
            sensitivity=sensitivity,
        ))

    return sections
