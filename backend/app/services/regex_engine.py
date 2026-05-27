"""
Enhanced Regex Engine — Agent 2 in the VeilNet multi-agent pipeline.

Upgrades over the original:
  • 20+ PII patterns (was 11)
  • India-specific: Aadhaar, PAN, Passport, Vehicle Reg, IFSC, UPI ID
  • Contextual gating  — high-FP patterns only fire when context keywords are nearby
  • Verhoeff checksum validation for Aadhaar
  • PAN category-letter validation
  • Background unchanged: overlap deduplication, priority ordering
"""

import re
from typing import List, Dict, Optional


# =============================================================================
# Verhoeff Checksum (for Aadhaar validation)
# =============================================================================

_VERHOEFF_D = [
    [0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0],
]

_VERHOEFF_P = [
    [0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8],
]

_VERHOEFF_INV = [0,4,3,2,1,5,6,7,8,9]


def _verhoeff_checksum(number: str) -> bool:
    """Validate a number string using the Verhoeff algorithm."""
    digits = [int(d) for d in reversed(number.replace(" ", ""))]
    c = 0
    for i, digit in enumerate(digits):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][digit]]
    return c == 0


# =============================================================================
# PII Detection Patterns
# =============================================================================

PATTERNS = {
    # ─── Identity ────────────────────────────────────────────────────────

    # Title-prefixed names (Mr. John Smith, Dr. Priya Sharma)
    "NAME": (
        r'\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Rev|Sr|Jr|Sir|Madam|Smt|Shri)'
        r'\.?[ \t]+'
        r'[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){1,3}\b'
    ),

    "EMAIL": r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}',

    # Phone: requires either a + country code prefix OR parens around area code
    "PHONE": (
        r'(?:'
        r'\+\d{1,3}[\-.\s]?\(?\d{2,5}\)?[\-.\s]?\d{3,5}[\-.\s]?\d{3,5}'
        r'|\(\d{3}\)[\-.\s]?\d{3}[\-.\s]?\d{4}'
        r'|\b\d{3}[\-.]?\d{3}[\-.]\d{4}\b'
        r')'
    ),

    "SSN": r'\b\d{3}-\d{2}-\d{4}\b',

    # ─── Financial ───────────────────────────────────────────────────────

    "CREDIT_CARD": (
        r'\b(?:'
        r'4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}'           # Visa
        r'|5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}'     # Mastercard
        r'|3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}'                  # Amex
        r'|6(?:011|5\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}'  # Discover
        r')\b'
    ),

    "IBAN": r'\b[A-Z]{2}\d{2}\s?[\dA-Z]{4}\s?(?:[\dA-Z]{4}\s?){2,7}[\dA-Z]{1,4}\b',

    # ─── India-Specific ─────────────────────────────────────────────────

    # Aadhaar — 12 digits, optionally separated by spaces
    # Contextual: only fire if "aadhaar", "uid", or "aadhar" is nearby
    "AADHAAR": r'\b\d{4}\s?\d{4}\s?\d{4}\b',

    # PAN Card — ABCDE1234F format (3rd char encodes category)
    "PAN": r'\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b',

    # Indian Passport — Letter followed by 7 digits
    # Contextual: only fire near "passport" keyword
    "PASSPORT": r'\b[A-Z]\d{7}\b',

    # Vehicle Registration — e.g. MH02AB1234, KA01MV1234
    "VEHICLE_REG": r'\b[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}\b',

    # IFSC Code — 4 letters + 0 + 6 alphanumeric
    "IFSC": r'\b[A-Z]{4}0[A-Z0-9]{6}\b',

    # UPI ID — with known handle suffixes
    "UPI_ID": (
        r'\b[\w.\-]+@(?:'
        r'ybl|okaxis|okhdfcbank|okicici|oksbi|paytm|apl|ibl'
        r'|upi|axl|sbi|hdfcbank|icici|kotak|boi|pnb|bandhan'
        r'|indus|federal|rbl|idbi|citi|hsbc|sc|dbs'
        r')\b'
    ),

    # Bank Account Number — 9-18 digits, contextual
    "BANK_ACCOUNT": r'\b\d{9,18}\b',

    # ─── Dates ───────────────────────────────────────────────────────────

    "DATE_OF_BIRTH": (
        r'(?:DOB|Date[ \t]*of[ \t]*Birth|Birth[ \t]*Date|Born|Birthday)'
        r'[ \t]*[:\-]?[ \t]*'
        r'(\b(?:'
        r'(?:0[1-9]|[12]\d|3[01])[/\-.](?:0[1-9]|1[0-2])[/\-.](?:19|20)\d{2}'
        r'|(?:0[1-9]|1[0-2])[/\-.](?:0[1-9]|[12]\d|3[01])[/\-.](?:19|20)\d{2}'
        r'|\d{4}[/\-.](?:0[1-9]|1[0-2])[/\-.](?:0[1-9]|[12]\d|3[01])'
        r')\b)'
    ),

    # ─── Digital / Network ───────────────────────────────────────────────

    "IP_ADDRESS": (
        r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
        r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
    ),

    "URL": r'https?://[^\s<>"\']+',

    "LINKEDIN": r'(?:linkedin\.com/in/[a-zA-Z0-9\-]+)',

    # ─── Location / Address ──────────────────────────────────────────────

    "ADDRESS": (
        r'\b\d{1,5}[ \t]+(?:[A-Z][a-z]+[ \t]*){1,4}'
        r'(?:Street|St|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Road|Rd|'
        r'Lane|Ln|Court|Ct|Place|Pl|Way|Circle|Cir|Terrace|Ter)'
        r'\.?\b'
    ),

    # Indian Pincode — 6 digits, contextual
    "PINCODE": r'\b[1-9]\d{5}\b',
}


# =============================================================================
# Contextual Gating
# =============================================================================
# Some patterns are too generic to fire unconditionally. They require context
# keywords within a window around the match to reduce false positives.

_CONTEXTUAL_GATES: Dict[str, Dict] = {
    "AADHAAR": {
        "keywords": [
            "aadhaar", "aadhar", "uid", "uidai", "unique identification",
            "enrolment", "enrollment", "identity", "aadhaar no", "aadhaar number",
        ],
        "window": 300,          # chars before/after to search
        "require_checksum": True,
    },
    "PASSPORT": {
        "keywords": [
            "passport", "travel document", "passport no", "passport number",
        ],
        "window": 200,
    },
    "BANK_ACCOUNT": {
        "keywords": [
            "account", "a/c", "bank", "saving", "current", "account no",
            "account number", "acct", "neft", "rtgs",
        ],
        "window": 200,
    },
    "PINCODE": {
        "keywords": [
            "pin", "pincode", "pin code", "postal", "zip", "post office",
            "address", "city", "state", "district",
        ],
        "window": 200,
    },
    "VEHICLE_REG": {
        "keywords": [
            "vehicle", "registration", "reg no", "car", "bike", "number plate",
            "rto", "rc", "registration certificate",
        ],
        "window": 250,
    },
}


# =============================================================================
# Contextual Name Patterns (Strategy 2 — keyword-triggered)
# =============================================================================

_NAME_CONTEXT_PATTERNS = [
    # "Name: John Smith", "User: Jane Doe", "Patient: Bob Ross" etc.
    (
        r'(?:'
        r'(?:Full[ \t]*)?Name|User|Patient|Client|Applicant|Employee|Author|'
        r'Recipient|Sender|Contact|Candidate|Witness|Beneficiary|'
        r'Account[ \t]*Holder|Member|Subscriber|Insured|Tenant|Owner|'
        r'Manager|Guardian|Representative|Guarantor|Policyholder|'
        r'Payee|Payer|Borrower|Lender|Debtor|Creditor|Assignee|'
        r'Claimant|Respondent|Petitioner|Defendant|Plaintiff'
        r')'
        r'[ \t]*[:][ \t]*'
        r'([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){1,3})'
    ),
    # "First Name: John" style
    (
        r'(?:First|Last|Middle|Given|Sur|Family)[ \t]*Name'
        r'[ \t]*[:][ \t]*'
        r'([A-Z][a-z]+)'
    ),
    # "Dear John Smith," (letter/email greeting)
    (
        r'(?:Dear|Attn|Attention)[ \t]*:?[ \t]*'
        r'([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){1,3})'
    ),
    # "Signed by John Smith" or "Authorized by Jane Doe"
    (
        r'(?:Signed|Authorized|Approved|Prepared|Reviewed|Submitted)'
        r'[ \t]*(?:by)[ \t]*:?[ \t]*'
        r'([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){1,3})'
    ),
    # "Father's Name: Ramesh Kumar", "Mother's Name: Priya Sharma"
    (
        r"(?:Father|Mother|Spouse|Guardian|Husband|Wife)(?:'s)?[ \t]*Name"
        r'[ \t]*[:][ \t]*'
        r'([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){1,3})'
    ),
    # "S/O, D/O, W/O" — Indian legal convention
    (
        r'(?:S/O|D/O|W/O|C/O|s/o|d/o|w/o|c/o)\.?[ \t]*:?[ \t]*'
        r'([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){1,3})'
    ),
]


# =============================================================================
# Overlap Detection
# =============================================================================

def _overlaps(new_start: int, new_end: int, existing_results: List[dict]) -> bool:
    """Check if a match overlaps with any higher-priority existing match."""
    for r in existing_results:
        if new_start < r["end"] and new_end > r["start"]:
            return True
    return False


# =============================================================================
# Contextual Gate Check
# =============================================================================

def _passes_context_gate(
    label: str,
    match_start: int,
    match_end: int,
    text: str,
    matched_value: str,
) -> bool:
    """
    Check if a match passes its contextual gate (if one exists).
    Returns True if no gate exists (unconditional pattern) or if gate passes.
    """
    gate = _CONTEXTUAL_GATES.get(label)
    if gate is None:
        return True  # No gate — always pass

    window = gate.get("window", 200)
    keywords = gate.get("keywords", [])

    # Look for keywords in a window around the match
    window_start = max(0, match_start - window)
    window_end = min(len(text), match_end + window)
    context = text[window_start:window_end].lower()

    has_keyword = any(kw in context for kw in keywords)
    if not has_keyword:
        return False

    # Special: Aadhaar checksum validation
    if label == "AADHAAR" and gate.get("require_checksum"):
        digits_only = matched_value.replace(" ", "")
        if len(digits_only) == 12 and digits_only.isdigit():
            if not _verhoeff_checksum(digits_only):
                return False

    return True


# =============================================================================
# PAN Validation
# =============================================================================

_VALID_PAN_CATEGORIES = set("ABCFGHLJPT")  # 4th character of PAN


def _is_valid_pan(value: str) -> bool:
    """Validate PAN card format — 4th character must be a valid category."""
    if len(value) != 10:
        return False
    return value[3] in _VALID_PAN_CATEGORIES


# =============================================================================
# Main Pattern Matching
# =============================================================================

def find_matches(text: str) -> List[dict]:
    """
    Scans text for all defined PII patterns and returns a list of results.
    Combines regex patterns with contextual name detection.
    Uses overlap detection, contextual gating, and validation to minimize FPs.
    """
    results: List[dict] = []

    # Run all standard regex patterns in priority order
    # More specific patterns first to prevent generic ones from stealing matches
    priority_order = [
        "SSN", "CREDIT_CARD", "IBAN",
        "AADHAAR", "PAN", "PASSPORT", "IFSC", "UPI_ID", "BANK_ACCOUNT",
        "VEHICLE_REG",
        "EMAIL", "LINKEDIN",
        "NAME", "ADDRESS", "DATE_OF_BIRTH",
        "IP_ADDRESS", "URL", "PHONE",
        "PINCODE",
    ]

    for label in priority_order:
        if label not in PATTERNS:
            continue

        pattern = PATTERNS[label]
        for m in re.finditer(pattern, text):
            matched_value = m.group(1) if label == "DATE_OF_BIRTH" and m.lastindex else m.group()
            match_start = m.start(1) if label == "DATE_OF_BIRTH" and m.lastindex else m.start()
            match_end = m.end(1) if label == "DATE_OF_BIRTH" and m.lastindex else m.end()

            # Skip if overlaps with a higher-priority match
            if _overlaps(match_start, match_end, results):
                continue

            # Contextual gating
            if not _passes_context_gate(label, match_start, match_end, text, matched_value):
                continue

            # PAN-specific validation
            if label == "PAN" and not _is_valid_pan(matched_value):
                continue

            results.append({
                "type": label,
                "value": matched_value,
                "start": match_start,
                "end": match_end,
                "source": "REGEX",
            })

    # Strategy 2: Contextual keyword name patterns
    for pattern in _NAME_CONTEXT_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            name_value = m.group(1).strip()
            if len(name_value) > 2 and not _overlaps(m.start(1), m.end(1), results):
                results.append({
                    "type": "NAME",
                    "value": name_value,
                    "start": m.start(1),
                    "end": m.end(1),
                    "source": "REGEX",
                })

    return results