"""
Ensemble Fusion & Confidence Calibration — Agent 5 in the VeilNet pipeline.

Responsibilities:
  1. Merge findings from regex, NER, and entity linker
  2. Resolve conflicts when engines disagree on the same span
  3. Weighted ensemble scoring
  4. Smart deduplication (semantic, not just positional)
  5. Risk-level classification for every finding
  6. Final confidence calibration
"""

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# Risk Level Classification
# =============================================================================

RISK_LEVELS: Dict[str, str] = {
    # CRITICAL — immediate identity theft / financial risk
    "SSN":            "critical",
    "CREDIT_CARD":    "critical",
    "AADHAAR":        "critical",
    "PASSPORT":       "critical",
    "BANK_ACCOUNT":   "critical",

    # HIGH — strong PII that enables identification
    "NAME":           "high",
    "EMAIL":          "high",
    "PHONE":          "high",
    "PAN":            "high",
    "DATE_OF_BIRTH":  "high",
    "ADDRESS":        "high",
    "UPI_ID":         "high",
    "IBAN":           "high",

    # MEDIUM — identification possible with context
    "ORG":            "medium",
    "LOCATION":       "medium",
    "LINKEDIN":       "medium",
    "VEHICLE_REG":    "medium",
    "IFSC":           "medium",

    # LOW — minimal standalone risk
    "IP_ADDRESS":     "low",
    "URL":            "low",
    "PINCODE":        "low",
    "MISC":           "low",
}

# Source priorities — when two sources cover the same span
SOURCE_PRIORITY = {
    "REGEX":        90,   # Regex patterns are precise / high-confidence
    "PROPAGATION":  70,   # Propagated from high-confidence NER
    "NER":          60,   # Model-detected
    "SYNTHETIC":    40,   # Inferred from cross-entity heuristics
}


# =============================================================================
# Ensemble Scoring
# =============================================================================

def compute_ensemble_score(
    raw_score: float,
    source: str,
    spacy_agrees: bool = False,
    context_bonus: float = 0.0,
    section_bonus: float = 0.0,
    email_corroborated: bool = False,
    multi_model_votes: int = 1,
    total_models: int = 2,
) -> float:
    """
    Compute a calibrated final confidence score using weighted signals.

    Weights:
      - Raw model score:          50%
      - SpaCy agreement:          15%
      - Contextual keywords:      10%
      - Section sensitivity:      10%
      - Email corroboration:       5%
      - Multi-model consensus:    10%
    """
    # Base
    score = raw_score * 0.50

    # SpaCy agreement
    score += (0.15 if spacy_agrees else 0.0)

    # Context keywords (already 0.0 – 0.10 from NER engine)
    score += min(context_bonus, 0.10) * 1.0

    # Section sensitivity
    score += min(max(section_bonus, -0.10), 0.10) * 1.0

    # Email corroboration
    if email_corroborated:
        score += 0.05

    # Multi-model consensus
    if total_models > 1:
        consensus_ratio = multi_model_votes / total_models
        score += consensus_ratio * 0.10

    # Source-specific adjustments
    if source == "REGEX":
        score = max(score, 0.95)  # regex matches are near-certain
    elif source == "PROPAGATION":
        score = max(score, raw_score * 0.90)  # propagated entities maintain high conf

    # Clamp to [0.0, 0.9999]
    return round(min(max(score, 0.0), 0.9999), 4)


# =============================================================================
# Conflict Resolution
# =============================================================================

def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Check if two spans overlap."""
    return a_start < b_end and b_start < a_end


def _overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Calculate the overlap ratio between two spans."""
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    if overlap_start >= overlap_end:
        return 0.0
    overlap_len = overlap_end - overlap_start
    min_span_len = min(a_end - a_start, b_end - b_start)
    return overlap_len / max(min_span_len, 1)


def resolve_conflicts(findings: List[dict]) -> List[dict]:
    """
    When multiple findings cover overlapping spans, resolve conflicts:
    1. Higher source priority wins (REGEX > PROPAGATION > NER > SYNTHETIC)
    2. Among same priority, higher confidence wins
    3. More specific type wins (SSN > PHONE for overlapping digits)
    """
    if not findings:
        return []

    # Sort by start position, then by priority (highest first)
    findings.sort(key=lambda f: (
        f["start"],
        -SOURCE_PRIORITY.get(f.get("source", "NER"), 50),
        -f.get("score", 0),
    ))

    resolved: List[dict] = []

    for candidate in findings:
        c_start = candidate["start"]
        c_end = candidate["end"]

        # Check if this candidate overlaps with any already-resolved finding
        has_conflict = False
        for existing in resolved:
            e_start = existing["start"]
            e_end = existing["end"]

            if _spans_overlap(c_start, c_end, e_start, e_end):
                overlap = _overlap_ratio(c_start, c_end, e_start, e_end)

                if overlap > 0.5:  # Significant overlap
                    # Compare priorities
                    c_priority = SOURCE_PRIORITY.get(candidate.get("source", "NER"), 50)
                    e_priority = SOURCE_PRIORITY.get(existing.get("source", "NER"), 50)

                    if c_priority > e_priority:
                        # Candidate wins — replace existing
                        resolved.remove(existing)
                        resolved.append(candidate)
                    elif c_priority == e_priority and candidate.get("score", 0) > existing.get("score", 0):
                        resolved.remove(existing)
                        resolved.append(candidate)
                    # else: existing wins, skip candidate

                    has_conflict = True
                    break

        if not has_conflict:
            resolved.append(candidate)

    return resolved


# =============================================================================
# Smart Deduplication
# =============================================================================

def smart_deduplicate(findings: List[dict]) -> List[dict]:
    """
    Deduplicate findings using both positional and semantic comparison.

    Two findings are duplicates if:
      - Same type AND same start position (±3 chars), OR
      - Same type AND same value AND positions within 5 chars of each other
    """
    if not findings:
        return []

    # Sort by type, then start position
    findings.sort(key=lambda f: (f.get("type", ""), f["start"]))

    deduped: List[dict] = []
    seen_keys: Set[Tuple] = set()

    for finding in findings:
        ftype = finding.get("type", "")
        start = finding["start"]
        value = finding.get("value", "").strip().lower()

        # Positional key — same type within ±3 chars
        positional_dup = False
        for offset in range(-3, 4):
            key = (start + offset, ftype)
            if key in seen_keys:
                positional_dup = True
                break

        if positional_dup:
            continue

        # Semantic key — same value + type within 5 chars of an existing finding
        semantic_dup = False
        for existing in deduped:
            if (existing.get("type") == ftype and
                existing.get("value", "").strip().lower() == value and
                abs(existing["start"] - start) <= 5):
                semantic_dup = True
                break

        if semantic_dup:
            continue

        # Not a duplicate
        seen_keys.add((start, ftype))
        deduped.append(finding)

    return deduped


# =============================================================================
# Risk Level Assignment
# =============================================================================

def assign_risk_levels(findings: List[dict]) -> List[dict]:
    """Assign a risk level to every finding."""
    for finding in findings:
        ftype = finding.get("type", "")
        finding["risk_level"] = RISK_LEVELS.get(ftype, "low")
    return findings


# =============================================================================
# Risk Summary
# =============================================================================

def compute_risk_summary(findings: List[dict]) -> Dict:
    """Compute an aggregated risk summary for the response."""
    summary = {
        "total": len(findings),
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "by_type": {},
    }

    for finding in findings:
        risk = finding.get("risk_level", "low")
        summary[risk] = summary.get(risk, 0) + 1

        ftype = finding.get("type", "UNKNOWN")
        summary["by_type"][ftype] = summary["by_type"].get(ftype, 0) + 1

    # Overall risk assessment
    if summary["critical"] > 0:
        summary["overall_risk"] = "CRITICAL"
    elif summary["high"] >= 3:
        summary["overall_risk"] = "HIGH"
    elif summary["high"] > 0:
        summary["overall_risk"] = "ELEVATED"
    elif summary["medium"] > 0:
        summary["overall_risk"] = "MODERATE"
    else:
        summary["overall_risk"] = "LOW"

    return summary


# =============================================================================
# Full Ensemble Pipeline
# =============================================================================

def fuse_and_score(
    regex_findings: List[dict],
    ner_findings: List[dict],
    propagated_findings: List[dict],
) -> Tuple[List[dict], Dict]:
    """
    Full ensemble fusion pipeline:
      1. Tag sources
      2. Merge all findings
      3. Resolve conflicts
      4. Smart deduplicate
      5. Assign risk levels
      6. Compute risk summary

    Returns (final_findings, risk_summary).
    """
    # Tag sources
    for f in regex_findings:
        f.setdefault("source", "REGEX")
    for f in ner_findings:
        f.setdefault("source", "NER")
    for f in propagated_findings:
        f.setdefault("source", "PROPAGATION")

    # Merge
    all_findings = regex_findings + ner_findings + propagated_findings

    # Resolve conflicts on overlapping spans
    resolved = resolve_conflicts(all_findings)

    # Smart dedup
    deduped = smart_deduplicate(resolved)

    # Assign risk levels
    final = assign_risk_levels(deduped)

    # Sort by page (if present), then start position
    final.sort(key=lambda f: (f.get("page", 1), f["start"]))

    # Compute summary
    summary = compute_risk_summary(final)

    return final, summary
