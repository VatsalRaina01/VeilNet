"""
Cross-Entity Intelligence Agent — Agent 4 in the VeilNet multi-agent pipeline.

Responsibilities:
  1. Email → Name extraction  (vinit.raina@gmail.com → "Vinit Raina")
  2. Entity propagation       (high-confidence name → tag all other occurrences)
  3. Name variant detection    ("Vinit Raina", "V. Raina", "VINIT RAINA" → same entity)
  4. Cross-entity inference    (email username matches nearby NER name → boost both)
  5. Entity consistency        (ORG on page 1 → ensure ORG on page 5 too)

This agent runs AFTER the NER engine and BEFORE ensemble fusion.  It takes the
raw findings from regex + NER and enriches them with relational intelligence.
"""

from __future__ import annotations
import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class EntityCluster:
    """A group of findings that refer to the same real-world entity."""
    canonical_name: str          # Best representation (longest, highest-confidence)
    entity_type: str             # NAME, ORG, LOCATION, etc.
    variants: Set[str] = field(default_factory=set)  # All surface forms
    positions: List[int] = field(default_factory=list)  # Start positions
    max_confidence: float = 0.0


# =============================================================================
# Email → Name Extraction
# =============================================================================

# Common email separators: firstname.lastname, firstname_lastname, firstnamelastname
_EMAIL_NAME_PATTERN = re.compile(
    r'^([a-zA-Z]+)[._]([a-zA-Z]+)@',  # firstname.lastname@ or firstname_lastname@
)

_EMAIL_SINGLE_PATTERN = re.compile(
    r'^([a-zA-Z]{2,})@',  # single word before @
)

# Common generic email prefixes that are NOT names
_GENERIC_PREFIXES = {
    "info", "admin", "support", "help", "contact", "sales", "hr",
    "noreply", "no-reply", "hello", "team", "office", "billing",
    "service", "enquiry", "inquiry", "feedback", "mail", "test",
    "webmaster", "postmaster", "hostmaster", "abuse",
}


def extract_names_from_emails(findings: List[dict]) -> List[dict]:
    """
    Extract candidate person names from email addresses found by regex.

    Returns synthetic NAME findings with moderate confidence that can
    boost or create NER detections.
    """
    synthetic_names = []

    for finding in findings:
        if finding.get("type") != "EMAIL":
            continue

        email = finding["value"].lower()

        # Try firstname.lastname pattern
        match = _EMAIL_NAME_PATTERN.match(email)
        if match:
            first = match.group(1).capitalize()
            last = match.group(2).capitalize()

            # Skip generic prefixes
            if first.lower() in _GENERIC_PREFIXES or last.lower() in _GENERIC_PREFIXES:
                continue

            # Skip very short fragments (likely initials or abbreviations)
            if len(first) < 2 or len(last) < 2:
                continue

            full_name = f"{first} {last}"
            synthetic_names.append({
                "value": full_name,
                "first_name": first,
                "last_name": last,
                "source_email": finding["value"],
                "confidence": 0.70,  # moderate — needs NER confirmation
            })
            continue

        # Try single-word pattern
        match = _EMAIL_SINGLE_PATTERN.match(email)
        if match:
            word = match.group(1)
            if word.lower() not in _GENERIC_PREFIXES and len(word) >= 3:
                synthetic_names.append({
                    "value": word.capitalize(),
                    "first_name": word.capitalize(),
                    "last_name": None,
                    "source_email": finding["value"],
                    "confidence": 0.45,  # low — single word, needs strong NER backup
                })

    return synthetic_names


# =============================================================================
# Name Variant Detection
# =============================================================================

def _normalize_name(name: str) -> str:
    """Normalize a name for comparison: lowercase, strip titles, collapse spaces."""
    name = name.strip().lower()
    # Remove common titles
    for title in ("mr.", "mrs.", "ms.", "dr.", "prof.", "sir", "smt.", "shri"):
        name = name.replace(title, "").strip()
    # Collapse whitespace
    name = re.sub(r'\s+', ' ', name)
    return name


def _name_parts(name: str) -> Set[str]:
    """Extract meaningful parts of a name."""
    normalized = _normalize_name(name)
    parts = set()
    for p in normalized.split():
        # Skip single-char initials like "j" from "J."
        if len(p) >= 2:
            parts.add(p.rstrip("."))
    return parts


def are_name_variants(name_a: str, name_b: str) -> bool:
    """
    Check if two name strings are likely variants of the same person.

    Examples that should match:
      - "Vinit Raina" ↔ "V. Raina"
      - "Mr. Vinit Raina" ↔ "Vinit Raina"
      - "VINIT RAINA" ↔ "Vinit Raina"

    Examples that should NOT match:
      - "Vinit Raina" ↔ "Priya Sharma"
      - "Vinit" ↔ "Vincent"
    """
    norm_a = _normalize_name(name_a)
    norm_b = _normalize_name(name_b)

    # Exact match after normalization
    if norm_a == norm_b:
        return True

    parts_a = _name_parts(name_a)
    parts_b = _name_parts(name_b)

    if not parts_a or not parts_b:
        return False

    # One name is a subset of the other (e.g. "Vinit" ⊂ "Vinit Raina")
    if parts_a.issubset(parts_b) or parts_b.issubset(parts_a):
        return True

    # Significant overlap (at least 1 part matches, and the other part
    # could be an initial — e.g. "V. Raina" shares "raina" with "Vinit Raina")
    overlap = parts_a & parts_b
    if overlap:
        # Check if non-overlapping parts are initials
        diff_a = parts_a - overlap
        diff_b = parts_b - overlap
        a_initials = all(len(p) <= 2 for p in diff_a)
        b_initials = all(len(p) <= 2 for p in diff_b)

        # If one side's non-overlapping parts are all initials, it's a match
        if (not diff_a or a_initials) or (not diff_b or b_initials):
            return True

        # Check if initials match first letters
        for part_short in (diff_a if a_initials else diff_b):
            for part_long in (diff_b if a_initials else diff_a):
                if part_long.startswith(part_short[0]):
                    return True

    return False


# =============================================================================
# Entity Clustering
# =============================================================================

def cluster_entities(findings: List[dict]) -> List[EntityCluster]:
    """
    Group findings that refer to the same real-world entity into clusters.
    """
    clusters: List[EntityCluster] = []

    for finding in findings:
        entity_type = finding.get("type", "")
        value = finding.get("value", "").strip()
        score = finding.get("score", finding.get("confidence", 0.5))
        start = finding.get("start", 0)

        if not value or entity_type not in ("NAME", "ORG", "LOCATION"):
            continue

        # Try to match to an existing cluster
        matched_cluster = None
        for cluster in clusters:
            if cluster.entity_type != entity_type:
                continue

            if entity_type == "NAME":
                if are_name_variants(value, cluster.canonical_name):
                    matched_cluster = cluster
                    break
                # Also check against all variants
                for variant in cluster.variants:
                    if are_name_variants(value, variant):
                        matched_cluster = cluster
                        break
                if matched_cluster:
                    break
            else:
                # For ORG/LOCATION: simpler — normalize and compare
                if _normalize_name(value) == _normalize_name(cluster.canonical_name):
                    matched_cluster = cluster
                    break

        if matched_cluster:
            matched_cluster.variants.add(value)
            matched_cluster.positions.append(start)
            if score > matched_cluster.max_confidence:
                matched_cluster.max_confidence = score
                # Update canonical to the highest-confidence form
                matched_cluster.canonical_name = value
        else:
            clusters.append(EntityCluster(
                canonical_name=value,
                entity_type=entity_type,
                variants={value},
                positions=[start],
                max_confidence=score,
            ))

    return clusters


# =============================================================================
# Entity Propagation
# =============================================================================

def propagate_entities(
    text: str,
    existing_findings: List[dict],
    clusters: List[EntityCluster],
    min_propagation_confidence: float = 0.80,
) -> List[dict]:
    """
    Second pass: for each high-confidence entity cluster, find ALL occurrences
    of any variant in the text that aren't already in the findings.

    This catches names that BERT missed on subsequent mentions — e.g. if
    "Vinit Raina" was detected with 0.95 on line 1 but missed on line 30.
    """
    # Build a set of already-covered (start, type) pairs
    covered: Set[Tuple[int, str]] = set()
    for f in existing_findings:
        covered.add((f["start"], f["type"]))

    new_findings: List[dict] = []

    for cluster in clusters:
        if cluster.max_confidence < min_propagation_confidence:
            continue

        # Search for all variants in the text
        for variant in cluster.variants:
            if len(variant) < 3:
                continue

            pattern = re.compile(re.escape(variant), re.IGNORECASE)
            for match in pattern.finditer(text):
                start = match.start()
                end = match.end()

                key = (start, cluster.entity_type)
                if key in covered:
                    continue

                # Also check ±3 chars to avoid near-duplicates
                near_covered = any(
                    (start + offset, cluster.entity_type) in covered
                    for offset in range(-3, 4)
                )
                if near_covered:
                    continue

                # Verify the matched text has correct casing (not inside a word)
                matched_text = text[start:end]
                if matched_text[0].islower() and cluster.entity_type in ("NAME", "ORG"):
                    continue

                new_findings.append({
                    "type": cluster.entity_type,
                    "value": matched_text,
                    "start": start,
                    "end": end,
                    "score": round(cluster.max_confidence * 0.90, 4),
                    "source": "PROPAGATION",
                    "propagated_from": cluster.canonical_name,
                })
                covered.add(key)

    return new_findings


# =============================================================================
# Cross-Entity Inference
# =============================================================================

def cross_entity_boost(
    findings: List[dict],
    email_names: List[dict],
) -> List[dict]:
    """
    Boost confidence of NER findings that are corroborated by email addresses.

    If regex found vinit.raina@gmail.com and NER found "Vinit Raina" nearby,
    boost the NER finding's score.

    If NER did NOT find the name but the email implies it, inject a synthetic
    finding with moderate confidence.
    """
    boosted = list(findings)
    existing_name_values = {
        _normalize_name(f["value"])
        for f in findings
        if f.get("type") in ("NAME", "PER")
    }

    for email_name in email_names:
        full = email_name.get("value", "")
        first = email_name.get("first_name", "")
        norm_full = _normalize_name(full)

        # Check if NER already found this name
        found_match = False
        for finding in boosted:
            if finding.get("type") not in ("NAME", "PER"):
                continue

            if are_name_variants(finding["value"], full):
                # Boost existing finding
                original_score = finding.get("score", 0.5)
                finding["score"] = min(0.9999, original_score + 0.10)
                finding["email_corroborated"] = True
                found_match = True

        # If name from email was NOT found by NER, check if it appears in the text
        # (synthetic injection is handled in the main pipeline by searching the text)
        if not found_match and norm_full not in existing_name_values:
            # Mark for injection by the orchestrator
            email_name["needs_injection"] = True

    return boosted


# =============================================================================
# Entity Consistency Check
# =============================================================================

def ensure_consistency(
    text: str,
    findings: List[dict],
    clusters: List[EntityCluster],
) -> List[dict]:
    """
    Ensure that entities tagged on one page/section are consistently tagged
    throughout the document.  Uses propagation for entities that appear
    multiple times but were only detected in some occurrences.
    """
    propagated = propagate_entities(text, findings, clusters, min_propagation_confidence=0.75)
    return findings + propagated
