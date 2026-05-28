"""
VeilNet API — Multi-Agent Intelligence Pipeline

Pipeline:
  1. Extract text + bold phrases from PDF (PyMuPDF)
  2. Agent 1: Document Classifier → doc_type + section map
  3. Agent 2: Enhanced Regex Scanner → regex_findings
  4. Agent 3: Intelligent NER → ner_findings (section-aware, email-boosted)
  5. Agent 4: Cross-Entity Intelligence → entity propagation + linking
  6. Agent 5: Ensemble Fusion → dedup, conflict resolution, risk scoring
  7. Return enriched analysis with risk summary + entity graph
"""

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF

# Agent imports
from app.services.doc_classifier import classify_document
from app.services.regex_engine import find_matches
from app.services.ner_engine import find_ner_matches
from app.services.entity_linker import (
    extract_names_from_emails,
    cluster_entities,
    cross_entity_boost,
    ensure_consistency,
)
from app.services.ensemble import fuse_and_score  
from app.routers.redact import router as redact_router

BOLD_FLAG = 16  # PyMuPDF font flag bit 4 = bold

app = FastAPI(
    title="VeilNet API",
    description="Multi-Agent PII Detection & Redaction Pipeline",
    version="2.0.0",
)

app.include_router(redact_router)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "VeilNet API v2.0 — Multi-Agent Intelligence Pipeline active.",
        "agents": [
            "Document Classifier",
            "Enhanced Regex Scanner",
            "Intelligent NER Engine",
            "Cross-Entity Linker",
            "Ensemble Fusion & Scoring",
        ],
        "docs": "/docs",
    }


# =============================================================================
# Helper: Extract bold phrases from a PDF page
# =============================================================================

def _extract_bold_phrases(page) -> list:
    """Extract bold text spans from a PDF page using PyMuPDF dict output."""
    bold_phrases = []
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        if "lines" in block:
            for line in block["lines"]:
                for span in line.get("spans", []):
                    is_bold = (span["flags"] & BOLD_FLAG) or ("bold" in span["font"].lower())
                    if is_bold:
                        text_val = span["text"].strip()
                        if len(text_val) > 1:
                            bold_phrases.append(text_val)
    return bold_phrases


# =============================================================================
# Main Analysis Endpoint
# =============================================================================

@app.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    """
    Multi-agent PII analysis pipeline.

    Accepts a PDF file and runs it through 5 intelligence agents:
      1. Document Classification (type + sections)
      2. Regex Scanning (20+ patterns, India-specific)
      3. NER Deep Scan (section-aware, email-boosted)
      4. Cross-Entity Intelligence (propagation, linking)
      5. Ensemble Fusion (scoring, dedup, risk levels)

    Returns enriched findings with risk summary and entity relationships.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    file_content = await file.read()
    doc = fitz.open(stream=file_content, filetype="pdf")

    # ── Collect text and structure across all pages ──────────────────────

    full_text_parts = []
    page_texts = []         # per-page text for coordinate mapping
    all_bold_phrases = []

    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        full_text_parts.append(page_text)
        page_texts.append(page_text)
        all_bold_phrases.extend(_extract_bold_phrases(page))

    full_text = "\n".join(full_text_parts)

    # ── Agent 1: Document Classification ────────────────────────────────

    doc_profile = classify_document(full_text, bold_phrases=all_bold_phrases)

    # ── Agent 2: Enhanced Regex Scanner (per-page) ──────────────────────

    all_regex_findings = []
    for page_num, page_text in enumerate(page_texts):
        regex_matches = find_matches(page_text)
        for match in regex_matches:
            match["page"] = page_num + 1
        all_regex_findings.extend(regex_matches)

    # ── Agent 4 (pre-pass): Extract name candidates from emails ─────────

    email_name_candidates = extract_names_from_emails(all_regex_findings)

    # ── Agent 3: Intelligent NER (per-page, section-aware) ──────────────

    all_ner_findings = []
    for page_num, page_text in enumerate(page_texts):
        page_bold = []
        # Re-extract bold for this specific page for accurate matching
        if page_num < len(doc):
            page_bold = _extract_bold_phrases(doc[page_num])

        ner_matches = find_ner_matches(
            page_text,
            bold_phrases=page_bold,
            doc_profile=doc_profile,
            email_name_candidates=email_name_candidates,
        )
        for match in ner_matches:
            match["page"] = page_num + 1
        all_ner_findings.extend(ner_matches)

    # ── Agent 4 (main pass): Cross-Entity Intelligence ──────────────────

    # Cluster entities across the entire document
    all_raw_findings = all_regex_findings + all_ner_findings
    clusters = cluster_entities(all_raw_findings)

    # Cross-entity boost: corroborate NER findings with email-derived names
    all_ner_findings = cross_entity_boost(all_ner_findings, email_name_candidates)

    # Entity propagation: ensure consistent tagging across all pages
    propagated_findings = []
    for page_num, page_text in enumerate(page_texts):
        page_findings = [f for f in all_raw_findings if f.get("page") == page_num + 1]
        from app.services.entity_linker import propagate_entities
        new_findings = propagate_entities(page_text, page_findings, clusters)
        for f in new_findings:
            f["page"] = page_num + 1
        propagated_findings.extend(new_findings)

    # ── Agent 5: Ensemble Fusion & Scoring ──────────────────────────────

    final_findings, risk_summary = fuse_and_score(
        regex_findings=all_regex_findings,
        ner_findings=all_ner_findings,
        propagated_findings=propagated_findings,
    )

    # ── Map findings to PDF coordinates ─────────────────────────────────

    output_findings = []
    seen_coords = set()

    for finding in final_findings:
        page_num = finding.get("page", 1) - 1  # 0-indexed for fitz
        matched_value = finding.get("value", "").strip()

        if not matched_value or page_num < 0 or page_num >= len(doc):
            continue

        page = doc[page_num]
        rects = page.search_for(matched_value, quads=False)

        for rect in rects:
            coord_key = (
                page_num,
                finding.get("type", ""),
                round(rect.x0, 1),
                round(rect.y0, 1),
                round(rect.x1, 1),
                round(rect.y1, 1),
            )
            if coord_key in seen_coords:
                continue
            seen_coords.add(coord_key)

            output_finding = {
                "page": page_num + 1,
                "type": finding.get("type", ""),
                "value": matched_value,
                "source": finding.get("source", "NER"),
                "coords": [rect.x0, rect.y0, rect.x1, rect.y1],
                "risk_level": finding.get("risk_level", "medium"),
            }

            # Include confidence for NER/PROPAGATION/SYNTHETIC detections
            if "score" in finding:
                output_finding["confidence"] = round(finding["score"], 4)

            # Include section info if available
            if finding.get("section"):
                output_finding["section"] = finding["section"]

            # Include linked entities
            if finding.get("propagated_from"):
                output_finding["linked_entities"] = [finding["propagated_from"]]
            if finding.get("inferred_from"):
                output_finding["linked_entities"] = [finding["inferred_from"]]

            output_findings.append(output_finding)

    doc.close()

    # ── Build entity links for response ─────────────────────────────────

    entity_links = []
    for candidate in email_name_candidates:
        if candidate.get("value") and candidate.get("source_email"):
            entity_links.append({
                "entity_a": candidate["value"],
                "entity_a_type": "NAME",
                "entity_b": candidate["source_email"],
                "entity_b_type": "EMAIL",
                "relationship": "contact_of",
                "confidence": candidate.get("confidence", 0.5),
            })

    # Build variant links from clusters
    for cluster in clusters:
        variants_list = list(cluster.variants)
        if len(variants_list) > 1:
            canonical = cluster.canonical_name
            for variant in variants_list:
                if variant != canonical:
                    entity_links.append({
                        "entity_a": canonical,
                        "entity_a_type": cluster.entity_type,
                        "entity_b": variant,
                        "entity_b_type": cluster.entity_type,
                        "relationship": "variant_of",
                        "confidence": round(cluster.max_confidence, 4),
                    })

    # ── Return enriched response ────────────────────────────────────────
    # Backward-compatible: findings_count + findings still at top level
    # NEW fields: document_info, risk_summary, entity_links

    return {
        "filename": file.filename,
        "findings_count": len(output_findings),
        "findings": output_findings,
        "full_text": full_text,
        "page_texts": page_texts,
        "document_info": {
            "doc_type": doc_profile.doc_type,
            "doc_type_confidence": doc_profile.doc_type_confidence,
            "sections_detected": len(doc_profile.sections),
            "sections": [
                {
                    "name": s.name,
                    "sensitivity": s.sensitivity,
                    "start": s.start,
                    "end": s.end,
                }
                for s in doc_profile.sections
            ],
        },
        "risk_summary": risk_summary,
        "entity_links": entity_links,
    }
