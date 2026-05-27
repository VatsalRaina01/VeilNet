"""
Pydantic schemas for VeilNet API request/response models.

Typed models ensure consistent API contracts and auto-generate
OpenAPI documentation at /docs.
"""

from __future__ import annotations
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Individual Finding
# =============================================================================

class Finding(BaseModel):
    """A single detected PII entity."""
    page: int = Field(..., description="1-indexed page number")
    type: str = Field(..., description="PII type (NAME, EMAIL, PHONE, SSN, etc.)")
    value: str = Field(..., description="The detected text")
    source: str = Field(
        default="NER",
        description="Detection source: REGEX, NER, PROPAGATION, SYNTHETIC"
    )
    coords: List[float] = Field(
        ...,
        description="Bounding box [x0, y0, x1, y1] on the page",
        min_length=4,
        max_length=4,
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score 0.0–1.0 (None for regex matches)",
        ge=0.0,
        le=1.0,
    )
    risk_level: str = Field(
        default="medium",
        description="Risk level: critical, high, medium, low",
    )
    section: Optional[str] = Field(
        default=None,
        description="Document section where this finding was detected",
    )
    linked_entities: Optional[List[str]] = Field(
        default=None,
        description="Other entities linked to this finding (e.g. email ↔ name)",
    )


# =============================================================================
# Entity Link
# =============================================================================

class EntityLink(BaseModel):
    """A relationship between two detected entities."""
    entity_a: str = Field(..., description="First entity value")
    entity_a_type: str = Field(..., description="First entity type")
    entity_b: str = Field(..., description="Second entity value")
    entity_b_type: str = Field(..., description="Second entity type")
    relationship: str = Field(
        ...,
        description="Relationship type: variant_of, contact_of, belongs_to"
    )
    confidence: float = Field(
        ...,
        description="Confidence of the relationship",
        ge=0.0,
        le=1.0,
    )


# =============================================================================
# Risk Summary
# =============================================================================

class RiskSummary(BaseModel):
    """Aggregated risk metrics for the document."""
    total: int = Field(..., description="Total number of findings")
    critical: int = Field(default=0, description="Count of critical-risk findings")
    high: int = Field(default=0, description="Count of high-risk findings")
    medium: int = Field(default=0, description="Count of medium-risk findings")
    low: int = Field(default=0, description="Count of low-risk findings")
    overall_risk: str = Field(
        default="LOW",
        description="Overall document risk: CRITICAL, HIGH, ELEVATED, MODERATE, LOW"
    )
    by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Finding counts grouped by PII type"
    )


# =============================================================================
# Document Profile (subset exposed to frontend)
# =============================================================================

class DocumentInfo(BaseModel):
    """Document classification metadata."""
    doc_type: str = Field(..., description="Detected document type")
    doc_type_confidence: float = Field(..., description="Classification confidence")
    sections_detected: int = Field(default=0, description="Number of sections found")


# =============================================================================
# Full Analysis Response
# =============================================================================

class AnalysisResponse(BaseModel):
    """Complete analysis response from /analyze-pdf."""
    filename: str
    findings_count: int
    findings: List[Finding]
    document_info: DocumentInfo
    risk_summary: RiskSummary
    entity_links: List[EntityLink] = Field(default_factory=list)
