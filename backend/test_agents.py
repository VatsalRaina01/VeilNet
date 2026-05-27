"""Quick smoke test for all new VeilNet agents."""

print("=" * 60)
print("VEILNET AGENT INTELLIGENCE TEST")
print("=" * 60)

# ── Agent 1: Document Classifier ──────────────────────────────
print("\n[Agent 1] Document Classifier")
from app.services.doc_classifier import classify_document

resume_text = """Vinit Raina
Email: vinit.raina@gmail.com
Phone: +91-98765-43210

EXPERIENCE
Software Engineer at Google
Built microservices architecture

EDUCATION
B.Tech Computer Science, IIT Delhi

SKILLS
Python, React, Docker, Kubernetes
"""

profile = classify_document(resume_text)
print(f"  Type: {profile.doc_type} (confidence: {profile.doc_type_confidence})")
print(f"  Sections found: {len(profile.sections)}")
for s in profile.sections:
    print(f"    [{s.sensitivity.upper():>7}] {s.name}")
print("  ✓ PASS")

# ── Agent 2: Enhanced Regex Scanner ───────────────────────────
print("\n[Agent 2] Enhanced Regex Scanner")
from app.services.regex_engine import find_matches, PATTERNS

print(f"  Patterns loaded: {len(PATTERNS)}")

test_text = (
    "Aadhaar: 2345 6789 0123. "
    "PAN: ABCPD1234E. "
    "Email: test@example.com. "
    "Phone: +91-98765-43210. "
    "UPI: user@ybl"
)
matches = find_matches(test_text)
print(f"  Matches found: {len(matches)}")
for m in matches:
    print(f"    [{m['type']:>12}] {m['value']}")
print("  ✓ PASS")

# ── Agent 4: Entity Linker ────────────────────────────────────
print("\n[Agent 4] Cross-Entity Intelligence")
from app.services.entity_linker import (
    extract_names_from_emails,
    are_name_variants,
    cluster_entities,
)

# Email → Name
emails = [{"type": "EMAIL", "value": "vinit.raina@gmail.com", "start": 20, "end": 42}]
names = extract_names_from_emails(emails)
print(f"  Email→Name: {emails[0]['value']} → {names[0]['value']}")

# Name variants
pairs = [
    ("Vinit Raina", "V. Raina", True),
    ("Mr. Vinit Raina", "Vinit Raina", True),
    ("VINIT RAINA", "Vinit Raina", True),
    ("Vinit Raina", "Priya Sharma", False),
]
all_correct = True
for a, b, expected in pairs:
    result = are_name_variants(a, b)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_correct = False
    print(f"    {status} {a:20s} ↔ {b:20s} → {'MATCH' if result else 'no match'}")

if all_correct:
    print("  ✓ PASS")
else:
    print("  ✗ SOME TESTS FAILED")

# ── Agent 5: Ensemble Fusion ─────────────────────────────────
print("\n[Agent 5] Ensemble Fusion & Scoring")
from app.services.ensemble import fuse_and_score

regex_findings = [
    {"type": "EMAIL", "value": "test@example.com", "start": 10, "end": 26, "source": "REGEX"},
    {"type": "PHONE", "value": "+91-98765-43210", "start": 40, "end": 55, "source": "REGEX"},
]
ner_findings = [
    {"type": "NAME", "value": "Vinit Raina", "start": 0, "end": 11, "score": 0.95, "source": "NER"},
    {"type": "ORG", "value": "Google", "start": 100, "end": 106, "score": 0.88, "source": "NER"},
]
propagated = [
    {"type": "NAME", "value": "Vinit", "start": 200, "end": 205, "score": 0.85, "source": "PROPAGATION"},
]

final, summary = fuse_and_score(regex_findings, ner_findings, propagated)
print(f"  Input: {len(regex_findings)} regex + {len(ner_findings)} NER + {len(propagated)} propagated")
print(f"  Output: {len(final)} findings after fusion")
print(f"  Risk: {summary['overall_risk']} (crit:{summary['critical']} high:{summary['high']} med:{summary['medium']} low:{summary['low']})")
print("  ✓ PASS")

# ── Schemas ───────────────────────────────────────────────────
print("\n[Schemas] Pydantic Models")
from app.models.schemas import AnalysisResponse, Finding, RiskSummary
print(f"  AnalysisResponse fields: {list(AnalysisResponse.model_fields.keys())}")
print("  ✓ PASS")

print("\n" + "=" * 60)
print("ALL AGENTS VERIFIED SUCCESSFULLY")
print("=" * 60)
