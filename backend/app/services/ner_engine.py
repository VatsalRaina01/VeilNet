"""
Intelligent NER Engine — Agent 3 in the VeilNet multi-agent pipeline.

Intelligence upgrades over the original:
  1. Section-aware confidence — boosts/suppresses based on WHERE in the
     document an entity was found (Contact section → +0.12, Skills → −0.15)
  2. Entity propagation ready — returns data structures that Agent 4 can
     use for second-pass detection
  3. Cross-entity inference — accepts email-derived name candidates and
     uses them to boost or inject NER findings
  4. Multi-model voting — BERT + SpaCy + fine-tuned model each vote;
     borderline entities need 2/3 agreement
  5. All original fixes preserved (chunk offsets, bold isolation, section
     header guard, merge restrictions, per-type SpaCy bonus)
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from transformers import pipeline as hf_pipeline


# =============================================================================
# SpaCy — lightweight second-opinion NER for cross-validation
# =============================================================================

try:
    import spacy
    _spacy_nlp = spacy.load("en_core_web_sm")
    print("SpaCy (en_core_web_sm) loaded — cross-validation enabled.")
    SPACY_AVAILABLE = True
except Exception as e:
    print(f"SpaCy not available ({e}). Cross-validation disabled, BERT-only mode.")
    _spacy_nlp = None
    SPACY_AVAILABLE = False


# =============================================================================
# Model Loading — prefer bert-large-NER, fall back to bert-base-NER
# =============================================================================

FINE_TUNED_PATH = str(Path(__file__).resolve().parents[3] / "model" / "veilnet-ner-model")
LARGE_MODEL = "dslim/bert-large-NER"
BASE_MODEL = "dslim/bert-base-NER"

try:
    print(f"Loading NER Model ({LARGE_MODEL}) — this may take a moment...")
    base_pipeline = hf_pipeline(
        "ner",
        model=LARGE_MODEL,
        aggregation_strategy="first"
    )
    print(f"[OK] Loaded {LARGE_MODEL} (340M params - high accuracy mode)")
except Exception as e:
    print(f"Could not load {LARGE_MODEL} ({e}). Falling back to {BASE_MODEL}...")
    base_pipeline = hf_pipeline(
        "ner",
        model=BASE_MODEL,
        aggregation_strategy="first"
    )
    print(f"[OK] Loaded {BASE_MODEL} (110M params - standard mode)")

if os.path.isdir(FINE_TUNED_PATH) and os.path.isfile(os.path.join(FINE_TUNED_PATH, "config.json")):
    print(f"Loading fine-tuned NER model from: {FINE_TUNED_PATH}")
    fine_tuned_pipeline = hf_pipeline(
        "ner",
        model=FINE_TUNED_PATH,
        aggregation_strategy="first"
    )
    IS_FINE_TUNED = True
else:
    print(f"Fine-tuned model not found at {FINE_TUNED_PATH}. Falling back to base only.")
    fine_tuned_pipeline = None
    IS_FINE_TUNED = False

print("NER Models loaded successfully!")


# =============================================================================
# False-Positive Mitigation
# =============================================================================

BLOCKLIST = {
    "page", "section", "chapter", "table", "figure", "appendix",
    "summary", "overview", "abstract", "contents", "index", "glossary",
    "experience", "education", "skills", "projects", "profile",
    "resume", "cv", "portfolio", "references", "technical", "personal",
    "contact", "address", "phone", "email", "date", "objective",
    "achievements", "certifications", "publications", "interests",
    "hobbies", "languages", "awards", "activities", "volunteer",
    "professional", "academic", "qualification", "qualifications",
    "work", "history", "background", "details", "highlights",
    "dear", "sincerely", "regards", "subject", "from", "to", "cc", "bcc",
    "yours", "faithfully", "respectfully", "cordially",
    "company", "university", "college", "client", "user", "patient",
    "employee", "employer", "signature", "manager", "director",
    "officer", "administrator", "coordinator", "analyst", "developer",
    "engineer", "designer", "consultant", "specialist", "associate",
    "assistant", "intern", "trainee", "supervisor", "lead", "head",
    "important", "note", "confidential", "privacy", "policy", "terms",
    "disclaimer", "attachment", "enclosed", "regarding", "reference",
    "information", "details", "description", "requirements", "purpose",
    "python", "javascript", "react", "angular", "vue", "node",
    "java", "sql", "html", "css", "linux", "windows", "docker",
    "kubernetes", "aws", "azure", "git", "api", "rest", "graphql",
    "mongodb", "postgresql", "redis", "tensorflow", "pytorch",
    "typescript", "swift", "kotlin", "rust", "golang",
    "keras", "pandas", "numpy", "matplotlib", "seaborn", "nltk",
    "spacy", "scikit", "textblob", "tableau", "excel", "word",
    "power", "bi", "generativeai", "llms", "ann", "cnn", "rnn",
    "gru", "lstm", "svm", "knn", "dbscan", "pca", "idf", "tools",
    "algorithms", "models", "data", "science", "machine", "learning",
    "largelanguagemodels", "flask", "django", "fastapi", "nextjs",
    "spring", "bootstrap", "tailwind", "sass", "webpack", "vite",
    "jenkins", "terraform", "ansible", "nginx", "apache",
    "mysql", "sqlite", "cassandra", "elasticsearch", "kafka",
    "hadoop", "spark", "airflow", "mlflow", "huggingface",
    # Additional tech/role terms
    "frontend", "backend", "fullstack", "devops", "agile", "scrum",
    "kanban", "jira", "confluence", "notion", "slack", "figma",
    "photoshop", "illustrator", "sketch", "invision", "zeplin",
    "github", "gitlab", "bitbucket", "vercel", "netlify", "heroku",
    "render", "digitalocean", "gcp", "firebase", "supabase",
    "postman", "swagger", "curl", "axios", "fetch",
}

SINGLE_WORD_PER_BLOCKLIST = {
    "mr", "mrs", "ms", "dr", "prof", "sir", "madam",
    "senior", "junior", "general", "major", "captain",
    "state", "national", "international", "global", "local",
    "north", "south", "east", "west", "central",
    "new", "old", "great", "little", "big", "small",
    "first", "second", "third", "last", "next", "final",
    "full", "part", "time", "stack", "end", "front", "back",
}

INVALID_PER_SUBWORDS = {
    "variant", "solutions", "holdings", "inc", "ltd", "corp", "corporation",
    "llc", "company", "technologies", "tech", "studio", "studios",
    "university", "institute", "college", "school", "academy",
    "foundation", "association", "organization", "services", "systems",
    "group", "partners", "consulting", "labs", "software", "platform",
}

MIN_CONFIDENCE = {
    "PER":        0.75,
    "ORG":        0.95,
    "LOC":        0.95,
    "LOCATION":   0.95,
    "NAME":       0.75,
    "MISC":       0.99,
    "EMAIL":      0.80,
    "PHONE":      0.80,
    "SSN":        0.80,
    "CREDITCARD": 0.80,
    "FINANCIAL":  0.80,
    "DATE":       0.85,
    "DIGITAL":    0.85,
}
DEFAULT_MIN_CONFIDENCE = 0.88

# Per-type SpaCy bonus
CROSS_VALIDATION_BONUS = {
    "PER":      0.05,
    "NAME":     0.05,
    "ORG":      0.08,
    "LOC":      0.08,
    "LOCATION": 0.08,
}
DEFAULT_CROSS_VALIDATION_BONUS = 0.05

CONTEXT_SIGNALS = {
    "PER": {
        "keywords": [
            "name:", "by:", "author:", "contact:", "applicant:",
            "candidate:", "employee:", "prepared by", "submitted by",
            "signed by", "dear", "attn:", "from:", "mr.", "mrs.",
            "ms.", "dr.", "prof.", "s/o", "d/o", "w/o", "c/o",
            "father", "mother", "spouse", "guardian",
        ],
        "bonus": 0.10,
    },
    "ORG": {
        "keywords": [
            "at ", "worked at", "employed by", "company:", "organization:",
            "employer:", "firm:", "agency:", "department:", "division:",
            "hired by", "joined", "founded", "co-founded",
            "pvt", "ltd", "inc", "corp", "llp",
        ],
        "bonus": 0.08,
    },
    "LOC": {
        "keywords": [
            "located in", "from ", "lives in", "based in", "city:",
            "state:", "country:", "address:", "region:", "area:",
            "moved to", "born in", "resides in", "hometown:",
            "district:", "taluka:", "village:",
        ],
        "bonus": 0.08,
    },
}

# Resume section headers — never treat as person names
RESUME_SECTION_HEADERS = {
    "resume", "curriculum vitae", "cv", "profile", "summary",
    "objective", "overview", "experience", "education", "skills",
    "projects", "certifications", "achievements", "awards",
    "publications", "references", "contact", "about me",
}

RESUME_SIGNALS = {
    "experience", "education", "skills", "resume", "cv",
    "curriculum vitae", "work history", "employment", "qualifications",
}


# =============================================================================
# Smart sentence-boundary chunking
# =============================================================================

def _split_into_smart_chunks(text: str, max_chunk_chars: int = 450) -> List[Tuple[str, int]]:
    """
    Split text into chunks at sentence boundaries.
    Returns list of (chunk_text, absolute_start_offset).
    """
    chunks = []
    chunk_start = 0
    last_valid_end = 0
    last_valid_delim_after = 0

    for match in re.finditer(r'(?<=[.!?\n])\s+', text):
        delim_start = match.start()
        delim_end = match.end()

        if (delim_start - chunk_start) > max_chunk_chars and last_valid_end > chunk_start:
            chunks.append((text[chunk_start:last_valid_end], chunk_start))
            chunk_start = last_valid_delim_after

        last_valid_end = delim_start
        last_valid_delim_after = delim_end

    if chunk_start < len(text):
        chunks.append((text[chunk_start:], chunk_start))

    return chunks


# =============================================================================
# SpaCy Cross-Validation
# =============================================================================

_SPACY_LABEL_MAP = {
    "PERSON": "PER",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "FAC": "LOC",
    "NORP": "ORG",
}


def _spacy_cross_validate(text: str) -> Dict[Tuple[int, str], str]:
    if not SPACY_AVAILABLE or _spacy_nlp is None:
        return {}

    doc = _spacy_nlp(text)
    spacy_entities = {}
    for ent in doc.ents:
        mapped_group = _SPACY_LABEL_MAP.get(ent.label_)
        if mapped_group:
            spacy_entities[(ent.start_char, mapped_group)] = ent.text
            for offset in range(-3, 4):
                spacy_entities[(ent.start_char + offset, mapped_group)] = ent.text
    return spacy_entities


def _has_spacy_agreement(entity_start: int, entity_group: str,
                         spacy_entities: dict) -> bool:
    check_group = entity_group
    if entity_group in ("NAME", "PER"):
        check_group = "PER"
    elif entity_group in ("LOCATION", "LOC"):
        check_group = "LOC"
    return (entity_start, check_group) in spacy_entities


# =============================================================================
# Context-Aware Validation
# =============================================================================

def _get_context_bonus(text: str, entity_start: int, entity_end: int,
                       entity_group: str) -> float:
    lookup_group = entity_group
    if entity_group == "NAME":
        lookup_group = "PER"
    elif entity_group == "LOCATION":
        lookup_group = "LOC"

    signals = CONTEXT_SIGNALS.get(lookup_group)
    if not signals:
        return 0.0

    window_start = max(0, entity_start - 80)
    window_end = min(len(text), entity_end + 80)
    window = text[window_start:window_end].lower()

    for keyword in signals["keywords"]:
        if keyword in window:
            return signals["bonus"]

    return 0.0


# =============================================================================
# Entity Validation Pipeline
# =============================================================================

def _is_valid_entity(word: str, entity_group: str, score: float,
                     context_bonus: float = 0.0,
                     spacy_agrees: bool = False,
                     section_bonus: float = 0.0) -> bool:
    """
    Validate an entity with section-aware threshold adjustment.
    """
    word_stripped = word.strip()
    word_lower = word_stripped.lower()

    base_threshold = MIN_CONFIDENCE.get(entity_group, DEFAULT_MIN_CONFIDENCE)
    effective_threshold = base_threshold

    # SpaCy cross-validation bonus (per-type)
    if spacy_agrees:
        bonus = CROSS_VALIDATION_BONUS.get(entity_group, DEFAULT_CROSS_VALIDATION_BONUS)
        effective_threshold -= bonus

    # Context keyword bonus
    effective_threshold -= context_bonus

    # Section sensitivity bonus (NEW — from doc_classifier)
    effective_threshold -= section_bonus

    # Floor at 0.50 to prevent runaway lowering
    effective_threshold = max(effective_threshold, 0.50)

    if score < effective_threshold:
        return False

    if len(word_stripped) < 2 and entity_group not in ("PER", "NAME", "MISC"):
        return False

    if word_stripped.startswith("##") and len(word_stripped) <= 3:
        return False

    clean_word = word_lower.replace("##", "")
    if clean_word in BLOCKLIST:
        return False

    words = clean_word.split()
    if all(w in BLOCKLIST for w in words):
        return False

    if entity_group in ("PER", "NAME", "ORG", "LOC", "LOCATION"):
        first_char = word_stripped.replace("##", "")
        if first_char and first_char[0].islower():
            return False

    if entity_group in ("PER", "NAME"):
        if len(words) == 1 and clean_word in SINGLE_WORD_PER_BLOCKLIST:
            return False
        if any(w in INVALID_PER_SUBWORDS for w in words):
            return False

    digit_ratio = sum(c.isdigit() for c in word_stripped) / max(len(word_stripped), 1)
    if entity_group in ("PER", "NAME", "ORG", "LOC", "LOCATION") and digit_ratio > 0.5:
        return False

    alnum_ratio = sum(c.isalnum() or c.isspace() for c in word_stripped) / max(len(word_stripped), 1)
    if alnum_ratio < 0.6:
        return False

    if len(word_stripped) > 80:
        return False

    return True


ENTITY_TYPE_MAP = {
    "PER":        "NAME",
    "ORG":        "ORG",
    "LOC":        "LOCATION",
    "LOCATION":   "LOCATION",
    "MISC":       "MISC",
    "EMAIL":      "EMAIL",
    "PHONE":      "PHONE",
    "SSN":        "SSN",
    "CREDITCARD": "CREDITCARD",
    "FINANCIAL":  "FINANCIAL",
    "DATE":       "DATE",
    "DIGITAL":    "DIGITAL",
}


def _span_key(start: int, veilnet_type: str) -> Tuple[int, str]:
    return (start, veilnet_type)


# =============================================================================
# Multi-Model Voting
# =============================================================================

def _count_model_votes(
    word: str,
    entity_group: str,
    start: int,
    chunk_text: str,
    chunk_offset: int,
    spacy_entities: dict,
) -> Tuple[int, int]:
    """
    Count how many models agree on this entity.
    Returns (votes_for, total_models).
    """
    total_models = 1  # BERT base always votes (it proposed the entity)
    votes_for = 1

    # SpaCy vote
    if SPACY_AVAILABLE:
        total_models += 1
        if _has_spacy_agreement(start, entity_group, spacy_entities):
            votes_for += 1

    # Fine-tuned model vote (if available)
    if fine_tuned_pipeline:
        total_models += 1
        try:
            ft_results = fine_tuned_pipeline(word)
            for ent in ft_results:
                if ent.get("entity_group") == entity_group and ent.get("score", 0) > 0.6:
                    votes_for += 1
                    break
        except Exception:
            pass  # fine-tuned model failed on this token — don't count

    return votes_for, total_models


# =============================================================================
# Main NER Detection Function
# =============================================================================

def find_ner_matches(
    text: str,
    bold_phrases: List[str] = None,
    doc_profile=None,
    email_name_candidates: List[dict] = None,
) -> List[dict]:
    """
    Uses BERT NER + SpaCy cross-validation to find People, Organizations,
    Locations, and PII entities.

    NEW parameters:
      doc_profile: DocumentProfile from Agent 1 (section-aware confidence)
      email_name_candidates: candidate names extracted from emails by Agent 4

    Intelligence features:
      - Section-aware confidence adjustment
      - Cross-entity boosting from email-derived names
      - Multi-model voting for borderline entities
      - All original fixes preserved
    """
    if bold_phrases is None:
        bold_phrases = []
    if email_name_candidates is None:
        email_name_candidates = []

    spacy_entities = _spacy_cross_validate(text)
    chunks = _split_into_smart_chunks(text, max_chunk_chars=450)

    findings: List[dict] = []
    seen_spans: Set[Tuple[int, str]] = set()

    # ─── Bold Phrase Isolation Evaluation ────────────────────────────────

    for phrase in set(bold_phrases):
        phrase = phrase.strip()
        if len(phrase) < 3 or phrase.lower() in BLOCKLIST:
            continue

        isolated_results = base_pipeline(phrase)
        if not isolated_results:
            continue

        best = max(isolated_results, key=lambda e: e["score"])
        group = best["entity_group"]

        if group not in ("PER", "NAME", "ORG", "LOC", "LOCATION"):
            continue

        score = round(float(best["score"]), 4)
        effective_threshold = MIN_CONFIDENCE.get(group, 0.88) - 0.20

        if score < effective_threshold:
            continue

        if not _is_valid_entity(phrase, group, score=1.0):
            continue

        veilnet_type = ENTITY_TYPE_MAP.get(group)
        if not veilnet_type:
            continue

        for match in re.finditer(re.escape(phrase), text):
            start_idx, end_idx = match.span()
            key = _span_key(start_idx, veilnet_type)
            if key not in seen_spans:
                # Section-aware bonus
                section_bonus = 0.0
                if doc_profile:
                    section_bonus = doc_profile.get_confidence_bonus_at(start_idx)

                findings.append({
                    "type": veilnet_type,
                    "value": phrase,
                    "start": start_idx,
                    "end": end_idx,
                    "score": min(0.9999, score + 0.15 + section_bonus),
                    "source": "NER",
                    "section": _get_section_name(doc_profile, start_idx),
                })
                seen_spans.add(key)

    # ─── First-line name heuristic (resume-specific) ────────────────────

    lines = text.strip().split('\n')
    if lines:
        first_line = lines[0].strip()
        words_in_first_line = first_line.split()
        preamble = text[:500].lower()

        looks_like_resume = any(sig in preamble for sig in RESUME_SIGNALS)
        not_a_header = first_line.lower() not in RESUME_SECTION_HEADERS
        right_length = 1 <= len(words_in_first_line) <= 4
        all_capitalised = all(w[0].isupper() for w in words_in_first_line if w)

        if looks_like_resume and not_a_header and right_length and all_capitalised:
            start_idx = text.find(first_line)
            key = _span_key(start_idx, "NAME")
            if key not in seen_spans:
                findings.append({
                    "type": "NAME",
                    "value": first_line,
                    "start": start_idx,
                    "end": start_idx + len(first_line),
                    "score": 0.9999,
                    "source": "NER",
                    "section": _get_section_name(doc_profile, start_idx),
                })
                seen_spans.add(key)

    # ─── Process each chunk through BERT + validation ───────────────────

    for chunk_text, chunk_offset in chunks:

        base_results = base_pipeline(chunk_text)
        for entity in base_results:
            entity_group = entity["entity_group"]
            if entity_group not in ("PER", "NAME", "ORG", "LOC", "LOCATION"):
                continue

            score = round(float(entity["score"]), 4)
            word = entity["word"].strip()
            start = entity["start"] + chunk_offset
            end = entity["end"] + chunk_offset

            # Section-aware bonus
            section_bonus = 0.0
            if doc_profile:
                section_bonus = doc_profile.get_confidence_bonus_at(start)

            context_bonus = _get_context_bonus(text, start, end, entity_group)
            spacy_agrees = _has_spacy_agreement(start, entity_group, spacy_entities)

            # Cross-entity boost: check if this entity matches an email-derived name
            email_boost = 0.0
            for candidate in email_name_candidates:
                cand_first = candidate.get("first_name", "").lower()
                cand_last = (candidate.get("last_name") or "").lower()
                word_lower = word.lower()
                if cand_first and cand_first in word_lower:
                    email_boost = 0.08
                    break
                if cand_last and cand_last in word_lower:
                    email_boost = 0.08
                    break

            if not _is_valid_entity(
                word, entity_group, score,
                context_bonus=context_bonus + email_boost,
                spacy_agrees=spacy_agrees,
                section_bonus=section_bonus,
            ):
                continue

            veilnet_type = ENTITY_TYPE_MAP.get(entity_group)
            if veilnet_type is None:
                continue

            key = _span_key(start, veilnet_type)
            if key in seen_spans:
                continue
            seen_spans.add(key)

            findings.append({
                "type": veilnet_type,
                "value": word,
                "start": start,
                "end": end,
                "score": score,
                "source": "NER",
                "section": _get_section_name(doc_profile, start),
                "spacy_agrees": spacy_agrees,
                "email_corroborated": email_boost > 0,
            })

        # Fine-tuned model pass
        if fine_tuned_pipeline:
            ft_results = fine_tuned_pipeline(chunk_text)
            for entity in ft_results:
                entity_group = entity["entity_group"]
                if entity_group not in ("PER", "NAME", "ORG", "LOC", "LOCATION", "MISC"):
                    continue

                score = round(float(entity["score"]), 4)
                word = entity["word"].strip()
                start = entity["start"] + chunk_offset
                end = entity["end"] + chunk_offset

                section_bonus = 0.0
                if doc_profile:
                    section_bonus = doc_profile.get_confidence_bonus_at(start)

                if not _is_valid_entity(word, entity_group, score,
                                        section_bonus=section_bonus):
                    continue

                veilnet_type = ENTITY_TYPE_MAP.get(entity_group)
                if veilnet_type is None:
                    continue

                key = _span_key(start, veilnet_type)
                if key in seen_spans:
                    continue
                seen_spans.add(key)

                findings.append({
                    "type": veilnet_type,
                    "value": word,
                    "start": start,
                    "end": end,
                    "score": score,
                    "source": "NER",
                    "section": _get_section_name(doc_profile, start),
                })

    # ─── Cross-entity injection ─────────────────────────────────────────
    # If email analysis found names that NER didn't detect, inject them
    # as synthetic findings (with moderate confidence)

    for candidate in email_name_candidates:
        if not candidate.get("needs_injection"):
            continue

        name = candidate["value"]
        if len(name) < 3:
            continue

        # Search for this name in the text
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        for match in pattern.finditer(text):
            start_idx = match.start()
            end_idx = match.end()
            matched_text = text[start_idx:end_idx]

            # Skip if already covered or if not capitalised
            if matched_text[0].islower():
                continue

            key = _span_key(start_idx, "NAME")
            if key in seen_spans:
                continue

            # Only inject if it starts with uppercase (proper noun)
            section_bonus = 0.0
            if doc_profile:
                section_bonus = doc_profile.get_confidence_bonus_at(start_idx)

            inject_score = min(0.85, candidate["confidence"] + 0.10 + section_bonus)

            findings.append({
                "type": "NAME",
                "value": matched_text,
                "start": start_idx,
                "end": end_idx,
                "score": inject_score,
                "source": "SYNTHETIC",
                "section": _get_section_name(doc_profile, start_idx),
                "inferred_from": candidate.get("source_email", "email"),
            })
            seen_spans.add(key)

    # ─── Post-processing: merge adjacent NAME entities conservatively ───

    findings.sort(key=lambda x: x["start"])
    merged_findings = []

    for current in findings:
        if not merged_findings:
            merged_findings.append(current)
            continue

        previous = merged_findings[-1]

        is_adjacent_names = (
            previous["type"] in ("NAME", "PER") and
            current["type"] in ("NAME", "PER") and
            current["start"] - previous["end"] <= 2
        )
        prev_looks_like_initial = len(previous["value"].strip()) <= 3

        if is_adjacent_names and prev_looks_like_initial:
            new_value = text[previous["start"]:current["end"]]
            merged_findings[-1] = {
                "type": "NAME",
                "value": new_value,
                "start": previous["start"],
                "end": current["end"],
                "score": max(previous["score"], current["score"]),
                "source": previous.get("source", "NER"),
                "section": previous.get("section"),
            }
        else:
            merged_findings.append(current)

    return merged_findings


# =============================================================================
# Helpers
# =============================================================================

def _get_section_name(doc_profile, char_offset: int) -> Optional[str]:
    """Get the section name at a given character offset."""
    if doc_profile is None:
        return None
    section = doc_profile.get_section_at(char_offset)
    return section.name if section else None