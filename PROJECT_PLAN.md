# PDF Redactor — Full Project Plan

## Project Overview

A web application where users upload a PDF, select redaction filters (email, phone, SSN, names, orgs, etc.), and receive a truly redacted PDF with underlying text removed — not just blacked out visually.

**Two redaction engines:**
1. **Regex Engine** — deterministic pattern matching (emails, phones, SSNs, credit cards, keywords)
2. **NER Engine** — fine-tuned transformer model detecting names, organizations, locations, sensitive terms

**Cost: $0** — every tool in this stack is free/open-source. Model training uses free Colab/Kaggle GPUs.

---

## Tech Stack (Final)

| Layer | Technology | Why |
|-------|-----------|-----|
| **Frontend** | React 18 + Vite + Tailwind CSS | Fast dev, great ecosystem, free |
| **PDF Preview** | react-pdf (pdf.js wrapper) | Render PDF pages in browser |
| **Backend** | FastAPI (Python 3.11+) | Async, fast, auto-docs, best ML ecosystem |
| **PDF Processing** | PyMuPDF (fitz) | Best layout control + native redaction API |
| **Regex Detection** | Python `re` module | Built-in, no dependencies |
| **NER Detection** | Fine-tuned BERT via HuggingFace Transformers | State-of-the-art NER, free to train on Colab |
| **NER Base Model** | `dslim/bert-base-NER` (pre-trained NER BERT) | Already trained on CoNLL-2003, great starting point |
| **Training Data** | CoNLL-2003 + custom PII-annotated data | Free, standard NER benchmark |
| **Training Platform** | Google Colab (free tier) / Kaggle Notebooks | Free GPU (T4) |
| **File Storage** | Local filesystem (temp dir, auto-cleanup) | Free, simple |
| **Deployment** | Render (backend) + Vercel (frontend) | Both have generous free tiers |
| **Version Control** | Git + GitHub | Free |

---

## Phase-by-Phase Execution Plan

---

### PHASE 1: Project Setup & Foundation (Days 1–2)

**Goal:** Repo structure, dev environment, basic frontend shell, basic backend shell.

#### 1.1 — Repository Setup
```
pdf-redactor/
├── frontend/              # React app
│   ├── src/
│   │   ├── components/
│   │   │   ├── UploadZone.jsx
│   │   │   ├── FilterPanel.jsx
│   │   │   ├── PdfPreview.jsx
│   │   │   ├── RedactionOverlay.jsx
│   │   │   └── DownloadButton.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI entry
│   │   ├── routers/
│   │   │   └── redact.py     # /upload, /redact endpoints
│   │   ├── services/
│   │   │   ├── pdf_service.py    # PyMuPDF text extraction + redaction
│   │   │   ├── regex_engine.py   # All regex patterns
│   │   │   └── ner_engine.py     # HuggingFace model inference
│   │   ├── models/
│   │   │   └── schemas.py        # Pydantic request/response models
│   │   └── utils/
│   │       └── cleanup.py        # Temp file cleanup
│   ├── requirements.txt
│   └── tests/
├── model/                  # Fine-tuning notebooks & scripts
│   ├── fine_tune_bert_ner.ipynb
│   ├── evaluate.py
│   └── data/
├── .gitignore
└── PROJECT_PLAN.md
```

#### 1.2 — Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn python-multipart pymupdf transformers torch
```

Minimal `main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PDF Redactor API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok"}
```

#### 1.3 — Frontend Setup
```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install
npm install tailwindcss @tailwindcss/vite react-pdf axios
```

#### Tasks Checklist — Phase 1
- [ ] Initialize git repo
- [ ] Create folder structure
- [ ] Set up FastAPI backend with health endpoint
- [ ] Set up React + Vite frontend with Tailwind
- [ ] Verify frontend can call backend (CORS working)
- [ ] Create .gitignore (exclude venv, node_modules, uploaded PDFs, model weights)

---

### PHASE 2: PDF Upload & Text Extraction (Days 3–5)

**Goal:** User uploads PDF → backend extracts text with coordinates → returns extracted text to frontend.

#### 2.1 — Backend: File Upload Endpoint
```python
# routers/redact.py
from fastapi import APIRouter, UploadFile, File
import tempfile, os

router = APIRouter()

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Save to temp file
    # Extract text with coordinates using PyMuPDF
    # Return text blocks with page numbers and bounding boxes
```

#### 2.2 — PDF Text Extraction with PyMuPDF
```python
# services/pdf_service.py
import fitz  # PyMuPDF

def extract_text_with_coords(pdf_path: str):
    doc = fitz.open(pdf_path)
    results = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        results.append({
                            "text": span["text"],
                            "bbox": span["bbox"],  # (x0, y0, x1, y1)
                            "page": page_num
                        })
    return results
```

**Why coordinates matter:** To redact, you need to know WHERE on the page each word is. PyMuPDF gives exact bounding boxes.

#### 2.3 — Frontend: Upload Component
- Drag-and-drop zone (or file picker)
- Show upload progress
- On success, display extracted text (for debugging/preview)

#### Tasks Checklist — Phase 2
- [ ] Build `/upload` endpoint accepting PDF files
- [ ] Implement `extract_text_with_coords()` using PyMuPDF
- [ ] Return structured JSON of all text spans with coordinates
- [ ] Build `UploadZone.jsx` component
- [ ] Test with 3–4 sample PDFs of varying complexity
- [ ] Add file size limit (10MB) and type validation

---

### PHASE 3: Regex-Based Redaction Engine (Days 6–10)

**Goal:** Detect sensitive data using regex → redact matching regions → return clean PDF.

#### 3.1 — Regex Patterns
```python
# services/regex_engine.py
import re

PATTERNS = {
    "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    "phone_us": r'(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
    "phone_in": r'(\+91[-.\s]?)?[6-9]\d{9}',
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
    "credit_card": r'\b(?:\d[ -]*?){13,19}\b',
    "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    "aadhaar": r'\b\d{4}\s?\d{4}\s?\d{4}\b',
    "pan": r'[A-Z]{5}\d{4}[A-Z]',
}

def detect_patterns(text: str, selected_filters: list[str]) -> list[dict]:
    """Returns list of {'type': ..., 'match': ..., 'start': ..., 'end': ...}"""
    findings = []
    for filter_name in selected_filters:
        pattern = PATTERNS.get(filter_name)
        if pattern:
            for match in re.finditer(pattern, text):
                findings.append({
                    "type": filter_name,
                    "match": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                })
    return findings
```

#### 3.2 — Apply TRUE Redaction with PyMuPDF
```python
# services/pdf_service.py
def apply_redactions(pdf_path: str, redaction_targets: list[dict]) -> str:
    doc = fitz.open(pdf_path)
    for target in redaction_targets:
        page = doc[target["page"]]
        # Search for the text to get exact rectangles
        rects = page.search_for(target["text"])
        for rect in rects:
            page.add_redact_annot(rect, fill=(0, 0, 0))  # black box

    # CRITICAL: This actually removes the underlying text
    for page in doc:
        page.apply_redactions()

    output_path = pdf_path.replace(".pdf", "_redacted.pdf")
    doc.save(output_path)
    doc.close()
    return output_path
```

**IMPORTANT:** `page.apply_redactions()` is what makes this TRUE redaction. It removes the text from the PDF structure, not just draws over it.

#### 3.3 — Redaction API Endpoint
```python
@router.post("/redact")
async def redact_pdf(file: UploadFile, filters: list[str]):
    # 1. Save uploaded file
    # 2. Extract text
    # 3. Run regex detection on selected filters
    # 4. Map detected text back to page coordinates
    # 5. Apply redactions
    # 6. Return redacted PDF as download
```

#### 3.4 — Frontend: Filter Selection Panel
- Checkboxes for each filter type (email, phone, SSN, etc.)
- Custom keyword input field
- "Redact" button
- Download link for result

#### Tasks Checklist — Phase 3
- [ ] Implement all regex patterns
- [ ] Write unit tests for each pattern (at least 5 test cases per pattern)
- [ ] Implement `apply_redactions()` using PyMuPDF redact API
- [ ] Verify redacted PDFs have text truly removed (copy-paste test)
- [ ] Build `/redact` endpoint
- [ ] Build `FilterPanel.jsx` with checkboxes
- [ ] Build `DownloadButton.jsx`
- [ ] End-to-end test: upload → select filters → download redacted PDF
- [ ] Add custom keyword filter support

---

### PHASE 4: PDF Preview & Redaction Highlighting (Days 11–14)

**Goal:** Show PDF in browser, highlight detected items before redaction, let user approve/reject.

#### 4.1 — PDF Preview
Using `react-pdf`:
```jsx
import { Document, Page } from 'react-pdf';

function PdfPreview({ fileUrl, highlights }) {
  return (
    <Document file={fileUrl}>
      <Page pageNumber={1}>
        {/* Overlay highlight boxes on detected items */}
      </Page>
    </Document>
  );
}
```

#### 4.2 — Detection Preview Endpoint
New endpoint: `/detect` (separate from `/redact`)
- Takes PDF + filters
- Returns detected items with coordinates (but does NOT redact yet)
- Frontend shows highlights
- User can uncheck false positives
- Then user clicks "Confirm Redaction"

#### 4.3 — Highlight Overlay
- Red semi-transparent boxes over detected sensitive data
- Each box has a checkbox to include/exclude
- Sidebar list of all detections grouped by type

#### Tasks Checklist — Phase 4
- [ ] Integrate react-pdf for in-browser preview
- [ ] Build `/detect` endpoint (detection without redaction)
- [ ] Build `RedactionOverlay.jsx` component
- [ ] Implement include/exclude toggle per detection
- [ ] Wire up "Confirm Redaction" → `/redact` with final selections
- [ ] Test with multi-page PDFs

---

### PHASE 5: Fine-Tune BERT for NER (Days 15–25)

**Goal:** Train a custom NER model that detects PERSON, ORG, LOCATION, and custom PII entities.

#### 5.1 — Why `dslim/bert-base-NER`?
- Already pre-trained on CoNLL-2003 (English NER benchmark)
- Detects: PER, ORG, LOC, MISC
- We fine-tune further on PII-specific data
- Free to use, Apache 2.0 license

#### 5.2 — Training Data

**Free datasets:**

| Dataset | What It Contains | Where |
|---------|-----------------|-------|
| CoNLL-2003 | Standard NER (PER, ORG, LOC, MISC) | HuggingFace `conll2003` |
| WikiNER | Larger NER dataset from Wikipedia | HuggingFace `wiki_ner` |
| ai4privacy/pii-masking-300k | PII-specific (names, emails, phones in context) | HuggingFace Hub |
| Few-shot custom data | Your own annotated PDFs | Manual labeling |

**Primary choice:** `ai4privacy/pii-masking-300k` — it's specifically built for PII detection and is free on HuggingFace.

#### 5.3 — Fine-Tuning Notebook (Google Colab)

```python
# fine_tune_bert_ner.ipynb — Run on Google Colab (free GPU)

# 1. Install dependencies
# !pip install transformers datasets seqeval accelerate

# 2. Load dataset
from datasets import load_dataset
dataset = load_dataset("ai4privacy/pii-masking-300k")

# 3. Load pre-trained model
from transformers import AutoTokenizer, AutoModelForTokenClassification
model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=NUM_LABELS)

# 4. Tokenize and align labels
def tokenize_and_align(examples):
    tokenized = tokenizer(examples["tokens"], truncation=True,
                          is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = [-100 if wid is None else label[wid] for wid in word_ids]
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

# 5. Training
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./ner-model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,  # Use mixed precision on Colab GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # Using seqeval
)

trainer.train()

# 6. Save model
trainer.save_model("./pii-ner-model")
tokenizer.save_pretrained("./pii-ner-model")
# Download from Colab or push to HuggingFace Hub (free)
```

#### 5.4 — Model Evaluation
```python
# evaluate.py
from seqeval.metrics import classification_report
# Evaluate precision, recall, F1 per entity type
# Target: F1 > 0.85 for PERSON, ORG, LOCATION
```

#### 5.5 — Integrate into Backend
```python
# services/ner_engine.py
from transformers import pipeline

# Load your fine-tuned model
ner_pipeline = pipeline("ner", model="./pii-ner-model", aggregation_strategy="simple")

def detect_entities(text: str, entity_types: list[str]) -> list[dict]:
    results = ner_pipeline(text)
    filtered = [r for r in results if r["entity_group"] in entity_types]
    return [
        {"text": r["word"], "type": r["entity_group"], "score": r["score"],
         "start": r["start"], "end": r["end"]}
        for r in filtered
    ]
```

#### Tasks Checklist — Phase 5
- [ ] Set up Google Colab notebook
- [ ] Load and explore `ai4privacy/pii-masking-300k` dataset
- [ ] Understand BIO/IOB tagging scheme
- [ ] Map dataset labels to your target entity types
- [ ] Implement tokenization with label alignment
- [ ] Train model (3 epochs, ~1-2 hours on Colab free GPU)
- [ ] Evaluate with seqeval (precision, recall, F1)
- [ ] If F1 < 0.80, adjust hyperparameters and retrain
- [ ] Export model weights
- [ ] Integrate into `ner_engine.py`
- [ ] Test NER detection on sample PDF text
- [ ] Add NER filter options to frontend (PERSON, ORG, LOCATION, etc.)

---

### PHASE 6: Combined Engine & Polish (Days 26–32)

**Goal:** Merge regex + NER, deduplicate, add confidence scores, polish UI.

#### 6.1 — Combined Detection Pipeline
```python
# services/combined_engine.py
def detect_all(text, regex_filters, ner_filters):
    regex_results = regex_engine.detect_patterns(text, regex_filters)
    ner_results = ner_engine.detect_entities(text, ner_filters)

    # Merge and deduplicate (prefer NER if overlap)
    all_results = merge_detections(regex_results, ner_results)
    return all_results
```

#### 6.2 — Frontend Polish
- Loading states during processing
- Error handling (invalid PDF, too large, processing failure)
- Responsive design (mobile-friendly)
- Dark mode (Tailwind makes this trivial)
- Redaction summary page before download

#### 6.3 — Security Hardening
- File cleanup: delete uploaded + redacted PDFs after 10 minutes
- File size limit: 10MB max
- File type validation: check magic bytes, not just extension
- Rate limiting: 10 requests/minute per IP
- No persistent storage of user files

#### Tasks Checklist — Phase 6
- [ ] Build combined detection engine
- [ ] Implement deduplication logic for overlapping detections
- [ ] Add confidence scores to NER results in UI
- [ ] Add loading spinners and progress indicators
- [ ] Add error handling (frontend + backend)
- [ ] Implement auto-cleanup of temp files
- [ ] Add file validation (size, type, magic bytes)
- [ ] Add rate limiting
- [ ] Test with 10+ diverse PDFs
- [ ] Cross-browser testing (Chrome, Firefox, Safari)

---

### PHASE 7: Deployment (Days 33–35)

**Goal:** Deploy for free, accessible via URL.

#### 7.1 — Backend Deployment (Render — Free Tier)
```yaml
# render.yaml
services:
  - type: web
    name: pdf-redactor-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
    plan: free
```
**Note:** Free Render tier spins down after inactivity (cold start ~30s). Fine for learning project.

**Model hosting:** Include model weights in the repo (< 500MB) or download on startup from HuggingFace Hub.

#### 7.2 — Frontend Deployment (Vercel — Free Tier)
```bash
cd frontend
npm run build
# Connect GitHub repo to Vercel — auto-deploys on push
```

#### Tasks Checklist — Phase 7
- [ ] Create `requirements.txt` with pinned versions
- [ ] Add `Procfile` or `render.yaml` for Render
- [ ] Deploy backend to Render
- [ ] Deploy frontend to Vercel
- [ ] Update frontend API URL to production backend
- [ ] Test full flow on deployed version
- [ ] Write README.md with setup instructions

---

## Learning Resources (All Free)

### Web Development
| Resource | URL | Covers |
|----------|-----|--------|
| React Official Tutorial | https://react.dev/learn | React fundamentals |
| Tailwind CSS Docs | https://tailwindcss.com/docs | Styling |
| Vite Guide | https://vite.dev/guide/ | Build tooling |
| MDN Web Docs | https://developer.mozilla.org | HTML, CSS, JS reference |

### Backend & API
| Resource | URL | Covers |
|----------|-----|--------|
| FastAPI Official Tutorial | https://fastapi.tiangolo.com/tutorial/ | Full FastAPI guide |
| PyMuPDF Documentation | https://pymupdf.readthedocs.io | PDF manipulation |
| PyMuPDF Redaction Recipe | https://pymupdf.readthedocs.io/en/latest/recipes-text.html | Redaction specifically |

### NLP & Model Training
| Resource | URL | Covers |
|----------|-----|--------|
| HuggingFace NLP Course | https://huggingface.co/learn/nlp-course | Transformers, fine-tuning |
| HuggingFace Token Classification Guide | https://huggingface.co/docs/transformers/tasks/token_classification | NER fine-tuning specifically |
| spaCy Course | https://course.spacy.io | NER concepts, pipelines |
| Stanford CS224N (YouTube) | Search "CS224N" on YouTube | Deep NLP theory (optional) |

### Datasets
| Dataset | URL | Use |
|---------|-----|-----|
| ai4privacy/pii-masking-300k | https://huggingface.co/datasets/ai4privacy/pii-masking-300k | PII training data |
| CoNLL-2003 | `load_dataset("conll2003")` via HuggingFace | Standard NER benchmark |
| dslim/bert-base-NER | https://huggingface.co/dslim/bert-base-NER | Pre-trained base model |

### General
| Resource | URL | Covers |
|----------|-----|--------|
| Git & GitHub | https://docs.github.com/en/get-started | Version control |
| Google Colab | https://colab.research.google.com | Free GPU for training |
| Kaggle Notebooks | https://www.kaggle.com/code | Alternative free GPU |

---

## Cost Breakdown

| Item | Cost |
|------|------|
| React + Vite + Tailwind | Free (open source) |
| FastAPI + PyMuPDF | Free (open source) |
| HuggingFace Transformers | Free (open source) |
| Training GPU (Colab) | Free (T4 GPU, ~12GB VRAM) |
| Training Data | Free (HuggingFace datasets) |
| Hosting — Backend (Render) | Free tier |
| Hosting — Frontend (Vercel) | Free tier |
| Domain (optional) | Free (.render.com / .vercel.app subdomains) |
| **Total** | **$0** |

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1. Setup | Days 1–2 | Repo, dev env, hello-world frontend + backend |
| 2. PDF Processing | Days 3–5 | Upload PDF → extract text with coordinates |
| 3. Regex Engine | Days 6–10 | Pattern detection + true redaction working |
| 4. Preview UI | Days 11–14 | In-browser PDF preview with highlight overlay |
| 5. BERT Fine-tuning | Days 15–25 | Trained NER model integrated into backend |
| 6. Polish | Days 26–32 | Combined engine, security, error handling |
| 7. Deploy | Days 33–35 | Live on Render + Vercel |

**Total: ~5 weeks** (assuming part-time, learning as you go)

---

## Key Technical Decisions & Rationale

**Why BERT over spaCy for NER?**
- BERT has better contextual understanding (bidirectional attention)
- HuggingFace ecosystem makes fine-tuning straightforward
- `dslim/bert-base-NER` gives a strong starting point
- Better learning experience (you'll understand transformers deeply)
- spaCy is great but less educational for understanding model internals

**Why PyMuPDF over other PDF libraries?**
- Only library with native redaction API (`add_redact_annot` + `apply_redactions`)
- Gives exact text coordinates (bounding boxes)
- Fast (C-based core)
- Other libraries (PyPDF2, pdfplumber) can extract text but can't do true redaction

**Why FastAPI over Flask?**
- Async by default (better for file processing)
- Auto-generates API docs (Swagger UI at `/docs`)
- Built-in request validation with Pydantic
- Type hints everywhere (better learning habit)
- As fast as Node.js, much faster than Flask

**Why React + Vite over Next.js?**
- Simpler — no SSR complexity needed for this project
- Vite is faster than Create React App
- You're learning; fewer abstractions = deeper understanding
- PDF preview is client-side anyway

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Colab GPU disconnects during training | Save checkpoints every epoch; resume from last checkpoint |
| Model too large for free Render tier (512MB RAM) | Use ONNX quantization to shrink model; or use `distilbert-base-NER` (smaller) |
| PDF text extraction fails on scanned PDFs | Add OCR fallback with `pytesseract` (Phase 6 stretch goal) |
| Regex false positives | Preview step (Phase 4) lets users exclude false matches |
| NER misses domain-specific terms | Allow custom keyword lists alongside NER |

---

## Stretch Goals (Post-MVP)

- [x] OCR support for scanned/image PDFs (pytesseract + Pillow)
- [x] Batch processing (upload multiple PDFs)
- [x] Redaction report (log of what was redacted, without showing the content)
- [x] User accounts + history (add PostgreSQL + auth)
- [x] Custom entity training UI (label your own data in-browser)
- [x] Browser extension for redacting PDFs without uploading
