from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import fitz  # PyMuPDF
import json
from typing import List

router = APIRouter()

@router.post("/redact-pdf")
async def redact_pdf(
    file: UploadFile = File(...),
    findings: str = Form(...)  # Expecting JSON string of findings
):
    """
    Accepts a PDF and a JSON string of findings containing coordinates.
    Redacts the specified areas and returns the redacted PDF as a downloadable file.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
        
    try:
        findings_data = json.loads(findings)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid findings format. Expected valid JSON.")

    file_content = await file.read()
    
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to open PDF: {str(e)}")

    # Apply redactions to each page
    for finding in findings_data:
        # findings usually contain { 'page': 1, 'coords': [x0, y0, x1, y1] } or manual word { 'page': 1, 'value': 'secret' }
        page_num = finding.get("page", 1) - 1  # 0-indexed for fitz
        coords = finding.get("coords")
        value = finding.get("value")
        
        if 0 <= page_num < len(doc):
            page = doc[page_num]
            
            if coords and len(coords) == 4:
                rect = fitz.Rect(coords[0], coords[1], coords[2], coords[3])
                page.add_redact_annot(rect, fill=(0, 0, 0))
            elif value:
                # If no coords are supplied (manual word selection), sweep page to locate coordinates and redact natively!
                rects = page.search_for(value)
                for rect in rects:
                    page.add_redact_annot(rect, fill=(0, 0, 0))
            
    # Commit redactions on all pages
    for page in doc:
        page.apply_redactions()
        
    # Generate the redacted PDF byte stream
    redacted_pdf_bytes = doc.write()
    doc.close()
    
    # Return as a downloadable PDF stream
    return Response(
        content=redacted_pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="redacted_{file.filename}"'
        }
    )
