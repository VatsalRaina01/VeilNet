import fitz  # PyMuPDF

def get_pdf_metadata(file_path: str):
    doc = fitz.open(file_path)
    metadata = {
        "page_count": len(doc),
        "metadata": doc.metadata,
        "is_encrypted": doc.is_encrypted
    }
    doc.close()
    return metadata