import fitz # PyMuPDF

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
                            "bbox": span["bbox"],
                            "page": page_num
                        })
    return results