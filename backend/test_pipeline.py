"""Test the full pipeline against a real resume PDF."""
import requests
import json

PDF_PATH = r"c:\Users\Vinit\Project VeilNet\VeilNet\testPDFs\AryanResume.pdf"

with open(PDF_PATH, "rb") as f:
    r = requests.post(
        "http://127.0.0.1:8000/analyze-pdf",
        files={"file": ("resume.pdf", f, "application/pdf")},
    )

data = r.json()

print("=" * 65)
print("FULL PIPELINE RESULTS")
print("=" * 65)

di = data["document_info"]
print(f"\nDocument Type  : {di['doc_type']} (confidence: {di['doc_type_confidence']})")
print(f"Sections Found : {di['sections_detected']}")
for s in di.get("sections", []):
    print(f"  [{s['sensitivity']:>7}] {s['name']}")

rs = data["risk_summary"]
print(f"\nOverall Risk   : {rs['overall_risk']}")
print(f"Total Findings : {rs['total']}")
print(f"  Critical: {rs['critical']}  High: {rs['high']}  Medium: {rs['medium']}  Low: {rs['low']}")
print(f"  By Type: {json.dumps(rs['by_type'])}")

print(f"\n{'RISK':>8} | {'SOURCE':>11} | {'TYPE':>10} | VALUE")
print("-" * 65)
for f in data["findings"]:
    val = f["value"][:40]
    conf = f.get("confidence", "")
    conf_str = f" ({conf:.2f})" if conf else ""
    section = f" [{f['section']}]" if f.get("section") else ""
    print(f"{f['risk_level']:>8} | {f['source']:>11} | {f['type']:>10} | {val}{conf_str}{section}")

links = data.get("entity_links", [])
if links:
    print(f"\nEntity Links ({len(links)}):")
    for l in links[:10]:
        print(f"  {l['entity_a']} ({l['entity_a_type']}) --[{l['relationship']}]--> {l['entity_b']} ({l['entity_b_type']})")

print("\n" + "=" * 65)
print("PIPELINE COMPLETE")
print("=" * 65)
