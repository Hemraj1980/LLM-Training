from pypdf import PdfReader
import json

pdf_path = "data/raw/ktx_docs_20240322.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

with open(
    "data/processed/extracted_text.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {"content": text},
        f,
        indent=2,
        ensure_ascii=False
    )

print("PDF Extracted")
print("Characters:", len(text))