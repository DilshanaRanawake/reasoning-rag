# prepare_data.py
from pypdf import PdfReader
import json, os, re

PAPERS_DIR = "papers"          # put your PDFs here
OUTPUT_FILE = "chunks.json"
CHUNK_SIZE = 800                # characters per chunk
OVERLAP = 150

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

all_chunks = []
for filename in os.listdir(PAPERS_DIR):
    if not filename.endswith(".pdf"):
        continue
    paper_id = re.sub(r"\.pdf$", "", filename)
    text = extract_text(os.path.join(PAPERS_DIR, filename))
    for i, chunk in enumerate(chunk_text(text)):
        all_chunks.append({
            "id": f"{paper_id}-{i}",
            "paper": paper_id,
            "text": chunk,
        })

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_chunks, f, indent=2)

print(f"Wrote {len(all_chunks)} chunks from {PAPERS_DIR}/ to {OUTPUT_FILE}")