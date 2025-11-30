# scripts/prepare_kb.py
# Run: python scripts/prepare_kb.py
import os, re, json
from pathlib import Path
from PyPDF2 import PdfReader

# RAW_DIR = Path(r"C:\Users\Lenovo\lexibot\data\raw")
# OUT_FILE = Path(r"C:\Users\Lenovo\lexibot\data\legal_kb.json")

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/legal_kb.json")


def extract_text_from_pdf(path):
    reader = PdfReader(str(path))
    text = []
    for p in reader.pages:
        page_text = p.extract_text() or ""
        text.append(page_text)
    return "\n\n".join(text)

def split_into_sections(text):
    # Try to split by "Section <number>" headings (simple heuristic)
    pattern = r"(Section\s+\d+[A-Za-z\-]*\b[^\n]*)"
    # Find indices of section headings
    headings = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if not headings:
        # fallback: split by double newlines
        return [t.strip() for t in text.split("\n\n") if t.strip()]
    sections = []
    for i, m in enumerate(headings):
        start = m.start()
        end = headings[i+1].start() if i+1 < len(headings) else len(text)
        sec_text = text[start:end].strip()
        sections.append(sec_text)
    return sections

def simple_meta_from_section(sec_text, source_url=None, act_name=None):
    # small parser: get "Section <num>" and small title (first line)
    first_line = sec_text.splitlines()[0].strip()
    m = re.match(r"Section\s+(\d+[A-Za-z\-]*)\b(.*)", first_line, flags=re.IGNORECASE)
    section_number = m.group(1).strip() if m else ""
    title = m.group(2).strip(" -:") if m else first_line[:80]
    # small keywords: title words
    keywords = [w.strip('.,()') for w in title.split()[:6]]
    return {
        "id": f"{act_name or 'ACT'}_{section_number or 'X'}",
        "act_name": act_name or "Unknown",
        "section_number": section_number,
        "title": title,
        "text": sec_text,
        "keywords": keywords,
        "source_url": source_url or ""
    }

def main():
    kb = {"name": "LexiBot_KB", "version": "1.0", "entries": []}
    for f in RAW_DIR.iterdir():
        if f.suffix.lower() == ".pdf":
            print("Processing PDF:", f)
            text = extract_text_from_pdf(f)
            # you should set act_name/source_url manually or from filename
            act_name = f.stem
            # split into sections
            secs = split_into_sections(text)
            for s in secs:
                meta = simple_meta_from_section(s, source_url="", act_name=act_name)
                kb["entries"].append(meta)
        elif f.suffix.lower() in [".txt", ".md"]:
            txt = f.read_text(encoding="utf-8")
            secs = split_into_sections(txt)
            for s in secs:
                meta = simple_meta_from_section(s, source_url="", act_name=f.stem)
                kb["entries"].append(meta)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(kb, fh, indent=2, ensure_ascii=False)
    print("Saved KB to", OUT_FILE)

if __name__ == "__main__":
    main()
