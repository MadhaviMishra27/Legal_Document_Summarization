import os
from bs4 import BeautifulSoup
import spacy

# Directories
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/preprocessed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Limits
MAX_PAGES = 30      # keep only docs ≤ 30 pages
WORDS_PER_PAGE = 500  # ~500 words per page

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# ✅ Key outcome terms (to highlight)
LEGAL_KEY_TERMS = [
    "allowed", "dismissed", "partly allowed", "disposed",
    "quashed", "set aside", "granted", "rejected",
    "convicted", "acquitted", "appeal allowed", "appeal dismissed",
    "writ petition allowed", "writ petition dismissed", "stayed"
]

def extract_text_from_html(filepath):
    """Extract judgment text from Indian Kanoon HTML with multiple fallbacks."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return ""

    candidates = [
        soup.find("div", {"class": "judgments"}),
        soup.find("div", {"id": "judgment"}),
        soup.find("div", {"class": "content"}),
        soup.find("div", {"id": "content-wrapper"}),
        soup.find("div", {"class": "doc_content"}),  # some cases use this
        soup.find("body")  # last fallback: entire body text
    ]

    for c in candidates:
        if c:
            text = c.get_text(separator="\n", strip=True)
            if text and len(text.split()) > 100:  # avoid very short junk pages
                return text
    return ""

def approx_page_count(text):
    words = text.split()
    return len(words) // WORDS_PER_PAGE

def clean_text(text):
    """Remove headers, boilerplate, duplicates."""
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()
        # Remove boilerplate junk
        if not line:
            continue
        if "premium member" in line.lower():
            continue
        if "query alert service" in line.lower():
            continue
        if "sign up today" in line.lower():
            continue
        if line.upper().startswith("IN THE HIGH COURT"):
            continue

        cleaned.append(line)

    # Remove duplicate lines
    cleaned = list(dict.fromkeys(cleaned))
    return "\n".join(cleaned)

def tokenize_sentences(text):
    """Split text into sentences, keep tokens, highlight key outcomes."""
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        s = sent.text.strip()
        if not s:
            continue

        # Highlight outcome terms
        for term in LEGAL_KEY_TERMS:
            if term.lower() in s.lower():
                s = f"[IMPORTANT] {s}"
                break
        sentences.append(s)
    return "\n".join(sentences)

def preprocess_all():
    count, skipped, empty = 0, 0, 0
    for file in os.listdir(RAW_DIR):
        if not file.endswith(".html"):
            continue

        input_path = os.path.join(RAW_DIR, file)
        output_path = os.path.join(PROCESSED_DIR, file.replace(".html", ".txt"))

        text = extract_text_from_html(input_path)
        if not text.strip():
            print(f"⚠️ No text extracted from {file}")
            empty += 1
            continue

        pages = approx_page_count(text)
        if pages > MAX_PAGES:
            print(f"⏩ Skipped {file} ({pages} pages > {MAX_PAGES})")
            skipped += 1
            continue

        # Step 1: Clean
        text = clean_text(text)

        # Step 2: Sentence segmentation + tokenization
        text = tokenize_sentences(text)

        # Save
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        count += 1
        print(f"✅ Processed {file} → {output_path} ({pages} pages)")

    print(f"\n🎉 Preprocessing Summary:")
    print(f"   ✅ Processed: {count}")
    print(f"   ⏩ Skipped (too long): {skipped}")
    print(f"   ⚠️ Empty/failed: {empty}")

if __name__ == "__main__":
    preprocess_all()
