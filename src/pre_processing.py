import os
import re
from bs4 import BeautifulSoup
import spacy
from tqdm import tqdm

# -----------------------------
# ✅ Directories
# -----------------------------
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/pre_processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# -----------------------------
# ✅ Parameters
# -----------------------------
MAX_PAGES = 30         # keep only docs ≤ 30 pages
WORDS_PER_PAGE = 500   # ~500 words per page
LEGAL_STOPWORDS_FILE = "data/legal_stopwords.txt"  # create this file manually

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# ✅ Key outcome terms (for highlighting)
LEGAL_KEY_TERMS = [
    "allowed", "dismissed", "partly allowed", "disposed",
    "quashed", "set aside", "granted", "rejected",
    "convicted", "acquitted", "appeal allowed", "appeal dismissed",
    "writ petition allowed", "writ petition dismissed", "stayed"
]

# -----------------------------
# ✅ Load Custom Legal Stopwords
# -----------------------------
def load_stopwords():
    stopwords = set()
    if os.path.exists(LEGAL_STOPWORDS_FILE):
        with open(LEGAL_STOPWORDS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    stopwords.add(word)
    else:
        print(f"⚠️ No stopword file found at {LEGAL_STOPWORDS_FILE}. Using minimal set.")
        stopwords.update([
            "section", "subsection", "court", "judge", "justice",
            "petition", "respondent", "appellant", "case", "act",
            "order", "tribunal", "article", "rule", "advocate"
        ])
    return stopwords

LEGAL_STOPWORDS = load_stopwords()

# -----------------------------
# ✅ Extract Text from HTML
# -----------------------------
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
        soup.find("div", {"class": "doc_content"}),
        soup.find("body")
    ]

    for c in candidates:
        if c:
            text = c.get_text(separator="\n", strip=True)
            if text and len(text.split()) > 100:
                return text
    return ""

# -----------------------------
# ✅ Page Count
# -----------------------------
def approx_page_count(text):
    words = text.split()
    return len(words) // WORDS_PER_PAGE

# -----------------------------
# ✅ Cleaning + Normalization
# -----------------------------
def clean_and_normalize(text):
    """Remove headers, boilerplate, lowercase, remove punctuation and numbers."""
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        l = line.lower()

        # Remove site junk
        if any(x in l for x in [
            "premium member", "query alert", "sign up", "login", "email alert",
            "print", "search", "subscribe"
        ]):
            continue

        # Remove headers
        if l.startswith("in the high court") or l.startswith("in the supreme court"):
            continue

        cleaned.append(line)

    # Join and normalize
    text = " ".join(cleaned)
    # Remove punctuation, numbers, and extra spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text

# -----------------------------
# ✅ Tokenization + Stopword Removal + Outcome Highlight
# -----------------------------
def tokenize_and_filter(text):
    """Tokenize, remove stopwords and uninformative POS, highlight outcome terms."""
    doc = nlp(text)
    sentences = []
    total_tokens, removed_tokens = 0, 0
    key_sentences = 0

    for sent in doc.sents:
        tokens = []
        for token in sent:
            total_tokens += 1

            # Skip stopwords, spaces, and short tokens
            if token.is_stop or token.text.lower() in LEGAL_STOPWORDS:
                removed_tokens += 1
                continue
            if token.pos_ in ["DET", "PART", "SYM", "PUNCT", "SPACE"]:
                removed_tokens += 1
                continue

            tokens.append(token.lemma_.lower())

        sent_text = " ".join(tokens).strip()
        if not sent_text:
            continue

        # Highlight outcome terms
        for term in LEGAL_KEY_TERMS:
            if term.lower() in sent.text.lower():
                sent_text = f"[IMPORTANT] {sent.text.strip()}"
                key_sentences += 1
                break

        sentences.append(sent_text)

    return "\n".join(sentences), total_tokens, removed_tokens, key_sentences

# -----------------------------
# ✅ Main Preprocessing Function
# -----------------------------
def preprocess_all():
    total_processed, skipped, empty = 0, 0, 0
    total_tokens, total_removed, total_keywords = 0, 0, 0

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".html")]
    for file in tqdm(files, desc="Preprocessing documents"):
        input_path = os.path.join(RAW_DIR, file)
        output_path = os.path.join(PROCESSED_DIR, file.replace(".html", ".txt"))

        text = extract_text_from_html(input_path)
        if not text.strip():
            empty += 1
            continue

        pages = approx_page_count(text)
        if pages > MAX_PAGES:
            skipped += 1
            continue

        text = clean_and_normalize(text)
        processed_text, tok, rem, keys = tokenize_and_filter(text)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(processed_text)

        total_processed += 1
        total_tokens += tok
        total_removed += rem
        total_keywords += keys

    print("\n🎯 Preprocessing Summary")
    print(f"   ✅ Processed: {total_processed}")
    print(f"   ⏩ Skipped (> {MAX_PAGES} pages): {skipped}")
    print(f"   ⚠️ Empty/Failed: {empty}")
    print(f"   🧮 Tokens processed: {total_tokens}")
    print(f"   ❌ Tokens removed (stopwords/irrelevant): {total_removed}")
    print(f"   ⭐ Important sentences detected: {total_keywords}")

# -----------------------------
# ✅ Run
# -----------------------------
if __name__ == "__main__":
    preprocess_all()
