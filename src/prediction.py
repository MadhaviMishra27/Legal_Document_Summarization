# src/predict.py
"""
Final prediction/inference script for deployed legal summarizer.
Supports both TXT and PDF input.
Handles long documents using chunking.
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader

# ---------------------- CONFIG ----------------------
MODEL_DIR = "models/final"
INPUT_FILE = "Test_Files/supreme_court.pdf"  # or .txt
OUTPUT_FILE = "Test_Output/supreme_court_summary.txt"

MAX_INPUT_TOKENS = 1024  # chunk size for BART
MAX_OUTPUT_TOKENS = 256
# ----------------------------------------------------


def load_text(file_path):
    """Load text from TXT or PDF file."""
    if not os.path.exists(file_path):
        print(f"❌ Input file not found: {file_path}")
        sys.exit(1)

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    elif ext == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    else:
        print("❌ Unsupported file format. Please provide .txt or .pdf")
        sys.exit(1)


def chunk_text(text, tokenizer, max_tokens=MAX_INPUT_TOKENS):
    """Split text into token-based chunks for long documents."""
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]


def summarize_text(text, model, tokenizer):
    """Generate abstractive summary for a chunk of text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="longest", max_length=MAX_INPUT_TOKENS)
    output_ids = model.generate(
        **inputs,
        max_length=MAX_OUTPUT_TOKENS,
        num_beams=4,
        length_penalty=0.8,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    print(f"🔹 Loading model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

    text = load_text(INPUT_FILE)
    print(f"✅ Loaded input ({len(text.split())} words). Splitting into chunks...")

    chunks = chunk_text(text, tokenizer, MAX_INPUT_TOKENS)
    print(f"🔹 Total chunks: {len(chunks)}")

    chunk_summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"🔹 Summarizing chunk {i}/{len(chunks)} ...")
        summary = summarize_text(chunk, model, tokenizer)
        chunk_summaries.append(summary)

    final_summary_text = " ".join(chunk_summaries)
    print("✅ All chunks summarized. Saving final summary...")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_summary_text)

    print(f"✅ Final summary saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
