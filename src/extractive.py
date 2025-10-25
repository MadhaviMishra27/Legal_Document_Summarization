# Step 3: TF-IDF / TextRank
import os
import re
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# ---------------------------------------------------
# 📁 Folder Paths (aligned with your project)
# ---------------------------------------------------
DATA_DIR = "data/pre_processed"
OUTPUT_DIR = "results/extractive"
METRICS_DIR = "reports/metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ---------------------------------------------------
# Legal Keyword Boost (same concept as paper)
# ---------------------------------------------------
LEGAL_KEYWORDS = [
    "appeal", "petition", "bail", "affidavit", "jurisdiction", "writ",
    "litigation", "judgment", "acquittal", "agreement", "indemnity",
    "liability", "dismissed", "allowed", "set aside", "rejected",
    "disposed", "convicted", "acquitted", "granted", "stayed"
]

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def sentence_split(text):
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if len(s.split()) > 3]  # remove short ones
    return sentences

def keyword_boost(sentence, base_score, boost_value=0.15):
    """Boost score if legal keyword present."""
    for kw in LEGAL_KEYWORDS:
        if kw.lower() in sentence.lower():
            return base_score * (1 + boost_value)
    return base_score

# ---------------------------------------------------
# TF-IDF Extractive Summarizer
# ---------------------------------------------------
def tfidf_summarizer(text, top_ratio=0.2):
    sentences = sentence_split(text)
    if len(sentences) < 5:
        return text, 0  # too short

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    scores = tfidf_matrix.mean(axis=1).A1  # mean TF-IDF per sentence

    # Apply keyword boost
    boosted_scores = [keyword_boost(sentences[i], s) for i, s in enumerate(scores)]

    # Select top N sentences
    top_n = max(1, int(len(sentences) * top_ratio))
    top_indices = np.argsort(boosted_scores)[-top_n:]
    top_indices.sort()  # preserve order

    summary = " ".join([sentences[i] for i in top_indices])
    return summary, np.mean(boosted_scores)

# ---------------------------------------------------
# TextRank Extractive Summarizer
# ---------------------------------------------------
def textrank_summarizer(text, top_ratio=0.2):
    sentences = sentence_split(text)
    if len(sentences) < 5:
        return text, 0

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(X)

    np.fill_diagonal(sim_matrix, 0)
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)

    ranked = sorted(((score, i) for i, score in scores.items()), reverse=True)
    top_n = max(1, int(len(sentences) * top_ratio))
    selected = sorted([i for (_, i) in ranked[:top_n]])

    summary = " ".join([sentences[i] for i in selected])
    return summary, np.mean(list(scores.values()))

# ---------------------------------------------------
# 📈Evaluation Metrics (F1, Precision, Recall, Cosine)
# ---------------------------------------------------
def evaluate_similarity(original, summary):
    """Compute cosine similarity, precision, recall, F1-score."""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([original, summary])
    cos_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    orig_words = set(original.lower().split())
    summ_words = set(summary.lower().split())
    common = orig_words.intersection(summ_words)

    precision = len(common) / len(summ_words) if summ_words else 0
    recall = len(common) / len(orig_words) if orig_words else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    return cos_sim, precision, recall, f1

# ---------------------------------------------------
# 🚀 Run Extractive Summarization for All Documents
# ---------------------------------------------------
def run_extractive():
    results = []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    print(f"📘 Found {len(files)} preprocessed documents.")

    for file in tqdm(files, desc="Extractive Summarization"):
        with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
            text = f.read()

        # Run both summarizers
        tfidf_summary, tfidf_score = tfidf_summarizer(text)
        textrank_summary, textrank_score = textrank_summarizer(text)

        # Save summaries
        base = file.replace(".txt", "")
        tfidf_path = os.path.join(OUTPUT_DIR, f"{base}_tfidf.txt")
        textrank_path = os.path.join(OUTPUT_DIR, f"{base}_textrank.txt")

        with open(tfidf_path, "w", encoding="utf-8") as f:
            f.write(tfidf_summary)
        with open(textrank_path, "w", encoding="utf-8") as f:
            f.write(textrank_summary)

        # Evaluate
        tfidf_metrics = evaluate_similarity(text, tfidf_summary)
        textrank_metrics = evaluate_similarity(text, textrank_summary)

        results.append({
            "document": file,
            "tfidf_cosine": tfidf_metrics[0],
            "tfidf_precision": tfidf_metrics[1],
            "tfidf_recall": tfidf_metrics[2],
            "tfidf_f1": tfidf_metrics[3],
            "textrank_cosine": textrank_metrics[0],
            "textrank_precision": textrank_metrics[1],
            "textrank_recall": textrank_metrics[2],
            "textrank_f1": textrank_metrics[3],
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(METRICS_DIR, "extractive_results.csv"), index=False)
    print("\n✅ Extractive summarization completed.")
    print(f"📊 Results saved to: {os.path.join(METRICS_DIR, 'extractive_results.csv')}")
    print(f"📝 Summaries saved in: {OUTPUT_DIR}")

# ---------------------------------------------------
# 🔧 Main
# ---------------------------------------------------
if __name__ == "__main__":
    run_extractive()
