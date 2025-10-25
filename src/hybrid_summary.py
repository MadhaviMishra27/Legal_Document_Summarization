"""
Step 7.1 — Hybrid Summary Generation
-------------------------------------

Implements the same hybrid summarization process from the research paper:

    Final Hybrid Summary = Extractive Summary (coverage) + Abstractive Summary (fluency)

Input:
    results/extractive/*.txt
    results/abstractive/*.txt

Output:
    results/hybrid/*.txt
    reports/metrics/hybrid_summary_stats.csv
"""

import os
import pandas as pd

EXTRACTIVE_DIR = "results/extractive"
ABSTRACTIVE_DIR = "results/abstractive"
HYBRID_DIR = "results/hybrid"
os.makedirs(HYBRID_DIR, exist_ok=True)

def find_matching_abstractive(base_name):
    """Return path of the abstractive summary matching given extractive base name."""
    for f in os.listdir(ABSTRACTIVE_DIR):
        if f.startswith(base_name) and f.endswith(".txt"):
            return os.path.join(ABSTRACTIVE_DIR, f)
    return None

def combine_summaries(extractive_text, abstractive_text):
    """
    Concatenate in the same order as in the paper:
        [Extractive Sentences]\n\n[Abstractive Summary]
    """
    return extractive_text.strip() + "\n\n" + abstractive_text.strip()

def main():
    stats = []
    for fname in os.listdir(EXTRACTIVE_DIR):
        if not fname.endswith(".txt"):
            continue
        base = fname.replace(".txt", "")
        ext_path = os.path.join(EXTRACTIVE_DIR, fname)
        abs_path = find_matching_abstractive(base)
        if not abs_path:
            continue

        with open(ext_path, "r", encoding="utf-8") as f:
            extractive_text = f.read().strip()
        with open(abs_path, "r", encoding="utf-8") as f:
            abstractive_text = f.read().strip()

        hybrid_text = combine_summaries(extractive_text, abstractive_text)

        out_path = os.path.join(HYBRID_DIR, f"{base}_hybrid.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(hybrid_text)

        stats.append({
            "file": base,
            "extractive_len": len(extractive_text.split()),
            "abstractive_len": len(abstractive_text.split()),
            "hybrid_len": len(hybrid_text.split())
        })

    if stats:
        df = pd.DataFrame(stats)
        out_csv = "reports/metrics/hybrid_summary_stats.csv"
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"✅ Hybrid summaries created and saved to {HYBRID_DIR}")
        print(f"✅ Summary statistics saved to {out_csv}")
    else:
        print("⚠️ No hybrid summaries generated. Check extractive/abstractive folders.")

if __name__ == "__main__":
    main()
