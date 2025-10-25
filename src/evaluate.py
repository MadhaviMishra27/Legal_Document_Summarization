# Step 6: ROUGE/BLEU/BERTScore
# src/evaluate.py
"""
Evaluation script following the research paper exactly.

- Loads:
    - reports/metrics/extractive_results.csv
    - reports/metrics/abstractive_results*.csv  (any CSVs in that folder that start with 'abstractive_results')
- Computes:
    - Aggregated extractive metrics (mean/std of cosine, precision, recall, f1)
    - Aggregated abstractive metrics (mean/std of rouge-1, rouge-2, rouge-L, rouge-sum)
    - Word reduction percentages per-document and per-method
- Outputs:
    - reports/metrics/model_comparison.csv  (aggregated table)
    - reports/visuals/rouge_sum_comparison.png
    - reports/visuals/extractive_f1_comparison.png

This script intentionally implements the paper's evaluation metrics only.
"""

import os
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Directories (must match your project)
METRICS_DIR = "reports/metrics"
VISUALS_DIR = "reports/visuals"
PREPROCESSED_DIR = "data/pre_processed"
EXTRACTIVE_RESULTS_FILE = os.path.join(METRICS_DIR, "extractive_results.csv")
OUT_COMPARISON_CSV = os.path.join(METRICS_DIR, "model_comparison.csv")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)


def safe_read_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None


def compute_word_counts_for_preprocessed():
    """
    Build a map {base_filename: token_count} for preprocessed documents.
    base_filename is file name without extension, e.g. 'case1' for 'case1.txt'.
    """
    counts = {}
    if not os.path.isdir(PREPROCESSED_DIR):
        return counts

    for fname in os.listdir(PREPROCESSED_DIR):
        if not fname.endswith(".txt"):
            continue
        base = fname.replace(".txt", "")
        p = os.path.join(PREPROCESSED_DIR, fname)
        try:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read().strip()
            tokens = text.split()
            counts[base] = len(tokens)
        except Exception:
            counts[base] = 0
    return counts


def compute_word_counts_for_results(result_dir, suffix):
    """
    For summaries saved in result_dir, with filenames like "{base}_{suffix}.txt"
    compute token counts per document and return dict {(base, method): token_count}
    """
    counts = {}
    if not os.path.isdir(result_dir):
        return counts

    for fname in os.listdir(result_dir):
        if not fname.endswith(".txt"):
            continue
        # expect pattern: base_method.txt or base_method_abstractive.txt
        base_method = fname.replace(".txt", "")
        # split last underscore to get model/method: base + method parts
        if "_" in base_method:
            parts = base_method.rsplit("_", 1)
            base = parts[0]
            method = parts[1]
        else:
            base = base_method
            method = "unknown"

        p = os.path.join(result_dir, fname)
        try:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read().strip()
            counts[(base, method)] = len(text.split())
        except Exception:
            counts[(base, method)] = 0
    return counts


def aggregate_extractive_metrics(df):
    """
    Expects df to have columns: document, tfidf_cosine, tfidf_precision, tfidf_recall, tfidf_f1,
                                textrank_cosine, textrank_precision, textrank_recall, textrank_f1
    Returns per-method aggregated metrics (mean, std).
    """
    rows = []
    # TF-IDF
    if {"tfidf_cosine", "tfidf_precision", "tfidf_recall", "tfidf_f1"}.issubset(df.columns):
        rows.append({
            "method": "tfidf",
            "cosine_mean": df["tfidf_cosine"].mean(),
            "cosine_std": df["tfidf_cosine"].std(),
            "precision_mean": df["tfidf_precision"].mean(),
            "recall_mean": df["tfidf_recall"].mean(),
            "f1_mean": df["tfidf_f1"].mean()
        })
    # TextRank
    if {"textrank_cosine", "textrank_precision", "textrank_recall", "textrank_f1"}.issubset(df.columns):
        rows.append({
            "method": "textrank",
            "cosine_mean": df["textrank_cosine"].mean(),
            "cosine_std": df["textrank_cosine"].std(),
            "precision_mean": df["textrank_precision"].mean(),
            "recall_mean": df["textrank_recall"].mean(),
            "f1_mean": df["textrank_f1"].mean()
        })
    return pd.DataFrame(rows)


def aggregate_abstractive_metrics(list_of_dfs):
    """
    Expects list_of_dfs where each df contains columns:
    input_file, model, output_file, rouge-1, rouge-2, rouge-L, rouge-sum
    Returns dataframe aggregated by model.
    """
    if not list_of_dfs:
        return pd.DataFrame(columns=["model", "rouge-1_mean", "rouge-2_mean", "rouge-L_mean", "rouge-sum_mean"])

    df = pd.concat(list_of_dfs, ignore_index=True)
    # ensure column names are correct
    needed = {"input_file", "model", "output_file", "rouge-1", "rouge-2", "rouge-L", "rouge-sum"}
    # If the CSVs have slightly different column names, try to rename common variations
    if not needed.issubset(set(df.columns)):
        # try to locate approximate columns
        colmap = {}
        for c in df.columns:
            lc = c.lower()
            if "rouge1" in lc or "rouge-1" in lc or "rouge_1" in lc:
                colmap[c] = "rouge-1"
            if "rouge2" in lc or "rouge-2" in lc or "rouge_2" in lc:
                colmap[c] = "rouge-2"
            if "rougel" in lc or "rouge-l" in lc or "rouge_l" in lc:
                colmap[c] = "rouge-L"
            if "rougesum" in lc or "rouge-sum" in lc or "rouge_sum" in lc:
                colmap[c] = "rouge-sum"
        if colmap:
            df = df.rename(columns=colmap)

    # group by model and compute mean and std
    agg = df.groupby("model").agg({
        "rouge-1": ["mean", "std"],
        "rouge-2": ["mean", "std"],
        "rouge-L": ["mean", "std"],
        "rouge-sum": ["mean", "std"]
    })
    # flatten columns
    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index().rename(columns={
        "rouge-1_mean": "rouge-1_mean",
        "rouge-2_mean": "rouge-2_mean",
        "rouge-L_mean": "rouge-L_mean",
        "rouge-sum_mean": "rouge-sum_mean"
    })
    return agg


def compute_word_reduction(pre_counts, extractive_counts, abstractive_counts):
    """
    Compute word reduction percentages aligned with the paper:
    - Extractive reduction % = 100 * (1 - extractive_len / original_len)
    - Abstractive reduction % = 100 * (1 - abstractive_len / extractive_len)
    We'll compute average per-method.
    """
    # Build per-document lists per method for averaging
    extractive_reductions = []
    abstractive_reductions = []

    # extractive_counts keys are (base, method)
    for (base, method), ext_len in extractive_counts.items():
        orig_len = pre_counts.get(base, None)
        if orig_len and orig_len > 0:
            red = 100.0 * (1.0 - (ext_len / orig_len))
            extractive_reductions.append({"method": method, "base": base, "reduction_pct": red})

    # abstractive_counts keys are (base, method)
    # interpret abstractive method as model shortname (e.g., base_tlongt5_abstractive -> method 'longt5_abstractive' earlier)
    for (base, method), abs_len in abstractive_counts.items():
        # match an extractive counterpart: usually extractive files are base_tfidf or base_textrank; we will try to match either
        # prefer tfidf length
        ext_len = None
        for candidate in [(base, "tfidf"), (base, "textrank")]:
            if candidate in extractive_counts:
                ext_len = extractive_counts[candidate]
                break
        if ext_len and ext_len > 0:
            red = 100.0 * (1.0 - (abs_len / ext_len))
            abstractive_reductions.append({"method": method, "base": base, "reduction_pct": red})

    # aggregate averages per method
    ext_df = pd.DataFrame(extractive_reductions)
    abs_df = pd.DataFrame(abstractive_reductions)
    ext_agg = pd.DataFrame()
    abs_agg = pd.DataFrame()
    if not ext_df.empty:
        ext_agg = ext_df.groupby("method")["reduction_pct"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "extractive_reduction_mean", "std": "extractive_reduction_std"})
    if not abs_df.empty:
        abs_agg = abs_df.groupby("method")["reduction_pct"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "abstractive_reduction_mean", "std": "abstractive_reduction_std"})
    return ext_agg, abs_agg


def plot_bar(df, x_col, y_col, title, out_path, ylabel=None):
    plt.figure(figsize=(8, 5))
    plt.bar(df[x_col].astype(str), df[y_col], color="tab:blue")
    plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    plt.xlabel(x_col)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    # 1) Load extractive results
    df_ex = safe_read_csv(EXTRACTIVE_RESULTS_FILE)
    if df_ex is None:
        print(f"⚠️ Extractive results file not found at {EXTRACTIVE_RESULTS_FILE}. Ensure extractive stage has been run.")
    else:
        ext_agg = aggregate_extractive_metrics(df_ex)
        print("\nExtractive aggregated metrics (paper-style):")
        print(ext_agg)

    # 2) Load all abstractive results CSVs in metrics folder that start with 'abstractive_results'
    abstractive_files = sorted(glob.glob(os.path.join(METRICS_DIR, "abstractive_results*.csv")))
    abstractive_dfs = []
    for f in abstractive_files:
        df = safe_read_csv(f)
        if df is not None:
            abstractive_dfs.append(df)
    if not abstractive_dfs:
        print(f"⚠️ No abstractive results CSVs found in {METRICS_DIR}. Expected files like 'abstractive_results.csv' or 'abstractive_results_*.csv'.")
        abs_agg = pd.DataFrame()
    else:
        abs_agg = aggregate_abstractive_metrics(abstractive_dfs)
        print("\nAbstractive aggregated metrics (paper-style):")
        print(abs_agg)

    # 3) Word counts (to compute reduction percentages)
    pre_counts = compute_word_counts_for_preprocessed()
    extractive_counts = compute_word_counts_for_results("results/extractive", "extractive")
    abstractive_counts = compute_word_counts_for_results("results/abstractive", "abstractive")
    ext_reduction_agg, abs_reduction_agg = compute_word_reduction(pre_counts, extractive_counts, abstractive_counts)

    # 4) Build a combined comparison table aligning with paper's presentation
    rows = []
    # include extractive aggregated rows
    if df_ex is not None:
        for _, r in ext_agg.iterrows():
            rows.append({
                "method": r["method"],
                "type": "extractive",
                "cosine_mean": r["cosine_mean"],
                "cosine_std": r["cosine_std"],
                "precision_mean": r["precision_mean"],
                "recall_mean": r["recall_mean"],
                "f1_mean": r["f1_mean"],
                "rouge-1_mean": np.nan,
                "rouge-2_mean": np.nan,
                "rouge-L_mean": np.nan,
                "rouge-sum_mean": np.nan
            })

    # include abstractive aggregated rows
    if not abs_agg.empty:
        for _, r in abs_agg.iterrows():
            rows.append({
                "method": r["model"],
                "type": "abstractive",
                "cosine_mean": np.nan,
                "cosine_std": np.nan,
                "precision_mean": np.nan,
                "recall_mean": np.nan,
                "f1_mean": np.nan,
                "rouge-1_mean": r.get("rouge-1_mean", np.nan),
                "rouge-2_mean": r.get("rouge-2_mean", np.nan),
                "rouge-L_mean": r.get("rouge-L_mean", np.nan),
                "rouge-sum_mean": r.get("rouge-sum_mean", np.nan)
            })

    comparison_df = pd.DataFrame(rows)
    # merge reduction statistics where appropriate
    if not ext_reduction_agg.empty:
        # ext_reduction_agg has column 'method' matching extractive methods like 'tfidf'/'textrank'
        comparison_df = comparison_df.merge(ext_reduction_agg, how="left", left_on="method", right_on="method")
    if not abs_reduction_agg.empty:
        comparison_df = comparison_df.merge(abs_reduction_agg, how="left", left_on="method", right_on="method", suffixes=("_ext", "_abs"))

    # Save comparison CSV
    comparison_df.to_csv(OUT_COMPARISON_CSV, index=False)
    print(f"\n✅ Saved model comparison to: {OUT_COMPARISON_CSV}")

    # 5) Visuals (strictly: ROUGE-Sum comparison for abstractive; Extractive F1 comparison)
    if not abs_agg.empty:
        # abs_agg currently aggregated by model index; restructure for plotting
        try:
            plot_df = abs_agg[["model", "rouge-sum_mean"]].copy()
            plot_df = plot_df.dropna()
            plot_path = os.path.join(VISUALS_DIR, "rouge_sum_comparison.png")
            plot_bar(plot_df, "model", "rouge-sum_mean", "ROUGE-Sum by Model (avg)", plot_path, ylabel="ROUGE-Sum")
            print(f"✅ Saved ROUGE-Sum comparison plot to: {plot_path}")
        except Exception as e:
            print(f"⚠️ Failed to plot abstractive ROUGE comparison: {e}")

    if df_ex is not None:
        try:
            f1_df = ext_agg[["method", "f1_mean"]].copy()
            plot_path = os.path.join(VISUALS_DIR, "extractive_f1_comparison.png")
            plot_bar(f1_df, "method", "f1_mean", "Extractive methods — F1 mean", plot_path, ylabel="F1")
            print(f"✅ Saved extractive F1 comparison plot to: {plot_path}")
        except Exception as e:
            print(f"⚠️ Failed to plot extractive comparison: {e}")

    print("\nEvaluation complete. The outputs follow the paper's evaluation metrics and are saved in reports/metrics and reports/visuals.")


if __name__ == "__main__":
    main()
