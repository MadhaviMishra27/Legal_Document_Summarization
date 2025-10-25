"""
Step 7.2 — Model Saving for Prediction (Fixed for your models)
--------------------------------------------------------------

Selects the best abstractive model based on ROUGE-Sum
and saves it locally for later prediction use.
"""

import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

METRICS_FILE = "reports/metrics/model_comparison.csv"
MODEL_SAVE_DIR = "models/final"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ✅ Your actual model mapping
MODEL_MAP = {
    "mt5": "google/mt5-small",
    "bart-small": "sshleifer/distilbart-cnn-12-6",
}

def get_best_model_name():
    """Select model with highest ROUGE-Sum mean (as per paper)."""
    if not os.path.exists(METRICS_FILE):
        print(f"❌ Metrics file not found at {METRICS_FILE}")
        return None

    df = pd.read_csv(METRICS_FILE)
    abs_df = df[df["type"] == "abstractive"]
    if abs_df.empty:
        print("⚠️ No abstractive results found.")
        return None

    best_row = abs_df.loc[abs_df["rouge-sum_mean"].idxmax()]
    short_name = str(best_row["method"]).lower().strip()
    model_name = MODEL_MAP.get(short_name, short_name)

    print(f"🏆 Best model by ROUGE-Sum: {short_name} → {model_name}")
    return model_name


def save_model(model_name):
    """Download and save tokenizer + model."""
    print(f"\n🔹 Downloading and saving {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    model.save_pretrained(MODEL_SAVE_DIR)
    print(f"✅ Model and tokenizer saved to {MODEL_SAVE_DIR}")


if __name__ == "__main__":
    model_name = get_best_model_name()
    if model_name:
        save_model(model_name)
