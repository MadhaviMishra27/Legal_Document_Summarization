# Step 4: transformer summarization

"""
Optimized abstractive summarization for CPU (VS Code)
-----------------------------------------------------
- Uses LoRA adapters for all models (LED, Long-T5, BART)
- Automatically detects correct target modules
- Skips quantization (QLoRA) since CPU 4-bit is slow
- Lightweight generation configuration for faster runs
- Saves summaries and ROUGE metrics
"""

import os
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

# --------------------------------------------------------
# Optional LoRA (PEFT)
# --------------------------------------------------------
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️ PEFT not installed. Run: pip install peft")

# --------------------------------------------------------
# Directories
# --------------------------------------------------------
INPUT_DIR = "results/extractive"
OUTPUT_DIR = "results/abstractive"
METRICS_DIR = "reports/metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# --------------------------------------------------------
# Models from the paper
# --------------------------------------------------------
MODEL_PRESETS = [
    #{"name": "google/long-t5-tglobal-base", "shortname": "longt5"},
    #{"name": "facebook/bart-large-cnn", "shortname": "bart"}
    #{"name": "models/bart-small", "shortname": "bart-small"}
    {"name": "google/mt5-small", "shortname": "mt5"},
    {"name": "sshleifer/distilbart-cnn-12-6", "shortname": "bart-small"}
]


# --------------------------------------------------------
# Generation Configuration (Optimized for CPU)
# --------------------------------------------------------
GEN_CFG = {
    "max_length": 256,
    "num_beams": 2,             # smaller beam for speed
    "length_penalty": 0.8,
    "early_stopping": True,
    "no_repeat_ngram_size": 3
}

# ROUGE scorer
SCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# CPU optimization
device = torch.device("cpu")
torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))


# --------------------------------------------------------
# Utility Functions
# --------------------------------------------------------
def compute_rouge(reference, hypothesis):
    """Compute ROUGE scores"""
    sc = SCORER.score(reference, hypothesis)
    return {
        "rouge-1": sc["rouge1"].fmeasure,
        "rouge-2": sc["rouge2"].fmeasure,
        "rouge-L": sc["rougeL"].fmeasure,
        "rouge-sum": (sc["rouge1"].fmeasure + sc["rouge2"].fmeasure + sc["rougeL"].fmeasure) / 3.0
    }


def detect_lora_targets(model):
    """Auto-detect LoRA target modules in the transformer architecture."""
    targets = []
    for name, _ in model.named_modules():
        if any(k in name for k in ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]):
            targets.append(name.split('.')[-1])
    targets = list(set([m for m in targets if "proj" in m or m in ["fc1", "fc2"]]))
    if not targets:
        targets = ["q_proj", "v_proj"]  # fallback
    print(f"🔧 Detected target modules for LoRA: {targets}")
    return targets


def safe_load_model(model_name):
    """Load model + tokenizer + attach LoRA (for CPU)"""
    print(f"\n🔹 Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None

    # Attach LoRA if available
    if PEFT_AVAILABLE:
        try:
            target_modules = detect_lora_targets(model)
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
            model = get_peft_model(model, lora_config)
            print("✅ LoRA adapters attached successfully.")
        except Exception as e:
            print(f"⚠️ Failed to attach LoRA adapters: {e}")
    else:
        print("⚠️ PEFT not available. Running without LoRA.")

    return tokenizer, model


def summarize_text(tokenizer, model, text):
    """Chunk-aware summarization (handles long judgments)."""
    max_input = min(2048, getattr(tokenizer, "model_max_length", 4096))
    words = text.split()
    chunk_size = 700  # roughly ~700 words per chunk
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    summaries = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding="longest").to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **GEN_CFG)
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        summaries.append(decoded.strip())

    return " ".join(summaries)


# --------------------------------------------------------
# Main Summarization Process
# --------------------------------------------------------
def main():
    input_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    if not input_files:
        print(f"No extractive inputs found in {INPUT_DIR}. Run extractive stage first.")
        return

    results = []

    for preset in MODEL_PRESETS:
        model_name = preset["name"]
        short = preset["shortname"]

        tokenizer, model = safe_load_model(model_name)
        if model is None:
            continue

        for fname in tqdm(input_files, desc=f"Summarizing with {short}"):
            in_path = os.path.join(INPUT_DIR, fname)
            with open(in_path, "r", encoding="utf-8") as f:
                input_text = f.read().strip()
            if not input_text:
                continue

            summary = summarize_text(tokenizer, model, input_text)
            base = fname.replace(".txt", "")
            out_path = os.path.join(OUTPUT_DIR, f"{base}_{short}_abstractive.txt")
            with open(out_path, "w", encoding="utf-8") as outf:
                outf.write(summary)

            # Evaluate (reference = preprocessed or extractive)
            ref_path = os.path.join("data/preprocessed", base + ".txt")
            reference = input_text
            if os.path.exists(ref_path):
                with open(ref_path, "r", encoding="utf-8") as rf:
                    reference = rf.read().strip()

            scores = compute_rouge(reference, summary)
            results.append({"file": fname, "model": short, **scores})

        del model, tokenizer
        torch.cuda.empty_cache()

    # Save metrics
    if results:
        df = pd.DataFrame(results)
        out_metrics = os.path.join(METRICS_DIR, "abstractive_results_cpu_lora.csv")
        df.to_csv(out_metrics, index=False)
        print(f"\n✅ Saved metrics to {out_metrics}")
    else:
        print("\n⚠️ No abstractive results generated. Check earlier logs.")


if __name__ == "__main__":
    main()
