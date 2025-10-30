# Step 9
# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader
import os

st.set_page_config(page_title="Legal Document Summarizer", layout="wide")

st.title("⚖️ Legal Document Summarizer")
st.write("Upload a legal judgment (PDF or TXT), and the model will generate an abstractive summary.")

MODEL_DIR = "models/final"
MAX_INPUT_TOKENS = 1024  # chunk size for long documents
MAX_OUTPUT_TOKENS = 256

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    return model, tokenizer

def extract_text(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    elif ext == ".txt":
        return uploaded_file.read().decode("utf-8")
    else:
        st.error("❌ Unsupported file format. Please upload a .txt or .pdf file.")
        return None

def chunk_text(text, tokenizer, max_tokens=MAX_INPUT_TOKENS):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def summarize_chunk(text, model, tokenizer):
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

model, tokenizer = load_model()

uploaded_file = st.file_uploader("Upload your legal document", type=["pdf", "txt"])

if uploaded_file:
    text = extract_text(uploaded_file)
    if text:
        st.write(f"📄 **Input Document Length:** {len(text.split())} words")
        if st.button("Generate Summary"):
            with st.spinner("Generating summary... This may take some time for long documents."):
                # Chunking for long document
                chunks = chunk_text(text, tokenizer, MAX_INPUT_TOKENS)
                st.write(f"🔹 Total chunks: {len(chunks)}")
                
                summaries = []
                for i, chunk in enumerate(chunks, 1):
                    st.write(f"🔹 Summarizing chunk {i}/{len(chunks)}...")
                    summaries.append(summarize_chunk(chunk, model, tokenizer))
                
                final_summary = " ".join(summaries)
            
            st.subheader("🧾 Summary:")
            st.write(final_summary)
