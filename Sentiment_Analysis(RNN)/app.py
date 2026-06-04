import re
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────
MAX_WORDS = 10000
MAX_LEN   = 50

# ── PAGE SETUP ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Type a movie review and the model will predict if it's **Positive** or **Negative**.")
st.divider()

# ── LOAD MODEL & TOKENIZER ───────────────────────────────────────────
@st.cache_resource
def load_resources():
    # Load model
    model = load_model("best_model.keras")

    # Rebuild tokenizer from dataset (must match training)
    df = pd.read_csv("IMDB Dataset.csv")
    df["review"] = df["review"].apply(clean_text)
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["review"])
    return model, tokenizer

# ── TEXT CLEANING ────────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.lower()

# ── PREDICT ──────────────────────────────────────────────────────────
def predict_sentiment(text, model, tokenizer):
    cleaned = clean_text(text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    prob    = model.predict(padded, verbose=0)[0][0]
    label   = "Positive" if prob > 0.5 else "Negative"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, float(confidence)

# ── LOAD ─────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    try:
        model, tokenizer = load_resources()
        st.success("Model loaded!", icon="✅")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure `best_model.keras` and `IMDB Dataset.csv` are in the same folder as `app.py`.")
        st.stop()

# ── INPUT ─────────────────────────────────────────────────────────────
review = st.text_area(
    "✍️ Enter a movie review:",
    placeholder="e.g. This movie was absolutely amazing, I loved every second!",
    height=150
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

# ── RESULT ────────────────────────────────────────────────────────────
if predict_btn:
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            label, confidence = predict_sentiment(review, model, tokenizer)

        st.divider()

        if label == "Positive":
            st.success(f"## 😊 {label} Review")
        else:
            st.error(f"## 😞 {label} Review")

        st.metric(label="Confidence", value=f"{confidence * 100:.1f}%")

        # Confidence bar
        st.progress(confidence)
        st.caption(f"The model is **{confidence * 100:.1f}%** confident this review is **{label}**.")

# ── BATCH SECTION ────────────────────────────────────────────────────
st.divider()
with st.expander("📋 Try Multiple Reviews at Once"):
    examples = st.text_area(
        "Enter one review per line:",
        placeholder="This film was incredible!\nBooring and way too long.\nGreat acting, poor storyline.",
        height=120
    )
    if st.button("Analyze All"):
        lines = [l.strip() for l in examples.strip().split("\n") if l.strip()]
        if lines:
            results = []
            for line in lines:
                lbl, conf = predict_sentiment(line, model, tokenizer)
                emoji = "😊" if lbl == "Positive" else "😞"
                results.append({"Review": line, "Sentiment": f"{emoji} {lbl}", "Confidence": f"{conf*100:.1f}%"})
            st.table(pd.DataFrame(results))
        else:
            st.warning("Please enter at least one review.")

# ── FOOTER ───────────────────────────────────────────────────────────
st.divider()
st.caption("Built with LSTM · Trained on IMDB Dataset · Streamlit UI")