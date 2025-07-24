import streamlit as st
import numpy as np
import pickle
import re, string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MODEL_PATH = "lstm_toxic_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 200
LABELS = ['toxic', 'severe_toxic', 'obscene',
          'threat', 'insult', 'identity_hate']

# -----------------------------------------------------------------------------
# Load model and tokenizer
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

# -----------------------------------------------------------------------------
# Text cleaning and preprocessing
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(texts):
    cleaned = [clean_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    return pad_sequences(seqs, maxlen=MAX_LEN)

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Toxic Comment Classifier", layout="centered")
st.title("🛡️ Toxic Comment Multi-Label Classifier")

st.write(
    "Enter a comment below and the model will predict whether it is "
    "toxic, obscene, insulting, threatening, **etc.**"
)

user_input = st.text_area("Comment text", height=150)

if st.button("Predict") and user_input.strip():
    X = preprocess([user_input])
    probas = model.predict(X)[0]          # shape (6,)
    preds = (probas >= 0.5).astype(int)   # binary labels

    # Build human-readable result
    positive_labels = [LABELS[i] for i, p in enumerate(preds) if p == 1]
    
    st.subheader("🧪 Prediction")
    if positive_labels:
        st.success("This comment is classified as: **" + ", ".join(positive_labels) + "**")
    else:
        st.info("✅ This comment is **clear** – no toxic categories detected.")

    # Optional: Show probabilities
    with st.expander("🔬 Show probabilities"):
        result_table = {
            "Category": LABELS,
            "Probability": np.round(probas, 3),
            "Toxic?": ["✅" if p else "❌" for p in preds],
        }
        st.table(result_table)

st.markdown("---")
st.caption("Model: Bidirectional LSTM • Trained on Jigsaw Toxic Comment dataset • Threshold = 0.5")
