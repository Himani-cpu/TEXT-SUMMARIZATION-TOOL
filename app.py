# app.py

import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk
import torch

from io import StringIO
import base64

# Download tokenizer
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

# Load BART model/tokenizer
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_model.to("cpu")

# --- Summarization Functions ---

def summarize_with_bart(text, max_len=130, min_len=30):
    inputs = bart_tokenizer([text], return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    with torch.no_grad():
        summary_ids = bart_model.generate(inputs["input_ids"], num_beams=4, max_length=max_len, min_length=min_len, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_with_textrank(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return ' '.join(str(sentence) for sentence in summary)

def get_download_link(text, filename="summary.txt"):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Download Summary</a>'

# --- Streamlit UI ---
st.set_page_config(page_title="AI Text Summarizer", layout="wide")

st.markdown("<h1 style='text-align: center;'>🧠 AI Text Summarizer</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🧰 Settings")
# Method selection with reset on change
if "last_method" not in st.session_state:
    st.session_state.last_method = "🧠 BART Transformer"

method = st.sidebar.radio("Choose Summarization Method:", ["🧠 BART Transformer", "⚡ TextRank (Fast)"])

# Reset summary if method changed
if method != st.session_state.last_method:
    st.session_state.summary_done = False
    st.session_state.summary_text = ""
    st.session_state.last_method = method

if method == "🧠 BART Transformer":
    max_len = st.sidebar.slider("Max Summary Length", 50, 300, 130, step=10)
    min_len = st.sidebar.slider("Min Summary Length", 10, 100, 30, step=5)
else:
    sentence_count = st.sidebar.slider("No. of Sentences (TextRank)", 1, 10, 3)

st.sidebar.markdown("Made with ❤️ using [Streamlit](https://streamlit.io)")

# Main Layout
tab1, tab2 = st.tabs(["📄 Input Text", "📝 Summary"])

with tab1:
    text_input = st.text_area("Paste your text here:", height=300, placeholder="Type or paste your article here...")

# Initialize session state
if "summary_done" not in st.session_state:
    st.session_state.summary_done = False
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""

# When button is clicked
if st.button("✨ Generate Summary"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        with st.spinner("🔄 Summarizing..."):
            try:
                if method == "🧠 BART Transformer":
                    summary = summarize_with_bart(text_input, max_len=max_len, min_len=min_len)
                else:
                    summary = summarize_with_textrank(text_input, sentence_count)

                # Store summary in session
                st.session_state.summary_text = summary
                st.session_state.summary_done = True

            except Exception as e:
                st.session_state.summary_text = f"🚨 Error: {e}"
                st.session_state.summary_done = False

# Display results in both tabs
if st.session_state.summary_done:


    with tab2:
        st.success("✅ Summary generated below:")
        st.text_area("Summary:", st.session_state.summary_text, height=250, key="summary_output_tab")
        st.markdown(get_download_link(st.session_state.summary_text), unsafe_allow_html=True)
