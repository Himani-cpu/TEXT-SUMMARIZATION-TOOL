# app.py
# 🚫 Suppress torch UserWarnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from gtts import gTTS
from io import StringIO
import base64
import nltk
import torch

# Download NLTK tokenizer
for resource in ["punkt"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

# Load BART model/tokenizer
bart_tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6", torch_dtype=torch.float32)
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

st.markdown("<h1 style='text-align: center;'>🤖 AI Text Summarizer</h1>", unsafe_allow_html=True)

# Sidebar Settings
st.sidebar.header("🧰 Settings")

# Theme Toggle
theme = st.sidebar.radio("🎨 Theme", ["🌞 Light", "🌙 Dark"], horizontal=True)
if theme == "🌙 Dark":
    st.markdown("""
    <style>
    body, .stApp {
        background-color: #1e1e1e;
        color: white;
    }
    .stTextArea, .stButton, .stRadio, .stSlider {
        background-color: #333333;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Summarization Method
method = st.sidebar.radio("Choose Summarization Method:", ["🧠 BART Transformer", "⚡ TextRank (Fast)", "🆚 Compare Both"])

# Params
if method == "🧠 BART Transformer":
    max_len = st.sidebar.slider("Max Summary Length", 50, 300, 130, step=10)
    min_len = st.sidebar.slider("Min Summary Length", 10, 100, 30, step=5)
elif method == "⚡ TextRank (Fast)":
    sentence_count = st.sidebar.slider("No. of Sentences", 1, 10, 3)
elif method == "🆚 Compare Both":
    max_len = st.sidebar.slider("BART Max Length", 50, 300, 130, step=10)
    min_len = st.sidebar.slider("BART Min Length", 10, 100, 30, step=5)
    sentence_count = st.sidebar.slider("TextRank Sentences", 1, 10, 3)

st.sidebar.markdown("Made with ❤️ using [Streamlit](https://streamlit.io)")

# Tabs
tab1, tab2 = st.tabs(["📄 Input Text", "📝 Summary / Comparison"])

# File/Text input
with tab1:
    uploaded_file = st.file_uploader("📂 Upload a `.txt` file:", type=["txt"])
    if uploaded_file is not None:
        text_input = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        st.success(f"✅ Loaded file: `{uploaded_file.name}`")
    else:
        text_input = st.text_area("Or paste your text here:", height=300, placeholder="Type or paste your article here...")

    # Show counts
    if text_input.strip():
        words = len(text_input.split())
        chars = len(text_input)
        st.markdown(f"📊 **Word Count**: {words} | **Character Count**: {chars}")

# State Init
if "summary_done" not in st.session_state:
    st.session_state.summary_done = False
if "summary_text_bart" not in st.session_state:
    st.session_state.summary_text_bart = ""
if "summary_text_textrank" not in st.session_state:
    st.session_state.summary_text_textrank = ""

# Generate Button
if st.button("✨ Generate Summary"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        with st.spinner("🔄 Summarizing..."):
            try:
                if method == "🧠 BART Transformer":
                    summary = summarize_with_bart(text_input, max_len=max_len, min_len=min_len)
                    st.session_state.summary_text_bart = summary
                    st.session_state.summary_done = True

                elif method == "⚡ TextRank (Fast)":
                    summary = summarize_with_textrank(text_input, sentence_count)
                    st.session_state.summary_text_textrank = summary
                    st.session_state.summary_done = True

                elif method == "🆚 Compare Both":
                    st.session_state.summary_text_bart = summarize_with_bart(text_input, max_len=max_len, min_len=min_len)
                    st.session_state.summary_text_textrank = summarize_with_textrank(text_input, sentence_count)
                    st.session_state.summary_done = True

            except Exception as e:
                st.error(f"🚨 Error: {e}")
                st.session_state.summary_done = False

# --- Output Tab ---
with tab2:
    if st.session_state.summary_done:
        if method in ["🧠 BART Transformer", "🆚 Compare Both"]:
            st.subheader("🧠 BART Summary")
            bart_summary = st.session_state.summary_text_bart
            st.text_area("BART Output", bart_summary, height=200, key="bart_output")
            st.markdown(get_download_link(bart_summary, "bart_summary.txt"), unsafe_allow_html=True)
            st.markdown(f"📊 **Words**: {len(bart_summary.split())} | **Chars**: {len(bart_summary)}")

            try:
                tts = gTTS(text=bart_summary)
                tts.save("bart_summary.mp3")
                with open("bart_summary.mp3", "rb") as audio_file:
                    audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
                b64 = base64.b64encode(audio_bytes).decode()
                st.markdown(f'<a href="data:audio/mp3;base64,{b64}" download="bart_summary.mp3">📥 Download MP3</a>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"🎧 Could not create audio: {e}")

        if method in ["⚡ TextRank (Fast)", "🆚 Compare Both"]:
            st.subheader("⚡ TextRank Summary")
            tr_summary = st.session_state.summary_text_textrank
            st.text_area("TextRank Output", tr_summary, height=200, key="textrank_output")
            st.markdown(get_download_link(tr_summary, "textrank_summary.txt"), unsafe_allow_html=True)
            st.markdown(f"📊 **Words**: {len(tr_summary.split())} | **Chars**: {len(tr_summary)}")

        try:
            tts_tr = gTTS(text=tr_summary)
            tts_tr.save("textrank_summary.mp3")
            with open("textrank_summary.mp3", "rb") as audio_file:
                audio_bytes_tr = audio_file.read()
            st.audio(audio_bytes_tr, format="audio/mp3")
            b64_tr = base64.b64encode(audio_bytes_tr).decode()
            st.markdown(f'<a href="data:audio/mp3;base64,{b64_tr}" download="textrank_summary.mp3">📥 Download MP3</a>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"🎧 Could not create TextRank audio: {e}")
