# 🤖 TEXT-SUMMARIZATION-TOOL

**COMPANY*: CODTECH IT SOLUTIONS

**NAME*: HIMANI JOSHI

**INTERN ID*: CTO4DN1683

**DOMAIN*: ARTIFICIAL INTELLIGENCE

**DURATION*: 4 WEEKS

**MENTOR*: NEELA SANTOSH

## 📝 Project Overview

It is an AI-powered web application that summarizes long-form text using:
- **Abstractive Summarization** with `sshleifer/distilbart-cnn-12-6` (transformers)
- **Extractive Summarization** with TextRank (sumy)

Built using **Python**, **Streamlit**, and **HuggingFace Transformers**, the app provides a fast, simple interface to generate and download clean summaries.


## 🚀 Features

🧠 BART Transformer (Abstractive)

⚡ TextRank (Extractive with Sumy + NLTK)

🖼️ Side-by-side comparison of both methods

📄 Upload .txt files or paste text directly

🗣️ Text-to-Speech (TTS) for generated summaries

✍️ Word & character count

📥 Summary download as .txt

🌓 Light/Dark mode toggle

🧪 Jupyter Notebook for development/testing



## 📁 Project Structure

 

 📄 **app.py** # Streamlit frontend
 
 📓 **Text_Summarizer.ipynb** # Jupyter Notebook (testing)
 
 📦 **requirements.txt** # Dependencies
 
 📘 **README.md**   # Project description 

 📜 **runtime.txt** # Optional: Python version for Streamlit Cloud



## 📱 How to Run the App

1️⃣ **Install Dependencies**
pip install -r requirements.txt

2️⃣ **Run the Streamlit app**
streamlit run app.py


## 📜 Sample Use Case
**Input**: A long article, blog post, research paper, or Wikipedia page

**Output**: A concise and meaningful summary using your preferred summarization technique — BART or TextRank.

## 🌐 Deploy on Streamlit Cloud

Push your code to GitHub

Go to streamlit.io/cloud

Connect your GitHub repo

Select app.py as the entry point

**🚀 Deploy!**


