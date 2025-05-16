import streamlit as st
import pandas as pd
import pdfplumber
import docx
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

# --- Summarization ---
def summarize_text(text, num_sentences=5):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_words = [w for w in words if w.isalnum() and w not in stop_words]

    freq = pd.Series(filtered_words).value_counts()
    sentence_scores = {}
    for sent in sentences:
        score = sum(freq.get(word.lower(), 0) for word in word_tokenize(sent))
        sentence_scores[sent] = score

    ranked = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary = " ".join(ranked[:num_sentences])
    return summary

# --- Text Extraction ---
def extract_text_from_file(file, filetype):
    if filetype == "txt":
        return file.read().decode("utf-8")
    elif filetype == "pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif filetype == "docx":
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif filetype == "csv":
        df = pd.read_csv(file)
        return df.to_string()
    else:
        return ""

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join(p.text for p in paragraphs)
    except Exception as e:
        return f"Error fetching URL: {e}"

def get_keywords(text, top_n=20):
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalnum()]
    stop_words = set(stopwords.words("english"))
    filtered = [w for w in words if w not in stop_words]
    freq = pd.Series(filtered).value_counts().reset_index()
    freq.columns = ["Keyword", "Count"]
    return freq.head(top_n)

# --- Word Cloud Plot ---
def plot_wordcloud(freq_df):
    word_freq = dict(zip(freq_df["Keyword"], freq_df["Count"]))
    wc = WordCloud(width=800, height=300, background_color="white").generate_from_frequencies(word_freq)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# --- Streamlit App ---
st.set_page_config("NLP Summarizer", layout="wide")
st.title("🧠 AI-Powered Text Summarizer & Keyword Extractor")

# Sidebar input
input_method = st.sidebar.radio("Choose Input Method:", ["📁 Upload File", "🔗 Enter URL"])
num_sentences = st.sidebar.slider("Summary Length", 1, 15, 5)
download_summary = st.sidebar.empty()

text = ""
filename = None

# File Upload
if input_method == "📁 Upload File":
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx", "csv"])
    if uploaded_file:
        filetype = uploaded_file.name.split(".")[-1].lower()
        filename = uploaded_file.name
        text = extract_text_from_file(uploaded_file, filetype)

# URL Input
elif input_method == "🔗 Enter URL":
    url = st.text_input("Enter a URL:")
    if url:
        text = extract_text_from_url(url)
        filename = "webpage"

# Preview text
if text:
    st.subheader("👀 Preview:")
    st.text(text[:1000] + ("..." if len(text) > 1000 else ""))

    # Generate summary
    if st.button("🔍 Generate Summary"):
        summary = summarize_text(text, num_sentences)
        st.subheader("✅ Summary")
        st.success(summary)

        # Download summary
        download_summary.download_button(
            label="📥 Download Summary",
            data=summary,
            file_name=f"{filename}_summary.txt",
            mime="text/plain"
        )

        # Keywords
        st.subheader("🔑 Keywords")
        keywords = get_keywords(text)
        col1, col2 = st.columns(2)
        with col1:
            st.table(keywords)
        with col2:
            plot_wordcloud(keywords)
