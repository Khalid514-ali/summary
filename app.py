import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

st.title("Professional Website URL Summarizer")

# Load model
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

# Extract main article text
def extract_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    article = soup.find("article")

    if article:
        paragraphs = article.find_all("p")
    else:
        paragraphs = soup.find_all("p")

    text = " ".join([p.get_text() for p in paragraphs])
    return text


# Chunk summarization
def summarize_long_text(text):
    max_chunk = 1000
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

    summaries = []

    for chunk in chunks:
        summary = summarizer(chunk, max_length=120, min_length=40, do_sample=False)
        summaries.append(summary[0]["summary_text"])

    final_summary = " ".join(summaries)
    return final_summary


url = st.text_input("Enter website URL")

if st.button("Summarize"):
    if url == "":
        st.warning("Enter a valid URL")
    else:
        with st.spinner("Extracting article..."):
            article = extract_text(url)

        if len(article) == 0:
            st.error("No content extracted")
        else:
            with st.spinner("Generating summary..."):
                final_summary = summarize_long_text(article)

            st.subheader("Summary")
            st.write(final_summary)
