import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Title
st.title("Professional Website URL Summarizer")

# 1. Loading the model correctly
@st.cache_resource
def load_model():
    # Adding device_map="auto" helps the library manage resources better
    return pipeline("summarization", model="facebook/bart-large-cnn")

try:
    summarizer = load_model()
except Exception as e:
    st.error(f"Model failed to load: {e}")

# 2. Extract text logic
def extract_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Look for article tag first, then fallback to all paragraphs
        article = soup.find("article")
        if article:
            paragraphs = article.find_all("p")
        else:
            paragraphs = soup.find_all("p")
            
        text = " ".join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        return None

# 3. UI logic
url_input = st.text_input("Enter Website URL:")

if url_input:
    with st.spinner("Processing..."):
        content = extract_text(url_input)
        if content:
            # BART has a limit of 1024 tokens
            summary = summarizer(content[:1024], max_length=130, min_length=30, do_sample=False)
            st.subheader("Summary")
            st.success(summary[0]['summary_text'])
        else:
            st.error("Could not retrieve text from the URL.")
