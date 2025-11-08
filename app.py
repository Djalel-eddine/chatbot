import nltk
import os

# Define a folder for NLTK data inside your project
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)

# Download the required NLTK resources into that folder
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

# Add this folder to NLTK search path
nltk.data.path.append(nltk_data_path)

import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# ----------------------------
# Load text file
# ----------------------------
with open('how_to_invest_money.html', 'r', encoding='utf-8') as f:
    data = f.read()

# ----------------------------
# Preprocess text
# ----------------------------
def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Simple split instead of word_tokenize
    words = sentence.split()
    words = [
        lemmatizer.lemmatize(word.lower().strip(string.punctuation))
        for word in words
        if word.lower() not in stop_words
    ]
    return " ".join(words)

# ----------------------------
# Split text into sentences using regex (no punkt)
# ----------------------------
sentences = re.split(r'(?<=[.!?])\s+', data)
preprocessed_sentences = [preprocess(sentence) for sentence in sentences]

# ----------------------------
# TF-IDF vectorizer
# ----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_sentences)

# ----------------------------
# Function to get most relevant sentence
# ----------------------------
def get_most_relevant_sentence(query):
    query_preprocessed = preprocess(query)
    query_vec = vectorizer.transform([query_preprocessed])
    similarities = cosine_similarity(query_vec, X).flatten()
    idx = similarities.argmax()
    return sentences[idx]

# ----------------------------
# Streamlit app
# ----------------------------
def main():
    st.title("📚 Investment Chatbot")
    st.write("Ask me anything about the book 'How to Invest Money' by George Garr Henry.")

    user_question = st.text_input("You:")

    if st.button("Submit"):
        answer = get_most_relevant_sentence(user_question)
        st.write("💬 Chatbot:", answer)

if __name__ == "__main__":
    main()
