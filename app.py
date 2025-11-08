# app.py
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from bs4 import BeautifulSoup

# ----------------------------
# Download NLTK resources (always needed in Streamlit Cloud)
# ----------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ----------------------------
# Load and clean the text file
# ----------------------------
with open('how_to_invest_money.html', 'r', encoding='utf-8') as f:
    html_data = f.read()

# Use BeautifulSoup to extract text from HTML
soup = BeautifulSoup(html_data, 'html.parser')
data = soup.get_text(separator=' ')  # Convert HTML to plain text

# ----------------------------
# Preprocess function
# ----------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(sentence):
    words = word_tokenize(sentence)
    words = [
        lemmatizer.lemmatize(word.lower())
        for word in words
        if word.lower() not in stop_words and word not in string.punctuation
    ]
    return " ".join(words)

# ----------------------------
# Split into sentences and preprocess
# ----------------------------
sentences = sent_tokenize(data)
preprocessed_sentences = [preprocess(sentence) for sentence in sentences]

# ----------------------------
# TF-IDF vectorizer
# ----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_sentences)

# ----------------------------
# Function to find most relevant sentence
# ----------------------------
def get_most_relevant_sentence(query):
    query_preprocessed = preprocess(query)
    query_vec = vectorizer.transform([query_preprocessed])
    similarities = cosine_similarity(query_vec, X).flatten()
    idx = similarities.argmax()
    return sentences[idx]

# ----------------------------
# Streamlit App
# ----------------------------
def main():
    st.title("📚 Investment Chatbot")
    st.write("Ask me anything about the book 'How to Invest Money' by George Garr Henry.")

    user_question = st.text_input("You:")

    if st.button("Submit"):
        if user_question.strip():
            answer = get_most_relevant_sentence(user_question)
            st.write("💬 Chatbot:", answer)
        else:
            st.write("Please ask a question!")

if __name__ == "__main__":
    main()
