# app.py
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string

# ----------------------------
# Download NLTK resources once
# ----------------------------
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# ----------------------------
# Load the text file
# ----------------------------
# Make sure 'how_to_invest_money.html' or '.txt' is in the same folder as this script
with open('how_to_invest_money.html', 'r', encoding='utf-8') as f:
    data = f.read()


# ----------------------------
# Preprocess text
# ----------------------------
def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    words = [
        lemmatizer.lemmatize(word.lower())
        for word in words
        if word.lower() not in stop_words and word not in string.punctuation
    ]
    return " ".join(words)


# Split the text into sentences
sentences = sent_tokenize(data)

# Preprocess each sentence
preprocessed_sentences = [preprocess(sentence) for sentence in sentences]

# ----------------------------
# Build TF-IDF vectorizer
# ----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_sentences)


# ----------------------------
# Function to get most relevant sentence
# ----------------------------
def get_most_relevant_sentence(query):
    query_preprocessed = preprocess(query)
    query_vec = vectorizer.transform([query_preprocessed])

    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_vec, X).flatten()

    # Get the index of the highest similarity
    idx = similarities.argmax()
    return sentences[idx]  # Return original sentence


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
