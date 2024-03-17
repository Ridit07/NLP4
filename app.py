import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load the SVM model
import joblib

# Load the SVM model
@st.cache
def load_model():
    model = joblib.load('model.pkl')  # Update with the actual path to your SVM model file
    return model

# Load stopwords and initialize lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Vectorize input text
def vectorize_text(text, vectorizer):
    return vectorizer.transform([text])

# Main function to predict sentiment
def predict_sentiment(text, model, vectorizer):
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorize_text(preprocessed_text, vectorizer)
    prediction = model.predict(vectorized_text)
    return prediction[0]

# Streamlit app
def main():
    st.title('Restaurant Review Sentiment Analysis')
    st.write('This application predicts sentiment (positive or negative) of restaurant reviews.')

    # Text input for user to enter review
    review_input = st.text_area('Enter your restaurant review here:', height=200)

    # Load SVM model and vectorizer
    model = load_model()
    vectorizer = TfidfVectorizer(max_features=1000)

    # Predict sentiment when user submits review
    if st.button('Predict Sentiment'):
        if review_input:
            sentiment = predict_sentiment(review_input, model, vectorizer)
            if sentiment == 1:
                st.success('Positive sentiment detected!')
            else:
                st.error('Negative sentiment detected.')
        else:
            st.warning('Please enter a review.')

if __name__ == '__main__':
    main()
