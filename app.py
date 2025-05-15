import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess(text):
    text = re.sub(r"http\S+|@\S+|#[A-Za-z0-9]+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return " ".join([word for word in text.split() if word not in stop_words])

# Load the model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis", page_icon="📝")
st.title("📝 Sentiment Analysis Web App")
st.subheader("Analyze the sentiment of your product reviews")

# User input
user_input = st.text_area("Enter your review below:")

# Prediction logic
if st.button("Analyze Sentiment"):
    if user_input.strip():
        cleaned_text = preprocess(user_input)
        vectorized_input = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_input)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.success(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("⚠️ Please enter a review to analyze.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Arkadeep using Streamlit & BERT+TF-IDF")
