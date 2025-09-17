import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the trained model, vectorizer, and label encoder
try:
    model = joblib.load('logistic_regression_sms_spam_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("❌ Model or vectorizer files not found. Please ensure they are in the correct folder.")
    st.stop()

# Initialize objects
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_sms(sms_text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', sms_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    tokens = nltk.word_tokenize(cleaned_text)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    return ' '.join(tokens)

def predict_sms(sms_text):
    processed_sms = preprocess_sms(sms_text)
    if not processed_sms:
        return "Cannot classify empty or highly filtered message."
    vectorized_sms = tfidf_vectorizer.transform([processed_sms])
    prediction = model.predict(vectorized_sms)
    predicted_label = le.inverse_transform(prediction)[0]
    return predicted_label

# Streamlit UI Setup
st.set_page_config(page_title="📩 SMS Spam Classifier", layout="centered")

st.markdown("""
<style>
body {
    background-color: #f0f8ff;
}
h1 {
    color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

st.title("📩 SMS Spam Detection App")
st.markdown("Type a message below to find out if it's **Spam 🚫** or **Not Spam ✅**.")

sms_input = st.text_area("Enter SMS message:", height=150)

if st.button("Predict"):
    if sms_input.strip():
        result = predict_sms(sms_input)
        if result == 'spam':
            st.error("🚫 This message is Spam!")
        else:
            st.success("✅ This message is Not Spam!")
    else:
        st.warning("⚠️ Please enter a message to analyze.")

st.markdown("---")
st.info("ℹ️ This app uses NLP techniques like tokenization, stopword removal, lemmatization, and TF-IDF vectorization combined with Logistic Regression to classify SMS messages.")
