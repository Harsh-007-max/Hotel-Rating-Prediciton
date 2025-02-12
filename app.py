import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

# Load the saved CPU model and transformers
model = joblib.load('model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer_cpu.joblib')
onehot_encoder = joblib.load('onehot_encoder_cpu.joblib')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define text preprocessing function (same as in training)
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)

# Function to join the text columns (must match training)
def join_text_columns(df):
    return df['Attractions'].astype(str) + " " + df['Description'].astype(str) + " " + df['HotelFacilities'].astype(str)

# Helper function to count words in a text string
def count_words(text):
    return len(text.split())

st.title("Hotel Rating Predictor")
st.write("Enter the hotel details to predict its rating:")

# Use a form to prevent processing on every change
with st.form(key="prediction_form"):
    attractions = st.text_area("Attractions", placeholder="Enter details about nearby attractions...")
    description = st.text_area("Description", placeholder="Enter a brief description of the hotel...")
    facilities = st.text_area("Hotel Facilities", placeholder="Enter the hotel facilities available...")
    county_name = st.text_input("County Name", placeholder="Enter the county name...")
    city_name = st.text_input("City Name", placeholder="Enter the city name...")
    
    submit_button = st.form_submit_button(label="Predict Hotel Rating")

if submit_button:
    # Check for empty fields
    missing_fields = []
    if not attractions.strip():
        missing_fields.append("Attractions")
    if not description.strip():
        missing_fields.append("Description")
    if not facilities.strip():
        missing_fields.append("Hotel Facilities")
    if not county_name.strip():
        missing_fields.append("County Name")
    if not city_name.strip():
        missing_fields.append("City Name")
    
    # Check for minimum word count (at least 50 words) in the text fields
    insufficient_fields = []
    if count_words(attractions) < 50:
        insufficient_fields.append("Attractions (at least 50 words required)")
    if count_words(description) < 50:
        insufficient_fields.append("Description (at least 50 words required)")
    if count_words(facilities) < 50:
        insufficient_fields.append("Hotel Facilities (at least 50 words required)")
    
    # Display errors if any fields are missing or insufficient
    if missing_fields or insufficient_fields:
        error_message = ""
        if missing_fields:
            error_message += "The following fields are empty: " + ", ".join(missing_fields) + ". "
        if insufficient_fields:
            error_message += "The following fields do not have at least 50 words: " + ", ".join(insufficient_fields) + "."
        st.error(error_message)
    else:
        # Create a DataFrame for input (ensure column names match training)
        input_df = pd.DataFrame({
            'Attractions': [attractions],
            'Description': [description],
            'HotelFacilities': [facilities],
            'countyName': [county_name],
            'cityName': [city_name]
        })
        
        # Combine text columns (as done during training)
        combined_text = join_text_columns(input_df)
        # Since combined_text is a Series with one element, extract the string
        processed_text = preprocess_text(combined_text.iloc[0])
        # Convert to a pandas Series (for TF-IDF transformation)
        processed_text_series = pd.Series([processed_text])
        # Transform the text using the loaded TF-IDF vectorizer
        x_text = tfidf_vectorizer.transform(processed_text_series)
        # Transform categorical features using the loaded OneHotEncoder
        x_cat = onehot_encoder.transform(input_df[['countyName', 'cityName']])
        x_cat_sparse = csr_matrix(x_cat)
        # Combine text and categorical features
        x_final = hstack([x_text, x_cat_sparse])
        
        # Predict the hotel rating using the loaded model
        prediction = model.predict(x_final)
        st.success(f"Predicted Hotel Rating: {prediction[0]}")

