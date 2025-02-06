import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit app
st.title("Hotel Rating Prediction App")

# Define the mappings for categorical features
price_range_map = {4: 4, 3: 3, 2: 2, 1: 1}  # Assuming 4, 3, 2, 1 represent the range directly
rating_color_map = {
    "Dark Green": 5,
    "Green": 4,
    "Yellow": 3,
    "Orange": 2,
    "White": 1,
    "Red": 0
}
rating_text_map = {
    "Excellent": 5,
    "Very Good": 4,
    "Good": 3,
    "Average": 2,
    "Poor": 1,
    "Not rated": 0
}

# User input
st.write("Enter input features:")

# User input for Price range (select from 4, 3, 2, 1)
price_range = st.selectbox("Price Range", options=[4, 3, 2, 1])

# User input for Rating color (choose one of the listed colors)
rating_color = st.selectbox("Rating Color", options=["Dark Green", "Green", "Yellow", "Orange", "White", "Red"])

# User input for Rating text (choose one of the listed texts)
rating_text = st.selectbox("Rating Text", options=["Excellent", "Very Good", "Good", "Average", "Poor", "Not rated"])

# User input for Votes (numeric input for votes)
votes = st.number_input("Votes", min_value=0)

# MinMax normalization for Votes
scaler = MinMaxScaler()
votes_normalized = scaler.fit_transform(np.array([[votes]]))  # Reshaping to 2D array for MinMaxScaler

# Convert the categorical values to numerical ones
price_range_value = price_range_map[price_range]
rating_color_value = rating_color_map[rating_color]
rating_text_value = rating_text_map[rating_text]
# votes_value = votes_normalized[0][0]  # Extract the normalized value
votes_value = votes_normalized

# Combine all inputs into a feature array
input_features = [price_range_value, rating_color_value, rating_text_value, votes_value]
# Prediction
if st.button("Predict"):
    # input_data = np.array(input_features).reshape(1, -1)  # Reshape for model input
    input_data = pd.DataFrame({
        "Price range": [price_range_value],
        "Rating color": [rating_color_value],
        "Rating text": [rating_text_value],
        "Votes": [votes_value]
    })
    prediction = model.predict(input_data)
    st.success(f"The predicted rating is: {prediction[0]:.1f}")

