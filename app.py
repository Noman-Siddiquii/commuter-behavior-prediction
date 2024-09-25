import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load data and model
df = pd.read_csv('Cleaned_data.csv')

# Load model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Fit transformers on the dataset
categorical_features = ['weather_description', 'time', 'Month', 'day']
numerical_features = ['air_pollution_index', 'wind_direction', 'temperature', 'rain_p_h']

# Fit LabelEncoders and MinMaxScaler on the training data
le_dict = {col: LabelEncoder().fit(df[col]) for col in categorical_features}
scaler = MinMaxScaler().fit(df[numerical_features])

def preprocess_data(new_data):
    """Preprocesses data by encoding categorical features and scaling numerical features.

    Args:
        new_data (dict): Dictionary containing the input data.

    Returns:
        Processed data as a NumPy array.
    """
    # Encode categorical features
    for col in categorical_features:
        new_data[col] = le_dict[col].transform([new_data[col]])[0]

    # Scale numerical features
    scaled_numerical = scaler.transform([[
        new_data[col] for col in numerical_features
    ]])

    # Combine processed data
    processed_data = np.concatenate([
        [new_data[col] for col in categorical_features],
        scaled_numerical[0]
    ])

    return processed_data.reshape(1, -1)

# Create Streamlit app
st.title('Commuter Traffic Prediction')

# Input fields
new_data = {}
new_data['air_pollution_index'] = st.number_input('Enter the Air Pollution Index')
new_data['wind_direction'] = st.number_input('Enter the Value of Direction of Wind')
new_data['temperature'] = st.number_input('Enter the value of Temperature')
new_data['rain_p_h'] = st.number_input('Enter pH level of Rain')
new_data['weather_description'] = st.selectbox('Select the Description of Weather', options=df['weather_description'].unique())
new_data['time'] = st.selectbox('Select the Time', options=df['time'].unique())
new_data['Month'] = st.selectbox('Select the Month', options=df['Month'].unique())
new_data['day'] = st.selectbox('Select the Day', options=df['day'].unique())

# Prediction button
if st.button('Predict'):
    # Preprocess data
    new_data_processed = preprocess_data(new_data)

    # Make prediction
    prediction = model.predict(new_data_processed)

    # Display prediction
    st.write(f'The Traffic Counts Prediction is: {prediction[0]}')