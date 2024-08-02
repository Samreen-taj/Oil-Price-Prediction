import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
  
  
#load the data
data_cleaned=pd.read_csv('Cleaned_data.csv')

#define scalar
scaler= MinMaxScaler(feature_range=(0,1))

#preprocess the data
scaled_data=scaler.fit_transform(data_cleaned[['Price']])

#Split the data into training and testing sets
train_size=int(len(scaled_data)*0.8)
train_data=scaled_data[:train_size]
test_data=scaled_data[train_size:]

#prepare data for LSTM
def create_dataset(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:i+time_steps])
        y.append(dataset[i+time_steps])
    return np.array(X), np.array(y)

time_steps = 10  # Number of time steps to consider
X_train, y_train = create_dataset(train_data, time_steps)
X_test, y_test = create_dataset(test_data, time_steps)

# Load the saved LSTM model
model_path = "lstm_model.keras"  # Replace with the path to your saved model
model = load_model(model_path)

# Function to predict future prices
def predict_future_prices(current_price, num_days):
    # Scale the current price
    scaled_current_price = scaler.transform([[current_price]])

    # Create input sequence for prediction
    input_sequence = scaled_current_price.reshape(1, -1)
    input_sequence = np.repeat(input_sequence, time_steps, axis=1)

    # Predict future prices
    predicted_prices = []
    for _ in range(num_days):
        # Make prediction based on input sequence
        prediction = model.predict(input_sequence)
        # Inverse scale the prediction
        prediction = scaler.inverse_transform(prediction)
        # Append the predicted price to the list
        predicted_prices.append(prediction[0][0])
        # Update input sequence for the next prediction
        input_sequence = np.roll(input_sequence, -1)
        input_sequence[0][-1] = prediction

        return predicted_prices
    

    #Streamlit app
    st.titlle('Oil Price Prediction')

    #Input current oil price
    current_price=st.number_input('Enter the current oil price:')

    #Input number of days to predict
    num_days=st.number_input('Enter the number of days to predict:', min_value=1,max_value=365)

    #Predict future prices
    predicted_prices=predict_future_prices(current_price, num_days)

    ##Display the predicted prices
    st.subheader('Predicted Prices')
    for i, price in enumerate(predicted_prices, statrt=1):
        st.write(f'Day {i}: {price}')
