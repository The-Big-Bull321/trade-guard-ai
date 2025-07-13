import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def create_lstm_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def preprocess_data(prices):
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)

    def create_sequences(data, seq_length=60):
        x = []
        y = []
        for i in range(len(data) - seq_length):
            x.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(x), np.array(y)

    x, y = create_sequences(scaled_prices)
    split = int(0.8 * len(x))
    return x[:split], x[split:], y[:split], y[split:], scaler
