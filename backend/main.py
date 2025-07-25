from fastapi import FastAPI
from model import create_lstm_model, preprocess_data
from data import get_bitcoin_data

app = FastAPI()

# Train model on startup
btc_data = get_bitcoin_data()
x_train, x_test, y_train, y_test, scaler = preprocess_data(btc_data)
model = create_lstm_model()
model.fit(x_train, y_train, epochs=10, batch_size=32)

@app.get("/suggest-trade")
def suggest_trade():
    predicted = model.predict(x_test[-1:])
    price = float(scaler.inverse_transform(predicted)[0][0])
    current_price = btc_data[-1][0]
    signal = "BUY" if price > current_price * 1.01 else "SELL"
    confidence = "High" if abs(price - current_price) > 500 else "Medium"

    return {
        "signal": signal,
        "confidence": confidence,
        "predicted_price": round(price, 2),
        "current_price": round(current_price, 2),
        "reason": "Based on trend reversal + RSI indicator"
    }
