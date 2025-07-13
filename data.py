import yfinance as yf
import pandas as pd

def get_bitcoin_data():
    data = yf.download('BTC-USD', start='2020-01-01', end='2024-01-01')
    return data['Close'].values.reshape(-1, 1)
