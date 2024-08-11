import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime
import plotly.graph_objects as go
import requests
from typing import Dict, Tuple, List

# Define the ticker symbol for Bitcoin
ticker = 'BTC-USD'

# Define the timezone for EST
est = pytz.timezone('America/New_York')

# Function to convert datetime to EST
def to_est(dt: pd.Timestamp) -> pd.Timestamp:
    if dt.tzinfo is None:
        return est.localize(dt)
    return dt.tz_convert(est)

# Fetch live data from Yahoo Finance
def fetch_data(ticker: str) -> pd.DataFrame:
    try:
        data = yf.download(ticker, period='1d', interval='30m')
        if data.index.tzinfo is None:
            data.index = data.index.tz_localize(pytz.utc).tz_convert(est)
        else:
            data.index = data.index.tz_convert(est)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Calculate technical indicators using the ta library
def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['STOCH'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
    data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
    data['ROC'] = ta.momentum.ROCIndicator(data['Close']).roc()
    data['WILLIAMSR'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
    return data

# Calculate Fibonacci retracement levels
def fibonacci_retracement(high: float, low: float) -> List[float]:
    diff = high - low
    levels = [high - diff * ratio for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]]
    return levels

# Detect Doji candlestick patterns
def detect_doji(data: pd.DataFrame) -> pd.DataFrame:
    threshold = 0.001
    data['Doji'] = abs(data['Close'] - data['Open']) / (data['High'] - data['Low']) < threshold
    return data

# Calculate support and resistance levels
def calculate_support_resistance(data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
    return data

# Generate buy/sell signals based on indicators and moving averages
def generate_signals(indicators: Dict[str, float], moving_averages: Dict[str, float]) -> Dict[str, str]:
    signals = {}
    signals['timestamp'] = to_est(data.index[-1]).strftime('%Y-%m-%d %I:%M:%S %p')

    def get_signal(value, buy_threshold, sell_threshold):
        if value < buy_threshold:
            return 'Buy'
        elif value > sell_threshold:
            return 'Sell'
        else:
            return 'Neutral'

    signals['RSI'] = get_signal(indicators['RSI'], 30, 70)
    signals['MACD'] = 'Buy' if indicators['MACD'] > 0 else 'Sell'
    signals['ADX'] = 'Buy' if indicators['ADX'] > 25 else 'Neutral'
    signals['CCI'] = get_signal(indicators['CCI'], -100, 100)
    signals['MA'] = 'Buy' if moving_averages['MA5'] > moving_averages['MA10'] else 'Sell'

    return signals

# Generate a perpetual options decision
def generate_perpetual_options_decision(indicators: Dict[str, float], moving_averages: Dict[str, float],
                                        fib_levels: List[float], current_price: float) -> str:
    decision = 'Neutral'
    resistance_levels = [fib_levels[3], fib_levels[4], high]

    if any([current_price >= level for level in resistance_levels]):
        decision = 'Go Short'
    else:
        buy_signals = [value for key, value in signals.items() if value == 'Buy']
        sell_signals = [value for key, value in signals.items() if value == 'Sell']

        if len(buy_signals) > len(sell_signals):
            decision = 'Go Long'
        elif len(sell_signals) > len(buy_signals):
            decision = 'Go Short'

    return decision

# Determine entry point
def determine_entry_point(signals: Dict[str, str]) -> str:
    if (signals['RSI'] == 'Buy' and signals['MACD'] == 'Buy' and signals['ADX'] == 'Buy'):
        return 'Buy Now'
    elif (signals['RSI'] == 'Sell' and signals['MACD'] == 'Sell' and signals['ADX'] == 'Sell'):
        return 'Sell Now'
    elif (signals['RSI'] == 'Buy' and signals['MACD'] == 'Buy'):
        return 'Potential Buy Opportunity'
    elif (signals['RSI'] == 'Sell' and signals['MACD'] == 'Sell'):
        return 'Potential Sell Opportunity'
    else:
        return 'Neutral or No Clear Entry Point'

# Fetch Fear and Greed Index
def fetch_fear_and_greed_index() -> Tuple[str, str]:
    url = "https://api.alternative.me/fng/"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        latest_data = data['data'][0]
        return latest_data['value'], latest_data['value_classification']
    except Exception as e:
        st.error(f"Error fetching Fear and Greed Index: {e}")
        return 'N/A', 'N/A'

def main():
    # Fetch and prepare data
    data = fetch_data(ticker)
    if data.empty:
        st.stop()

    # Calculate indicators and levels
    data = calculate_indicators(data)
    data = detect_doji(data)
    data = calculate_support_resistance(data)

    # Calculate Fibonacci retracement levels
    high = data['High'].max()
    low = data['Low'].min()
    fib_levels = fibonacci_retracement(high, low)

    # Calculate moving averages
    moving_averages = {
        'MA5': data['Close'].rolling(window=5).mean().iloc[-1],
        'MA10': data['Close'].rolling(window=10).mean().iloc[-1],
        'MA20': data['Close'].rolling(window=20).mean().iloc[-1],
        'MA50': data['Close'].rolling(window=50).mean().iloc[-1],
        'MA100': data['Close'].rolling(window=100).mean().iloc[-1],
        'MA200': data['Close'].rolling(window=200).mean().iloc[-1]
    }

    indicators = {
        'RSI': data['RSI'].iloc[-1],
        'MACD': data['MACD'].iloc[-1],
        'ADX': data['ADX'].iloc[-1],
        'CCI': data['CCI'].iloc[-1],
        'WILLIAMSR': data['WILLIAMSR'].iloc[-1]
    }

    # Generate signals
    signals = generate_signals(indicators, moving_averages)
    entry_point = determine_entry_point(signals)

    # Fetch Fear and Greed Index
    fear_and_greed_value, fear_and_greed_classification = fetch_fear_and_greed_index()

    # Generate perpetual options decision
    current_price = data['Close'].iloc[-1]
    perpetual_options_decision = generate_perpetual_options_decision(indicators, moving_averages, fib_levels, current_price)

    # Display results
    st.write(f"### Entry Point")
    st.write(entry_point)
    st.write(f"### Perpetual Options Decision")
    st.write(perpetual_options_decision)
    st.write(f"### Fear and Greed Index")
    st.write(f"Value: {fear_and_greed_value}")
    st.write(f"Classification: {fear_and_greed_classification}")

    # Plot candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
