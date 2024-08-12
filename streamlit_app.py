import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime
import plotly.graph_objects as go
import requests
import time

# Refresh the app every 60 seconds
def refresh_app():
    while True:
        st.experimental_rerun()
        time.sleep(60)

if st.session_state.get("refresh_started", False) is False:
    import threading
    threading.Thread(target=refresh_app, daemon=True).start()
    st.session_state.refresh_started = True

# Define the ticker symbol for Bitcoin
ticker = 'BTC-USD'

# Define the timezone for EST
est = pytz.timezone('America/New_York')

# Function to convert datetime to EST
def to_est(dt):
    if dt.tzinfo is None:
        dt = est.localize(dt)
    return dt.astimezone(est)

# Function to fetch live data from Yahoo Finance
@st.cache_data(ttl=30)
def fetch_data(ticker):
    try:
        data = yf.download(ticker, period='1d', interval='1m')
        if data.empty:
            st.error("No data fetched from Yahoo Finance.")
            return None

        if data.index.tzinfo is None:
            data.index = data.index.tz_localize(pytz.utc).tz_convert(est)
        else:
            data.index = data.index.tz_convert(est)

        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to calculate technical indicators
def calculate_indicators(data):
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['MACD_Signal'] = ta.trend.MACD(data['Close']).macd_signal()
    data['STOCH'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
    data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
    data['ROC'] = ta.momentum.ROCIndicator(data['Close']).roc()
    data['WILLIAMSR'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
    data.dropna(inplace=True)
    return data

# Function to calculate support and resistance levels
def calculate_support_resistance(data, window=5):
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
    return data

# Function to detect Doji candlestick patterns
def detect_doji(data, threshold=0.1):
    data['Doji'] = np.where(
        (data['Close'] - data['Open']).abs() / (data['High'] - data['Low']) < threshold,
        'Yes',
        'No'
    )
    return data

# Function to generate summary of technical indicators
def technical_indicators_summary(data):
    indicators = {
        'RSI': data['RSI'].iloc[-1],
        'MACD': data['MACD'].iloc[-1] - data['MACD_Signal'].iloc[-1],
        'STOCH': data['STOCH'].iloc[-1],
        'ADX': data['ADX'].iloc[-1],
        'CCI': data['CCI'].iloc[-1],
        'ROC': data['ROC'].iloc[-1],
        'WILLIAMSR': data['WILLIAMSR'].iloc[-1]
    }
    return indicators

# Function to generate summary of moving averages
def moving_averages_summary(data):
    ma = {
        'MA5': data['Close'].rolling(window=5).mean().iloc[-1],
        'MA10': data['Close'].rolling(window=10).mean().iloc[-1],
        'MA20': data['Close'].rolling(window=20).mean().iloc[-1],
        'MA50': data['Close'].rolling(window=50).mean().iloc[-1],
        'MA100': data['Close'].rolling(window=100).mean().iloc[-1],
        'MA200': data['Close'].rolling(window=200).mean().iloc[-1]
    }
    return ma

# Function to generate weighted signals
def generate_weighted_signals(indicators, moving_averages, data):
    weights = {
        'RSI': 0.2,
        'MACD': 0.3,
        'ADX': 0.2,
        'CCI': 0.2,
        'MA': 0.1
    }
    
    signals = generate_signals(indicators, moving_averages, data)
    
    # Ensure all keys from weights are present in signals
    missing_keys = [key for key in weights if key not in signals]
    if missing_keys:
        st.warning(f"Missing signals for: {', '.join(missing_keys)}")
        # Provide default values for missing keys if needed
        for key in missing_keys:
            signals[key] = 'Neutral'
    
    if not isinstance(signals, dict):
        st.error("Error: Signals is not a dictionary.")
        return {'Error': 'Signals is not a dictionary.'}
    
    weighted_score = sum([
        weights.get(key, 0) if value == 'Buy' else -weights.get(key, 0)
        for key, value in signals.items()
    ])
    
    st.write("Signals:", signals)  # Debugging line
    st.write("Weighted Score:", weighted_score)  # Debugging line
    
    return signals, weighted_score

# Function to generate signals based on indicators and moving averages
def generate_signals(indicators, moving_averages, data):
    signals = {}
    last_timestamp = to_est(data.index[-1]).strftime('%Y-%m-%d %I:%M:%S %p')
    signals['timestamp'] = last_timestamp
    
    # RSI Signal
    if indicators['RSI'] < 30:
        signals['RSI'] = 'Buy'
    elif indicators['RSI'] > 70:
        signals['RSI'] = 'Sell'
    else:
        signals['RSI'] = 'Neutral'
    
    # MACD Signal
    signals['MACD'] = 'Buy' if indicators['MACD'] > 0 else 'Sell'
    
    # ADX Signal
    signals['ADX'] = 'Buy' if indicators['ADX'] > 25 else 'Neutral'
    
    # CCI Signal
    if indicators['CCI'] > 100:
        signals['CCI'] = 'Buy'
    elif indicators['CCI'] < -100:
        signals['CCI'] = 'Sell'
    else:
        signals['CCI'] = 'Neutral'
    
    # Moving Averages Signal
    signals['MA'] = 'Buy' if moving_averages['MA5'] > moving_averages['MA10'] else 'Sell'
    
    return signals

# Function to log signals with additional details
def log_signals(signals, decision, entry_point, take_profit, stop_loss):
    log_file = 'signals_log.csv'
    try:
        logs = pd.read_csv(log_file)
    except FileNotFoundError:
        logs = pd.DataFrame(columns=['timestamp', 'RSI', 'MACD', 'ADX', 'CCI', 'MA', 'Entry Point', 'Take Profit', 'Stop Loss', 'Decision'])

    # Add new log
    new_log = pd.DataFrame([{
        'timestamp': signals['timestamp'],
        'RSI': signals['RSI'],
        'MACD': signals['MACD'],
        'ADX': signals['ADX'],
        'CCI': signals['CCI'],
        'MA': signals['MA'],
        'Entry Point': entry_point,
        'Take Profit': take_profit,
        'Stop Loss': stop_loss,
        'Decision': decision
    }])
    logs = pd.concat([new_log, logs], ignore_index=True)
    logs.to_csv(log_file, index=False)

# Function to calculate accuracy of signals
def calculate_accuracy():
    log_file = 'signals_log.csv'
    try:
        logs = pd.read_csv(log_file)
    except FileNotFoundError:
        return 0.0

    correct_signals = 0
    total_signals = len(logs)
    
    if total_signals == 0:
        return 0.0

    for index, row in logs.iterrows():
        # Here you should add your logic to check if the decision was correct based on historical data
        # For demonstration, let's assume a placeholder for accuracy calculation
        # Replace the following logic with real accuracy calculation
        if row['Decision'] == 'Go Long':  # Placeholder condition
            correct_signals += 1
    
    return correct_signals / total_signals * 100

# Function to fetch Fear and Greed Index
def fetch_fear_and_greed_index():
    url = 'https://api.alternative.me/fng/?limit=1'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        fear_and_greed_index = data['data'][0]['value']
        classification = data['data'][0]['value_classification']
        return fear_and_greed_index, classification
    except Exception as e:
        st.error(f"Error fetching Fear and Greed Index: {e}")
        return None, None

# Function to generate trading decision for perpetual options
def generate_perpetual_options_decision(indicators, moving_averages, data, account_balance):
    signals, weighted_score = generate_weighted_signals(indicators, moving_averages, data)
    
    if 'Error' in signals:
        st.error("Error generating weighted signals.")
        return 'Error', 0, 0, 0, 0
    
    buy_signals = [value for key, value in signals.items() if value == 'Buy']
    sell_signals = [value for key, value in signals.items() if value == 'Sell']
    
    if len(buy_signals) > len(sell_signals):
        decision = 'Go Long'
        take_profit_pct = 0.02
        stop_loss_pct = 0.01
    elif len(sell_signals) > len(buy_signals):
        decision = 'Go Short'
        take_profit_pct = -0.02
        stop_loss_pct = 0.01
    else:
        decision = 'Neutral'
        take_profit_pct = 0
        stop_loss_pct = 0
    
    entry_point = data['Close'].iloc[-1]
    take_profit_level = entry_point * (1 + take_profit_pct)
    stop_loss_level = entry_point * (1 - stop_loss_pct)  # Corrected calculation for stop loss

    # Log the signals with the calculated values
    log_signals(signals, decision, entry_point, take_profit_level, stop_loss_level)
    
    # Calculate accuracy of signals
    def calculate_accuracy():
    log_file = 'signals_log.csv'
    try:
        logs = pd.read_csv(log_file)
    except FileNotFoundError:
        return 0.0

    correct_signals = 0
    total_signals = len(logs)
    
    if total_signals == 0:
        return 0.0

    for index, row in logs.iterrows():
        # Check if the decision was correct based on historical data
        if row['Decision'] == 'Go Long' and row['Take Profit'] > row['Entry Point']:
            correct_signals += 1
        elif row['Decision'] == 'Go Short' and row['Stop Loss'] < row['Entry Point']:
            correct_signals += 1
    
    accuracy = correct_signals / total_signals * 100
    st.write("Accuracy Log:", accuracy)
    return accuracy


# Main app logic
st.title("Bitcoin Trading Signals")

data = fetch_data(ticker)
if data is not None:
    data = calculate_indicators(data)
    data = calculate_support_resistance(data)
    data = detect_doji(data)

    indicators = technical_indicators_summary(data)
    moving_averages = moving_averages_summary(data)
    
    st.write("Technical Indicators:")
    st.write(indicators)
    
    st.write("Moving Averages:")
    st.write(moving_averages)
    
    decision, entry_point, take_profit, stop_loss, _ = generate_perpetual_options_decision(indicators, moving_averages, data, account_balance=1000)
    
    accuracy = calculate_accuracy()  # Call the function here
    
    if decision == 'Error':
        st.error("Trading Decision could not be generated.")
    else:
        st.write("Trading Decision:")
        st.write(f"Decision: {decision}")
        st.write(f"Entry Point: {entry_point:.2f}")
        st.write(f"Take Profit Level: {take_profit:.2f}")
        st.write(f"Stop Loss Level: {stop_loss:.2f}")
        st.write(f"Accuracy: {accuracy:.2f}%")
