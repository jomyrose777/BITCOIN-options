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
import os

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
    
    logs = pd.concat([logs, new_log], ignore_index=True)
    logs.to_csv(log_file, index=False)
    
    # Optional: Log to console or other logging mechanism
    st.write("Signals logged:", new_log.to_dict(orient='records'))

# Function to generate perpetual options decision
def generate_perpetual_options_decision(indicators, moving_averages, data, account_balance):
    signals, weighted_score = generate_weighted_signals(indicators, moving_averages, data)
    
    # Decision-making logic
    if weighted_score > 0.5:
        decision = 'Go Long'
    elif weighted_score < -0.5:
        decision = 'Go Short'
    else:
        decision = 'Hold'

    # Set entry point, take profit, and stop loss based on decision
    entry_point = data['Close'].iloc[-1]
    take_profit = entry_point * 1.05 if decision == 'Go Long' else entry_point * 0.95
    stop_loss = entry_point * 0.95 if decision == 'Go Long' else entry_point * 1.05
    
    # Log the decision and details
    log_signals(signals, decision, entry_point, take_profit, stop_loss)
    
    return decision, entry_point, take_profit, stop_loss

# Streamlit app layout
st.title('Bitcoin Trading Signals')

data = fetch_data(ticker)

if data is not None:
    data = calculate_indicators(data)
    data = calculate_support_resistance(data)
    data = detect_doji(data)

    indicators = technical_indicators_summary(data)
    moving_averages = moving_averages_summary(data)
    
    st.subheader('Technical Indicators')
    st.write(indicators)
    
    st.subheader('Moving Averages')
    st.write(moving_averages)
    
    st.subheader('Support and Resistance')
    st.write(data[['Support', 'Resistance']].tail())
    
    st.subheader('Candlestick Patterns')
    st.write(data[['Doji']].tail())
    
    decision, entry_point, take_profit, stop_loss = generate_perpetual_options_decision(indicators, moving_averages, data, 10000)
    
    st.subheader('Trading Decision')
    st.write(f"Decision: {decision}")
    st.write(f"Entry Point: ${entry_point:.2f}")
    st.write(f"Take Profit: ${take_profit:.2f}")
    st.write(f"Stop Loss: ${stop_loss:.2f}")

    # Plot data
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='Candlestick'))
    fig.update_layout(title=f'BTC-USD Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

else:
    st.error("Failed to fetch data.")
