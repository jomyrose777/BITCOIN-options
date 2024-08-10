import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime
import plotly.graph_objects as go
import requests
from sklearn.metrics import matthews_corrcoef

# Define the ticker symbol for Bitcoin 
ticker = 'BTC-USD'

# Define the timezone for EST
est = pytz.timezone('America/New_York')

# Function to convert datetime to EST
def to_est(dt):
    return dt.tz_convert(est) if dt.tzinfo else est.localize(dt)

# Fetch live data from Yahoo Finance
def fetch_data(ticker):
    try:
        data = yf.download(ticker, period='1d', interval='1m')
        if data.index.tzinfo is None:
            data.index = data.index.tz_localize(pytz.utc).tz_convert(est)
        else:
            data.index = data.index.tz_convert(est)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

data = fetch_data(ticker)

# Check if data is available
if data.empty:
    st.stop()

# Calculate technical indicators using the ta library
def calculate_indicators(data):
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['MACD_Signal'] = ta.trend.MACD(data['Close']).macd_signal()
    data['STOCH'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
    data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
    data['ROC'] = ta.momentum.ROCIndicator(data['Close']).roc()
    data['WILLIAMSR'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
    return data

data = calculate_indicators(data)

# Drop rows with NaN values
data.dropna(inplace=True)

# Calculate Fibonacci retracement levels
def fibonacci_retracement(high, low):
    diff = high - low
    levels = [high - diff * ratio for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]]
    return levels

high = data['High'].max()
low = data['Low'].min()
fib_levels = fibonacci_retracement(high, low)

# Detect Doji candlestick patterns
def detect_doji(data):
    threshold = 0.001
    data['Doji'] = abs(data['Close'] - data['Open']) / (data['High'] - data['Low']) < threshold
    return data

data = detect_doji(data)

# Calculate support and resistance levels
def calculate_support_resistance(data, window=5):
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
    return data

data = calculate_support_resistance(data)

# Plotting functions
def plot_support_resistance(data, fib_levels):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Support'], name='Support', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Resistance'], name='Resistance', line=dict(dash='dash')))
    for level in fib_levels:
        fig.add_trace(go.Scatter(x=data.index, y=[level] * len(data.index), name=f'Fib Level {level:.2f}', line=dict(dash='dash')))
    fig.update_layout(title='Support and Resistance Levels', xaxis_title='Time', yaxis_title='Price')
    return fig

st.plotly_chart(plot_support_resistance(data, fib_levels))

# Generate summary of technical indicators
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

indicators = technical_indicators_summary(data)

# Generate summary of moving averages
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

moving_averages = moving_averages_summary(data)

# Generate buy/sell signals based on indicators and moving averages
def generate_signals(indicators, moving_averages):
    signals = {}
    signals['timestamp'] = to_est(data.index[-1]).strftime('%Y-%m-%d %I:%M:%S %p')

    # RSI Signal
    signals['RSI'] = 'Buy' if indicators['RSI'] < 30 else 'Sell' if indicators['RSI'] > 70 else 'Neutral'

    # MACD Signal
    signals['MACD'] = 'Buy' if indicators['MACD'] > 0 else 'Sell'

    # ADX Signal
    signals['ADX'] = 'Buy' if indicators['ADX'] > 25 else 'Neutral'

    # CCI Signal
    signals['CCI'] = 'Buy' if indicators['CCI'] > 100 else 'Sell' if indicators['CCI'] < -100 else 'Neutral'

    # Moving Averages Signal
    signals['MA'] = 'Buy' if moving_averages['MA5'] > moving_averages['MA10'] else 'Sell'

    return signals

signals = generate_signals(indicators, moving_averages)

# Calculate signal accuracy using Matthews correlation coefficient
def calculate_signal_accuracy(logs, signals):
    if len(logs) == 0:
        return 'N/A'
    y_true = logs.iloc[-1][1:]  # Assuming logs contains columns for actual signals
    y_pred = pd.Series(signals).reindex(y_true.index, fill_value='Neutral')
    return matthews_corrcoef(y_true, y_pred)

# Log signals
log_file = 'signals_log.csv'
try:
    logs = pd.read_csv(log_file)
except FileNotFoundError:
    logs = pd.DataFrame(columns=['timestamp', 'RSI', 'MACD', 'ADX', 'CCI', 'MA'])

new_log = pd.DataFrame([signals])
logs = pd.concat([logs, new_log], ignore_index=True)
logs.to_csv(log_file, index=False)

# Fetch Fear and Greed Index
def fetch_fear_and_greed_index():
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

fear_and_greed_value, fear_and_greed_classification = fetch_fear_and_greed_index()

# Generate a perpetual options decision
def generate_perpetual_options_decision(indicators, moving_averages, fib_levels, current_price):
    decision = 'Neutral'
    resistance_levels = [fib_levels[3], fib_levels[4], high]
    
    # Decision based on indicators
    if (indicators['RSI'] == 'Buy' and 
        indicators['MACD'] == 'Buy' and 
        indicators['ADX'] == 'Buy'):
        decision = 'Buy Now'
    elif (indicators['RSI'] == 'Sell' and 
          indicators['MACD'] == 'Sell' and 
          indicators['ADX'] == 'Sell'):
        decision = 'Sell Now'
    elif (indicators['RSI'] == 'Buy' and 
          indicators['MACD'] == 'Buy'):
        decision = 'Potential Buy Opportunity'
    elif (indicators['RSI'] == 'Sell' and 
          indicators['MACD'] == 'Sell'):
        decision = 'Potential Sell Opportunity'
    else:
        decision = 'Neutral or No Clear Entry Point'
    
    return decision

# Determine entry point based on signals
def determine_entry_point(signals):
    if (signals['RSI'] == 'Buy' and 
        signals['MACD'] == 'Buy' and 
        signals['ADX'] == 'Buy'):
        return 'Buy Now'
    elif (signals['RSI'] == 'Sell' and 
          signals['MACD'] == 'Sell' and 
          signals['ADX'] == 'Sell'):
        return 'Sell Now'
    elif (signals['RSI'] == 'Buy' and 
          signals['MACD'] == 'Buy'):
        return 'Potential Buy Opportunity'
    elif (signals['RSI'] == 'Sell' and 
          signals['MACD'] == 'Sell'):
        return 'Potential Sell Opportunity'
    else:
        return 'Neutral or No Clear Entry Point'

# Get current price
current_price = data['Close'].iloc[-1]

# Make decisions
perpetual_decision = generate_perpetual_options_decision(indicators, moving_averages, fib_levels, current_price)
entry_point = determine_entry_point(signals)

# Display results on Streamlit
st.title('Trading Dashboard')

st.write(f"### Technical Indicators Summary")
st.write(indicators)

st.write(f"### Moving Averages Summary")
st.write(moving_averages)

st.write(f"### Current Price")
st.write(f"${current_price:.2f}")

st.write(f"### Perpetual Options Decision")
st.write(perpetual_decision)

st.write(f"### Entry Point")
st.write(entry_point)

st.write(f"### Fear and Greed Index")
st.write(f"Value: {fear_and_greed_value}")
st.write(f"Classification: {fear_and_greed_classification}")

st.write(f"### Signal Accuracy")
st.write(calculate_signal_accuracy(logs, signals))

st.write(f"### Signals Log")
st.write(logs.tail())
