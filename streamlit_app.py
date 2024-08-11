
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime
import plotly.graph_objects as go
import requests
from sklearn.metrics import f1_score
import time

# Define the ticker symbol for Bitcoin
ticker = 'BTC-USD'

# Define the timezone for EST
est = pytz.timezone('America/New_York')

# Function to convert datetime to EST
def to_est(dt):
    if dt.tzinfo is None:
        return est.localize(dt)
    return dt.tz_convert(est)

# Function to fetch data with retry mechanism
def fetch_data(ticker, retries=5, delay=5):
    for attempt in range(retries):
        try:
            data = yf.download(ticker, period='1d', interval='1m')
            if data.index.tzinfo is None:
                data.index = data.index.tz_localize(pytz.utc).tz_convert(est)
            else:
                data.index = data.index.tz_convert(est)
            return data
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                st.error(f"Error fetching data after {retries} attempts: {e}")
                return pd.DataFrame()  # Return empty DataFrame on failure

# Function to fetch Fear and Greed Index with retry mechanism
def fetch_fear_and_greed_index(retries=5, delay=5):
    url = "https://api.alternative.me/fng/"
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            latest_data = data['data'][0]
            return latest_data['value'], latest_data['value_classification']
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                st.error(f"Error fetching Fear and Greed Index after {retries} attempts: {e}")
                return 'N/A', 'N/A'

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
    return data

# Function to calculate Fibonacci retracement levels
def fibonacci_retracement(high, low):
    diff = high - low
    levels = [high - diff * ratio for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]]
    return levels

# Function to detect Doji candlestick patterns
def detect_doji(data):
    threshold = 0.001
    data['Doji'] = abs(data['Close'] - data['Open']) / (data['High'] - data['Low']) < threshold
    return data

# Function to calculate support and resistance levels
def calculate_support_resistance(data, window=5):
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
    return data

# Function to generate buy/sell signals based on indicators and moving averages
def generate_signals(indicators, moving_averages):
    signals = {}
    signals['timestamp'] = to_est(data.index[-1]).strftime('%Y-%m-%d %I:%M:%S %p')

    # RSI Signal
    if indicators['RSI'] < 30:
        signals['RSI'] = 'Buy'
    elif indicators['RSI'] > 70:
        signals['RSI'] = 'Sell'
    else:
        signals['RSI'] = 'Neutral'

    # MACD Signal
    if indicators['MACD'] > indicators['MACD_Signal']:
        signals['MACD'] = 'Buy'
    else:
        signals['MACD'] = 'Sell'

    # ADX Signal
    if indicators['ADX'] > 25:
        signals['ADX'] = 'Buy'
    else:
        signals['ADX'] = 'Neutral'

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

# Function to calculate signal accuracy
def calculate_signal_accuracy(logs, signals):
    if len(logs) == 0:
        return 'N/A'
    y_true = logs.iloc[-1][1:]  # Assuming logs contains columns for actual signals
    y_pred = pd.Series(signals).reindex(y_true.index, fill_value='Neutral')
    return f1_score(y_true, y_pred, average='weighted')

# Function to generate a perpetual options decision
def generate_perpetual_options_decision(signals, moving_averages, fib_levels, current_price):
    decision = 'Neutral'
    resistance_levels = [fib_levels[3], fib_levels[4], high]

    # Check if current price is near any resistance level
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

# Function to determine entry point
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

# Function to plot support and resistance levels
def plot_support_resistance(data, fib_levels):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Support'], name='Support', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Resistance'], name='Resistance', line=dict(dash='dash')))
    for level in fib_levels:
        fig.add_trace(go.Scatter(x=data.index, y=[level] * len(data.index), name=f'Fib Level {level:.2f}', line=dict(dash='dash')))
    fig.update_layout(title='Support and Resistance Levels', xaxis_title='Time', yaxis_title='Price')
    return fig

# Main function
def main():
    while True:
        data = fetch_data(ticker)
        if data.empty:
            st.stop()

        data = calculate_indicators(data)
        data = detect_doji(data)
        high = data['High'].max()
        low = data['Low'].min()
        fib_levels = fibonacci_retracement(high, low)
        data = calculate_support_resistance(data)

        moving_averages = {
            'MA5': data['Close'].rolling(window=5).mean().iloc[-1],
            'MA10': data['Close'].rolling(window=10).mean().iloc[-1]
        }

        indicators = {
            'RSI': data['RSI'].iloc[-1],
            'MACD': data['MACD'].iloc[-1],
            'MACD_Signal': data['MACD_Signal'].iloc[-1],
            'STOCH': data['STOCH'].iloc[-1],
            'ADX': data['ADX'].iloc[-1],
            'CCI': data['CCI'].iloc[-1],
            'ROC': data['ROC'].iloc[-1],
            'WILLIAMSR': data['WILLIAMSR'].iloc[-1]
        }

        signals = generate_signals(indicators, moving_averages)
        current_price = data['Close'].iloc[-1]
        decision = generate_perpetual_options_decision(signals, moving_averages, fib_levels, current_price)
        entry_point = determine_entry_point(signals)

        st.write(f"**Latest Data Timestamp:** {signals['timestamp']}")
        st.write(f"**Fear and Greed Index:** {fetch_fear_and_greed_index()[0]} ({fetch_fear_and_greed_index()[1]})")
        st.write(f"**Trading Decision:** {decision}")
        st.write(f"**Entry Point Suggestion:** {entry_point}")

        # Plotting
        fig = plot_support_resistance(data, fib_levels)
        st.plotly_chart(fig)

        # Update every 30 seconds
        time.sleep(30)

if __name__ == "__main__":
    main()
