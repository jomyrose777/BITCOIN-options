import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime, timedelta
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

def fibonacci_retracement(high, low):
    diff = high - low
    levels = [high - diff * ratio for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]]
    return levels

def detect_doji(data):
    threshold = 0.001
    data['Doji'] = abs(data['Close'] - data['Open']) / (data['High'] - data['Low']) < threshold
    return data

def calculate_support_resistance(data, window=5):
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
    return data

def plot_support_resistance(data, fib_levels):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Support'], name='Support', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Resistance'], name='Resistance', line=dict(dash='dash')))
    for level in fib_levels:
        fig.add_trace(go.Scatter(x=data.index, y=[level] * len(data.index), name=f'Fib Level {level:.2f}', line=dict(dash='dash')))
    fig.update_layout(title='Support and Resistance Levels', xaxis_title='Time', yaxis_title='Price')
    return fig

def butterfly_spread_signal(data):
    # Placeholder logic for Butterfly Spread
    # You need actual options data to implement this
    current_price = data['Close'].iloc[-1]
    strike1 = 9800
    strike2 = 10000
    strike3 = 10200
    if current_price > strike1 and current_price < strike3:
        return 'Neutral'
    else:
        return 'Go Short' if current_price < strike1 else 'Go Long'

def iron_condor_signal(data):
    # Placeholder logic for Iron Condor
    # You need actual options data to implement this
    current_price = data['Close'].iloc[-1]
    lower_strike = 9500
    upper_strike = 10500
    if lower_strike <= current_price <= upper_strike:
        return 'Neutral'
    else:
        return 'Go Short' if current_price < lower_strike else 'Go Long'

def gmmma_signal(data):
    # Guppy Multiple Moving Averages (GMMMA)
    short_term_ma = data['Close'].rolling(window=3).mean().iloc[-1]
    long_term_ma = data['Close'].rolling(window=10).mean().iloc[-1]
    if short_term_ma > long_term_ma:
        return 'Buy'
    elif short_term_ma < long_term_ma:
        return 'Sell'
    else:
        return 'Neutral'

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
    if indicators['MACD'] > 0:
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

    # Integrate other strategies
    signals['Butterfly Spread'] = butterfly_spread_signal(data)
    signals['Iron Condor'] = iron_condor_signal(data)
    signals['GMMMA'] = gmmma_signal(data)

    return signals

def calculate_signal_accuracy(logs, signals):
    if len(logs) == 0:
        return 'N/A'
    y_true = logs.iloc[-1][1:] # Assuming logs contains columns for actual signals
    y_pred = pd.Series(signals).reindex(y_true.index, fill_value='Neutral')
    return f1_score(y_true, y_pred, average='weighted')

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

def generate_perpetual_options_decision(indicators, moving_averages, fib_levels, current_price):
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

# Main function to update every 30 seconds
def main():
    while True:
        data = fetch_data(ticker)
        if data.empty:
            st.stop()

        data = calculate_indicators(data)
        data.dropna(inplace=True)

        high = data['High'].max()
        low = data['Low'].min()
        fib_levels = fibonacci_retracement(high, low)
        data = detect_doji(data)
        data = calculate_support_resistance(data)

        indicators = data.iloc[-1]
        moving_averages = {
            'MA5': data['Close'].rolling(window=5).mean().iloc[-1],
            'MA10': data['Close'].rolling(window=10).mean().iloc[-1]
        }

        signals = generate_signals(indicators, moving_averages)
        accuracy = calculate_signal_accuracy(data, signals)

        current_price = data['Close'].iloc[-1]
        decision = generate_perpetual_options_decision(signals, moving_averages, fib_levels, current_price)
        entry_point = determine_entry_point(signals)

        # Display results
        st.write(f"**Timestamp:** {signals['timestamp']}")
        st.write(f"**Current Price:** {current_price:.2f}")
        st.write(f"**Decision:** {decision}")
        st.write(f"**Entry Point:** {entry_point}")
        st.write(f"**Signal Accuracy (F1 Score):** {accuracy:.2f}")

        fig = plot_support_resistance(data, fib_levels)
        st.plotly_chart(fig)

        # Fetch Fear and Greed Index
        fng_value, fng_classification = fetch_fear_and_greed_index()
        st.write(f"**Fear and Greed Index:** {fng_value} ({fng_classification})")

        # Update every 30 seconds
        time.sleep(30)

if __name__ == "__main__":
    main()
