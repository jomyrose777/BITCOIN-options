import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime
import plotly.graph_objects as go
import requests
import logging
from sklearn.metrics import f1_score
import time
from typing import Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the ticker symbol for Bitcoin
ticker = 'BTC-USD'

# Define the timezone for EST
est = pytz.timezone('America/New_York')

# Function to convert datetime to EST
def to_est(dt: pd.Timestamp) -> pd.Timestamp:
    if dt.tzinfo is None:
        return est.localize(dt)
    return dt.tz_convert(est)

# Function to fetch data with retry mechanism
def fetch_data(ticker: str, retries: int = 5, delay: int = 5) -> pd.DataFrame:
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
                logging.error(f"Error fetching data: {e}")
                return pd.DataFrame()  # Return empty DataFrame on failure

# Function to fetch Fear and Greed Index with retry mechanism
def fetch_fear_and_greed_index(retries: int = 5, delay: int = 5) -> Tuple[str, str]:
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
                logging.error(f"Error fetching Fear and Greed Index: {e}")
                return 'N/A', 'N/A'

# Function to calculate technical indicators
def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    try:
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        data['MACD'] = ta.trend.MACD(data['Close']).macd()
        data['MACD_Signal'] = ta.trend.MACD(data['Close']).macd_signal()
        data['STOCH'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
        data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
        data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
        data['ROC'] = ta.momentum.ROCIndicator(data['Close']).roc()
        data['WILLIAMSR'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
        return data
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        logging.error(f"Error calculating indicators: {e}")
        return data

# Function to calculate Fibonacci retracement levels
def fibonacci_retracement(high: float, low: float) -> list:
    diff = high - low
    levels = [high - diff * ratio for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]]
    return levels

# Function to detect Doji candlestick patterns
def detect_doji(data: pd.DataFrame) -> pd.DataFrame:
    threshold = 0.001
    data['Doji'] = abs(data['Close'] - data['Open']) / (data['High'] - data['Low']) < threshold
    return data

# Function to calculate support and resistance levels
def calculate_support_resistance(data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
    return data

# Function to generate a single trading signal based on an indicator
def generate_signal(indicator_name: str, value: float, threshold: float) -> str:
    if value < threshold:
        return 'Buy'
    elif value > threshold:
        return 'Sell'
    else:
        return 'Neutral'

# Function to generate signals based on indicators
def generate_signals(indicators: Dict[str, float], moving_averages: Dict[str, float]) -> Dict[str, str]:
    signals = {}
    # Define thresholds for indicators
    thresholds = {
        'RSI': (30, 70),
        'MACD': 0,
        'ADX': 25,
        'CCI': (100, -100),
    }

    # Generate signals for each indicator
    signals['RSI'] = generate_signal('RSI', indicators['RSI'], thresholds['RSI'][0])
    signals['MACD'] = 'Buy' if indicators['MACD'] > indicators['MACD_Signal'] else 'Sell'
    signals['ADX'] = 'Buy' if indicators['ADX'] > thresholds['ADX'] else 'Neutral'
    signals['CCI'] = generate_signal('CCI', indicators['CCI'], thresholds['CCI'][0])
    signals['MA'] = 'Buy' if moving_averages['MA5'] > moving_averages['MA10'] else 'Sell'

    return signals

# Function to calculate signal accuracy
def calculate_signal_accuracy(logs: pd.DataFrame, signals: Dict[str, str]) -> float:
    if logs.empty:
        return float('NAN')
    y_true = logs.iloc[-1][1:]  # Assuming logs contain actual signals
    y_pred = pd.Series(signals).reindex(y_true.index, fill_value='Neutral')
    return f1_score(y_true, y_pred, average='weighted')

# Function to generate a perpetual options decision
def generate_perpetual_options_decision(signals: Dict[str, str], moving_averages: Dict[str, float], fib_levels: list, current_price: float) -> str:
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
def determine_entry_point(signals: Dict[str, str]) -> str:
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

# Function to plot candlestick chart with support and resistance levels
def plot_candlestick(data: pd.DataFrame, fib_levels: list) -> go.Figure:
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    ))

    # Plot Fibonacci levels
    for level in fib_levels:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=[level] * len(data),
            mode='lines',
            name=f'Fibonacci Level {level:.2f}',
            line=dict(dash='dash')
        ))

    fig.update_layout(
        title='Candlestick Chart with Fibonacci Levels',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )
    return fig

# Main function
def main():
    st.title("Bitcoin Trading Analysis")

    try:
        # Fetch data and indicators
        data = fetch_data(ticker)
        if data.empty:
            st.error("Failed to fetch data.")
            return

        data = calculate_indicators(data)
        current_price = data['Close'].iloc[-1]

        # Calculate moving averages
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA10'] = data['Close'].rolling(window=10).mean()
        moving_averages = {
            'MA5': data['MA5'].iloc[-1],
            'MA10': data['MA10'].iloc[-1]
        }

        # Detect Doji patterns
        data = detect_doji(data)

        # Calculate Fibonacci retracement levels
        high = data['High'].max()
        low = data['Low'].min()
        fib_levels = fibonacci_retracement(high, low)

        # Generate signals
        indicators = {
            'RSI': data['RSI'].iloc[-1],
            'MACD': data['MACD'].iloc[-1],
            'MACD_Signal': data['MACD_Signal'].iloc[-1],
            'ADX': data['ADX'].iloc[-1],
            'CCI': data['CCI'].iloc[-1],
            'ROC': data['ROC'].iloc[-1],
            'WILLIAMSR': data['WILLIAMSR'].iloc[-1]
        }
        signals = generate_signals(indicators, moving_averages)

        # Make trading decision
        decision = generate_perpetual_options_decision(signals, moving_averages, fib_levels, current_price)
        entry_point = determine_entry_point(signals)

        # Display results
        st.write(f"**Latest Data Timestamp:** {to_est(data.index[-1]).strftime('%Y-%m-%d %I:%M:%S %p')}")
        st.write(f"**Fear and Greed Index:** {fetch_fear_and_greed_index()[0]} ({fetch_fear_and_greed_index()[1]})")
        st.write(f"**Trading Decision:** {decision}")
        st.write(f"**Entry Point Suggestion:** {entry_point}")

        # Plotting
        fig = plot_candlestick(data, fib_levels)
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")

    # Update every 30 seconds
    time.sleep(30)

if __name__ == "__main__":
    main()
