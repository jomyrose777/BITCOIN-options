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
def to_est(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Convert a UTC timestamp to Eastern Standard Time (EST)."""
    return timestamp.tz_localize(pytz.utc).tz_convert(est)

# Fetch live data from Yahoo Finance
def fetch_data(ticker: str) -> pd.DataFrame:
    """Fetch historical data from Yahoo Finance."""
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
    """Calculate technical indicators."""
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
    """Calculate Fibonacci retracement levels."""
    diff = high - low
    levels = [high - diff * ratio for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]]
    return levels

# Detect Doji candlestick patterns
def detect_doji(data: pd.DataFrame) -> pd.DataFrame:
    """Detect Doji candlestick patterns."""
    threshold = 0.001
    data['Doji'] = abs(data['Close'] - data['Open']) / (data['High'] - data['Low']) < threshold
    return data

# Calculate support and resistance levels
def calculate_support_resistance(data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Calculate support and resistance levels."""
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
    return data

# Generate buy/sell signals based on indicators and moving averages
def generate_signals(indicators: Dict[str, float], moving_averages: Dict[str, float]) -> Dict[str, str]:
    """Generate trading signals based on indicators and moving averages."""
    def get_signal(value, buy_threshold, sell_threshold):
        if value < buy_threshold:
            return 'Buy'
        elif value > sell_threshold:
            return 'Sell'
        return 'Neutral'

    # Use pytz to get the current time in EST
    current_time_est = datetime.now(pytz.timezone('America/New_York'))

    signals = {
        'timestamp': current_time_est.strftime('%Y-%m-%d %I:%M:%S %p'),
        'RSI': get_signal(indicators['RSI'], 30, 70),
        'MACD': 'Buy' if indicators['MACD'] > 0 else 'Sell',
        'ADX': 'Buy' if indicators['ADX'] > 25 else 'Neutral',
        'CCI': get_signal(indicators['CCI'], -100, 100),
        'MA': 'Buy' if moving_averages['MA5'] > moving_averages['MA10'] else 'Sell'
    }
    return signals

def generate_perpetual_options_decision(signals: Dict[str, str], moving_averages: Dict[str, float],
                                        fib_levels: List[float], high: float, low: float, 
                                        current_price: float) -> str:
    """Generate a decision for perpetual options trading based on indicators, moving averages, and Fibonacci levels."""
    decision = 'Neutral'
    
    # Define resistance levels for decision making
    resistance_levels = [fib_levels[3], fib_levels[4], high]

    # Check if the current price is above any resistance levels
    if any(current_price >= level for level in resistance_levels):
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
    """Determine the entry point for trading based on signals."""
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
    """Fetch the Fear and Greed Index from an API."""
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
    while True:
        # Fetch and prepare data
        data = fetch_data(ticker)
        if data.empty:
            st.stop()

        # Calculate indicators and levels
        data = calculate_indicators(data)
        data = detect_doji(data)
        high = data['High'].max()
        low = data['Low'].min()
        fib_levels = fibonacci_retracement(high, low)
        data = calculate_support_resistance(data)

        # Calculate moving averages
        moving_averages = {
            'MA5': data['Close'].rolling(window=5).mean().iloc[-1],
            'MA10': data['Close'].rolling(window=10).mean().iloc[-1]
        }

        # Retrieve indicators
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

        # Generate trading signals
        signals = generate_signals(indicators, moving_averages)
        current_price = data['Close'].iloc[-1]
        
        # Generate perpetual options decision
        decision = generate_perpetual_options_decision(signals, moving_averages, fib_levels, high, low, current_price)
        entry_point = determine_entry_point(signals)
        
        # Display results
        st.write(f"### Entry Point")
        st.write(entry_point)
        st.write(f"### Perpetual Options Decision")
        st.write(decision)

        # Fetch and display Fear and Greed Index
        fear_and_greed_value, fear_and_greed_classification = fetch_fear_and_greed_index()
        st.write(f"### Fear and Greed Index")
        st.write(f"Value: {fear_and_greed_value}")
        st.write(f"Classification: {fear_and_greed_classification}")

        # Plot candlestick chart
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'])])
        fig.update_layout(title='Bitcoin Candlestick Chart')
        st.plotly_chart(fig)

        # Wait before refreshing
        st.time.sleep(60)

if __name__ == "__main__":
    main()
