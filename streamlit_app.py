import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta.momentum as momentum
import ta.trend as trend
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
    """Fetch the last 1 day of data from Yahoo Finance."""
    try:
        ticker_obj = yf.Ticker(ticker)
        data = yf.download(ticker, period='1d', interval='1m')
        if data.empty:
            st.error("No data retrieved from Yahoo Finance.")
            return pd.DataFrame()
        
        data.reset_index(inplace=True)  # Reset index
        data['Date'] = pd.to_datetime(data['Date'])  # Convert to datetime
        data.set_index('Date', inplace=True)  # Set index
        data.index = data.index.tz_localize('UTC')  # Set timezone to UTC
        data.index = data.index.tz_convert(est)  # Convert to EST
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")  # Print error message
        return pd.DataFrame()

# Calculate technical indicators using the ta library
def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
    if len(data) < 14:
        st.error("Not enough data to calculate indicators. Please check the data length.")
        return data
    
    data = data.dropna()

    try:
        data['RSI'] = momentum.RSIIndicator(data['Close'], window=14).rsi()
        macd = trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['STOCH'] = momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
        data['ADX'] = trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
        data['CCI'] = trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
        data['ROC'] = momentum.ROCIndicator(data['Close']).roc()
        data['WILLIAMSR'] = momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
    
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

# Generate trading signals based on indicators and moving averages
def generate_signals(indicators: Dict[str, float], moving_averages: Dict[str, float] = {}, rsi_buy_threshold: float = 30, rsi_sell_threshold: float = 70, cci_buy_threshold: float = -100, cci_sell_threshold: float = 100) -> Dict[str, str]:
    """Generate trading signals based on indicators and moving averages."""
    def get_signal(value, buy_threshold, sell_threshold):
        if value < buy_threshold:
            return 'Buy'
        elif value > sell_threshold:
            return 'Sell'
        return 'Neutral'

    current_time_est = datetime.now(pytz.timezone('America/New_York'))
    signals = {
        'timestamp': current_time_est.strftime('%Y-%m-%d %I:%M:%S %p'),
    }

    signals['RSI'] = get_signal(indicators.get('RSI', np.nan), rsi_buy_threshold, rsi_sell_threshold)
    signals['MACD'] = 'Buy' if indicators.get('MACD', 0) > indicators.get('MACD_Signal', 0) else 'Sell'
    signals['ADX'] = 'Buy' if indicators.get('ADX', 0) > 25 else 'Neutral'
    signals['CCI'] = get_signal(indicators.get('CCI', np.nan), cci_buy_threshold, cci_sell_threshold)
    signals['MA'] = 'Buy' if moving_averages.get('MA5', 0) > moving_averages.get('MA10', 0) else 'Sell'

    return signals

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

# Calculate signal accuracy
def calculate_signal_accuracy(signals: Dict[str, str], actual: Dict[str, str] = {}) -> float:
    """Calculate accuracy of signals."""
    correct_signals = sum(1 for key in signals if signals[key] == actual.get(key, 'Neutral'))
    return correct_signals / len(signals) if signals else 0.0

# Suggest trade action based on signals and latest price
def suggest_trade_action(data: pd.DataFrame, indicators: Dict[str, float], stop_loss_percent: float = 0.02, take_profit_percent: float = 0.02) -> Dict[str, float]:
    """Suggest trade action based on indicators and latest price."""
    if data.empty or 'Close' not in data.columns:
        st.error("No data available for trade action suggestion.")
        return {}

    latest_price = data['Close'].iloc[-1]
    stop_loss = latest_price * (1 - stop_loss_percent)  # Stop-loss 2% below the current price
    take_profit = latest_price * (1 + take_profit_percent)  # Take-profit 2% above the current price
    
    return {
        'entry': latest_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit
    }

# Main function to execute the trading strategy
def main():
    st.title('Bitcoin Trading Analysis')
    
    data = fetch_data(ticker)
    
    if data.empty:
        st.write("No data to analyze.")
        return

    indicators = calculate_indicators(data)
    moving_averages = {'MA5': data['Close'].rolling(window=5).mean().iloc[-1],
                       'MA10': data['Close'].rolling(window=10).mean().iloc[-1]}
    signals = generate_signals(indicators, moving_averages)

    st.write("Indicators:")
    st.write(indicators.tail())
    
    st.write("Signals:")
    st.write(signals)
    
    # Plotting the price and indicators
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
    fig.update_layout(title='BTC-USD Price Chart', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)
    
    # Fetch Fear and Greed Index
    fng_value, fng_classification = fetch_fear_and_greed_index()
    st.write(f"Fear and Greed Index: {fng_value} ({fng_classification})")
    
    # Suggest trade action
    trade_action = suggest_trade_action(data, indicators)
    st.write("Suggested Trade Action:")
    st.write(trade_action)
    
    # Display backtesting results and accuracy metrics
    actual_signals = {'RSI': 'Buy', 'MACD': 'Buy', 'ADX': 'Buy', 'CCI': 'Buy', 'MA': 'Buy'}
    accuracy = calculate_signal_accuracy(signals, actual_signals)
    st.write(f"Signal Accuracy: {accuracy * 100:.2f}%")
    
    # Store actual market outcomes
    actual_outcomes = {'RSI': 'Buy', 'MACD': 'Buy', 'ADX': 'Buy', 'CCI': 'Buy', 'MA': 'Buy'}
    
    # Compare signals with actual outcomes
    win_loss_ratio = 0
    for signal in signals:
        if signals[signal] == actual_outcomes.get(signal):
            win_loss_ratio += 1
        else:
            win_loss_ratio -= 1
    
    # Update accuracy percentage
    accuracy_percentage = (win_loss_ratio / len(signals)) * 100
    
    # Display performance metrics
    st.write(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
    st.write(f"Accuracy Percentage: {accuracy_percentage:.2f}%")

if __name__ == '__main__':
    main()
