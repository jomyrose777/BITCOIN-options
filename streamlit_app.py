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
    # Ensure there are enough data points
    if len(data) < 14:
        st.error("Not enough data to calculate indicators. Please check the data length.")
        return data
    
    data = data.dropna()

    try:
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['STOCH'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
        data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
        data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
        data['ROC'] = ta.momentum.ROCIndicator(data['Close']).roc()
        data['WILLIAMSR'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
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

# Iron Condor P&L Calculation
def iron_condor_pnl(current_price: float, strikes: Tuple[float, float, float, float],
                    premiums: Tuple[float, float, float, float]) -> float:
    """Calculate Iron Condor profit and loss."""
    lower_put_strike, higher_put_strike, lower_call_strike, higher_call_strike = strikes
    put1_premium, put2_premium, call1_premium, call2_premium = premiums
    
    if current_price < lower_put_strike:
        pnl = put1_premium + put2_premium - (lower_put_strike - current_price)
    elif current_price > higher_call_strike:
        pnl = call1_premium + call2_premium - (current_price - higher_call_strike)
    elif current_price < higher_put_strike:
        pnl = put1_premium - (lower_put_strike - current_price)
    elif current_price > lower_call_strike:
        pnl = call1_premium - (current_price - lower_call_strike)
    else:
        pnl = put1_premium + call1_premium - (current_price - lower_put_strike) - (higher_call_strike - current_price)
    
    return pnl

# Gamma Scalping Adjustment
def gamma_scalping(current_price: float, option_delta: float, underlying_position: float) -> float:
    """Adjust the hedge to remain delta-neutral."""
    target_position = -option_delta
    adjustment = target_position - underlying_position
    return adjustment

# Butterfly Spread P&L Calculation
def butterfly_spread_pnl(current_price: float, strikes: Tuple[float, float, float],
                        premiums: Tuple[float, float, float]) -> float:
    """Calculate Butterfly Spread profit and loss."""
    lower_strike, middle_strike, higher_strike = strikes
    lower_premium, middle_premium, higher_premium = premiums
    
    if current_price < lower_strike or current_price > higher_strike:
        pnl = - (lower_premium + higher_premium - middle_premium)
    elif current_price < middle_strike:
        pnl = (current_price - lower_strike) - (middle_strike - lower_strike) - lower_premium
    else:
        pnl = (higher_strike - current_price) - (higher_strike - middle_strike) - higher_premium
    
    return pnl

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

# Calculate signal accuracy (dummy function for example)
def calculate_signal_accuracy(signals: Dict[str, str]) -> float:
    """Calculate accuracy of signals. Placeholder implementation."""
    # Dummy implementation: You should replace this with actual calculation or backtesting
    return 0.75  # Example accuracy

# Example function to suggest entry, stop-loss, and take-profit
def suggest_trade_action(data: pd.DataFrame, indicators: Dict[str, float]) -> Dict[str, float]:
    """Suggest trade action based on current indicators and data."""
    latest_price = data['Close'].iloc[-1]
    
    # Example values - you can calculate these based on data
    stop_loss = latest_price * 1.02  # Stop-loss 2% above the current price for short
    take_profit = latest_price * 0.95  # Take-profit 5% below the current price for short

    # Adjust based on your strategy
    if indicators['MACD'] == 'Sell':
        action = 'Go Short'
    elif indicators['MACD'] == 'Buy':
        action = 'Go Long'
        stop_loss = latest_price * 0.98  # Stop-loss 2% below the current price for long
        take_profit = latest_price * 1.05  # Take-profit 5% above the current price for long
    else:
        action = 'Hold'

    return {
        'Action': action,
        'Stop-Loss': stop_loss,
        'Take-Profit': take_profit
    }

# Main function to drive the Streamlit app
def main():
    st.title("Options Trading Analysis")

    # Fetch and display data
    data = fetch_data(ticker)
    if data.empty:
        return
    
    st.write("### Historical Data")
    st.write(data.tail())

    # Calculate indicators
    indicators = calculate_indicators(data)
    
    # Generate trading signals
    moving_averages = {'MA5': data['Close'].rolling(window=5).mean().iloc[-1],
                       'MA10': data['Close'].rolling(window=10).mean().iloc[-1]}
    signals = generate_signals(indicators, moving_averages)
    st.write("### Signals")
    st.write(signals)

    # Calculate and display P&L
    iron_condor_pnl_val = iron_condor_pnl(current_price=data['Close'].iloc[-1],
                                          strikes=(23000, 24000, 25000, 26000),
                                          premiums=(100, 100, 100, 100))
    gamma_scalping_adjustment = gamma_scalping(current_price=data['Close'].iloc[-1],
                                               option_delta=0.5,
                                               underlying_position=0)
    butterfly_spread_pnl_val = butterfly_spread_pnl(current_price=data['Close'].iloc[-1],
                                                    strikes=(23000, 24000, 25000),
                                                    premiums=(100, 150, 100))
    
    st.write("### P&L Calculations")
    st.write(f"Iron Condor P&L: ${iron_condor_pnl_val:.2f}")
    st.write(f"Gamma Scalping Adjustment: {gamma_scalping_adjustment:.2f}")
    st.write(f"Butterfly Spread P&L: ${butterfly_spread_pnl_val:.2f}")

    # Fetch Fear and Greed Index
    fear_and_greed_index, classification = fetch_fear_and_greed_index()
    st.write(f"### Fear and Greed Index: {fear_and_greed_index} ({classification})")

    # Calculate signal accuracy
    accuracy = calculate_signal_accuracy(signals)
    st.write(f"### Signal Accuracy: {accuracy:.2f}")

    # Suggest trade action
    trade_suggestion = suggest_trade_action(data, indicators)
    st.write("### Trade Suggestion")
    st.write(trade_suggestion)

if __name__ == "__main__":
    main()
