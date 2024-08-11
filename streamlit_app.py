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

def main():
    st.title("Options Trading Strategies")

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
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()

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

    # Generate signals
    signals = generate_signals(indicators, {'MA5': data['MA5'].iloc[-1], 'MA10': data['MA10'].iloc[-1]})
    st.write("Trading Signals:", signals)

    # Input fields for Iron Condor
    st.subheader("Iron Condor Strategy")
    lower_put_strike = st.number_input("Lower Put Strike Price", min_value=0.0, value=40000.0)
    higher_put_strike = st.number_input("Higher Put Strike Price", min_value=0.0, value=39000.0)
    lower_call_strike = st.number_input("Lower Call Strike Price", min_value=0.0, value=41000.0)
    higher_call_strike = st.number_input("Higher Call Strike Price", min_value=0.0, value=42000.0)
    put1_premium = st.number_input("Put 1 Premium", min_value=0.0, value=100.0)
    put2_premium = st.number_input("Put 2 Premium", min_value=0.0, value=100.0)
    call1_premium = st.number_input("Call 1 Premium", min_value=0.0, value=100.0)
    call2_premium = st.number_input("Call 2 Premium", min_value=0.0, value=100.0)

    # Calculate Iron Condor P&L
    iron_condor_premium = iron_condor_pnl(data['Close'].iloc[-1], 
                                          (lower_put_strike, higher_put_strike, lower_call_strike, higher_call_strike), 
                                          (put1_premium, put2_premium, call1_premium, call2_premium))
    st.write(f"Iron Condor P&L: ${iron_condor_premium:.2f}")

    # Input fields for Butterfly Spread
    st.subheader("Butterfly Spread Strategy")
    lower_strike_butterfly = st.number_input("Lower Strike Price", min_value=0.0, value=40000.0)
    middle_strike = st.number_input("Middle Strike Price", min_value=0.0, value=40500.0)
    higher_strike_butterfly = st.number_input("Higher Strike Price", min_value=0.0, value=41000.0)
    lower_premium = st.number_input("Lower Strike Premium", min_value=0.0, value=100.0)
    middle_premium = st.number_input("Middle Strike Premium", min_value=0.0, value=200.0)
    higher_premium = st.number_input("Higher Strike Premium", min_value=0.0, value=100.0)

    # Calculate Butterfly Spread P&L
    butterfly_premium = butterfly_spread_pnl(data['Close'].iloc[-1], 
                                             (lower_strike_butterfly, middle_strike, higher_strike_butterfly), 
                                             (lower_premium, middle_premium, higher_premium))
    st.write(f"Butterfly Spread P&L: ${butterfly_premium:.2f}")

    # Gamma Scalping Input
    st.subheader("Gamma Scalping Adjustment")
    option_delta = st.number_input("Option Delta", value=0.0)
    underlying_position = st.number_input("Underlying Position", value=0.0)

    # Calculate Gamma Scalping Adjustment
    adjustment = gamma_scalping(data['Close'].iloc[-1], option_delta, underlying_position)
    st.write(f"Gamma Scalping Adjustment: {adjustment:.2f}")

    # Display latest news and Fear & Greed Index
    fear_and_greed_index, classification = fetch_fear_and_greed_index()
    st.write(f"Fear and Greed Index: {fear_and_greed_index} ({classification})")

    # Plot the data
    fig = go.Figure()

    # Check if columns exist before adding traces
    if 'MA5' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['MA5'], mode='lines', name='MA5'))

    if 'MA10' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['MA10'], mode='lines', name='MA10'))

    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlestick'))

    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
