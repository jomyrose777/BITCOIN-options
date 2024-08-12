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
    """Fetch current market price and last 1 day of minute-by-minute data."""
    try:
        ticker_obj = yf.Ticker(ticker)
        current_price = ticker_obj.info['regularMarketPrice']
        print(f"Current Price: {current_price}")
        
        data = yf.download(ticker, period='1d', interval='1m')
        print(f"Fetched data: {data.shape}")  # Print data shape
        data.reset_index(inplace=True)  # Reset index
        data['Date'] = pd.to_datetime(data['Date'])  # Convert to datetime
        data.set_index('Date', inplace=True)  # Set index
        data.index = data.index.tz_localize('UTC')  # Set timezone to UTC
        data.index = data.index.tz_convert(est)  # Convert to EST
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")  # Print error message
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

# Suggest trade action based on signals and latest price
def suggest_trade_action(data: pd.DataFrame, indicators: Dict[str, float], signals: Dict[str, str], stop_loss_percent: float = 0.02, take_profit_percent: float = 0.02) -> Dict[str, str]:
    """Suggest trade action based on indicators and latest price."""
    if data.empty or 'Close' not in data.columns:
        st.error("No data available for trade action suggestion.")
        return {}

    latest_price = data['Close'].iloc[-1]
    
    # Basic trade signal based on indicator signals
    if signals['RSI'] == 'Buy' and signals['MACD'] == 'Buy' and signals['CCI'] == 'Buy':
        trade_action = 'Go Long'
    elif signals['RSI'] == 'Sell' and signals['MACD'] == 'Sell' and signals['CCI'] == 'Sell':
        trade_action = 'Go Short'
    else:
        trade_action = 'Neutral'

    stop_loss = latest_price * (1 - stop_loss_percent)  # Stop-loss 2% below the current price
    take_profit = latest_price * (1 + take_profit_percent)  # Take-profit 2% above the current price
    
    return {
        'action': trade_action,
        'entry': latest_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit
    }

# Define Iron Condor strategy
def iron_condor_strategy(current_price: float, strikes: List[float], premiums: List[float]) -> Dict[str, float]:
    """Calculate the profitability of an Iron Condor options strategy."""
    if len(strikes) != 4 or len(premiums) != 4:
        raise ValueError("Four strikes and four premiums are required for Iron Condor strategy.")

    # Unpack strikes and premiums
    strike1, strike2, strike3, strike4 = strikes
    premium1, premium2, premium3, premium4 = premiums

    # Calculate Iron Condor components
    credit_received = premium1 + premium2 - (premium3 + premium4)
    max_profit = credit_received
    max_loss = (strike2 - strike1) - credit_received

    # Check if current price is within the range of the Iron Condor
    if strike1 < current_price < strike4:
        profit_loss = credit_received
    elif current_price <= strike1:
        profit_loss = max_loss
    elif current_price >= strike4:
        profit_loss = max_loss
    else:
        profit_loss = 0

    return {
        'credit_received': credit_received,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'current_profit_loss': profit_loss
    }

# Fetch Fear and Greed Index
def fetch_fear_and_greed_index() -> Tuple[int, str]:
    """Fetch the Fear and Greed Index value."""
    try:
        response = requests.get('https://api.alternative.me/fng/?limit=1')
        data = response.json()
        index_value = int(data['data'][0]['value'])
        index_classification = data['data'][0]['value_classification']
        return index_value, index_classification
    except Exception as e:
        st.error(f"Error fetching Fear and Greed Index: {e}")
        return 0, "Unknown"

# Calculate signal accuracy
def calculate_signal_accuracy(predicted_signals: Dict[str, str], actual_signals: Dict[str, str]) -> float:
    """Calculate the accuracy of trading signals."""
    correct_predictions = sum(1 for key in predicted_signals if predicted_signals[key] == actual_signals.get(key, 'Neutral'))
    return correct_predictions / len(predicted_signals) if predicted_signals else 0

# Main function to execute the trading strategy
def main():
    st.title('Bitcoin Trading Analysis and Options Strategies')
    
    data = fetch_data(ticker)
    
    if data.empty:
        st.write("No data to analyze.")
        return

    indicators = calculate_indicators(data)
    moving_averages = {'MA5': data['Close'].rolling(window=5).mean().iloc[-1],
                       'MA10': data['Close'].rolling(window=10).mean().iloc[-1]}
    signals = generate_signals(indicators, moving_averages)
    trade_action = suggest_trade_action(data, indicators, signals)
    
    # Display trading suggestion
    st.write("### Trading Suggestion")
    st.write(f"Action: {trade_action['action']}")
    st.write(f"Entry Price: ${trade_action['entry']:.2f}")
    st.write(f"Stop-Loss Price: ${trade_action['stop_loss']:.2f}")
    st.write(f"Take-Profit Price: ${trade_action['take_profit']:.2f}")

    # Display Fear and Greed Index
    fear_and_greed_index, index_classification = fetch_fear_and_greed_index()
    st.write("### Fear and Greed Index")
    st.write(f"Index Value: {fear_and_greed_index}")
    st.write(f"Classification: {index_classification}")
    
    # Options strategies: Iron Condor example
    current_price = data['Close'].iloc[-1]
    strikes = [current_price - 1000, current_price - 500, current_price + 500, current_price + 1000]  # Example strikes
    premiums = [50, 40, 30, 20]  # Example premiums
    iron_condor = iron_condor_strategy(current_price, strikes, premiums)
    
    st.write("### Iron Condor Strategy")
    st.write(f"Credit Received: ${iron_condor['credit_received']:.2f}")
    st.write(f"Maximum Profit: ${iron_condor['max_profit']:.2f}")
    st.write(f"Maximum Loss: ${iron_condor['max_loss']:.2f}")
    st.write(f"Current Profit/Loss: ${iron_condor['current_profit_loss']:.2f}")
    
    # Plot data and indicators
    st.write("### Bitcoin Price and Indicators")
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='MACD Signal'))
    fig.add_trace(go.Scatter(x=data.index, y=data['STOCH'], mode='lines', name='Stochastic Oscillator'))
    fig.add_trace(go.Scatter(x=data.index, y=data['ADX'], mode='lines', name='ADX'))
    fig.add_trace(go.Scatter(x=data.index, y=data['CCI'], mode='lines', name='CCI'))
    fig.add_trace(go.Scatter(x=data.index, y=data['ROC'], mode='lines', name='ROC'))
    fig.add_trace(go.Scatter(x=data.index, y=data['WILLIAMSR'], mode='lines', name='Williams %R'))
    
    st.plotly_chart(fig)
    
# Run the main function
if __name__ == "__main__":
    main()
