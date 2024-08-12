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

# Define the ticker symbol for Bitcoin
ticker = 'BTC-USD'

# Define the timezone for EST
est = pytz.timezone('America/New_York')

# Function to convert datetime to EST
def to_est(dt):
    return dt.tz_convert(est) if dt.tzinfo else est.localize(dt)

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

# Function to update the data every half minute
def update_data():
    while True:
        data = fetch_data(ticker)
        if data is None:
            st.stop()

        # Calculate technical indicators using the ta library
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        data['MACD'] = ta.trend.MACD(data['Close']).macd()
        data['MACD_Signal'] = ta.trend.MACD(data['Close']).macd_signal()
        data['STOCH'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
        data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
        data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
        data['ROC'] = ta.momentum.ROCIndicator(data['Close']).roc()
        data['WILLIAMSR'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()

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
            threshold = 0.001  # Define a threshold for identifying Doji
            data['Doji'] = abs(data['Close'] - data['Open']) / (data['High'] - data['Low']) < threshold
            return data

        data = detect_doji(data)

        # Calculate support and resistance levels
        def calculate_support_resistance(data, window=5):
            data['Support'] = data['Low'].rolling(window=window).min()
            data['Resistance'] = data['High'].rolling(window=window).max()
            return data

        data = calculate_support_resistance(data)

        # Add chart to display support and resistance levels
        st.title('Bitcoin Technical Analysis and Signal Summary')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
        fig.add_trace(go.Scatter(x=data.index, y=data['Support'], name='Support', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data['Resistance'], name='Resistance', line=dict(dash='dash')))
        fig.update_layout(title='Support and Resistance Levels', xaxis_title='Time', yaxis_title='Price')
        st.plotly_chart(fig)

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
        def generate_signals(indicators, moving_averages, data):
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
            
            return signals

        signals = generate_signals(indicators, moving_averages, data)

        # Calculate signal accuracy
        def calculate_signal_accuracy(logs, signals):
            if len(logs) == 0:
                return 'N/A'
            last_signal = logs.iloc[-1]
            accurate_signals = sum([last_signal[key] == signals[key] for key in signals if key in last_signal])
            total_signals = len(signals)
            accuracy = (accurate_signals / total_signals) * 100
            return f"{accuracy:.2f}%"

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
            response = requests.get(url)
            data = response.json()
            latest_data = data['data'][0]
            return latest_data['value'], latest_data['value_classification']

        fear_and_greed_value, fear_and_greed_classification = fetch_fear_and_greed_index()

        # Generate a perpetual options decision
        def generate_perpetual_options_decision(indicators, moving_averages):
            signals = generate_signals(indicators, moving_averages, data)
            
            # Decision logic
            buy_signals = [value for key, value in signals.items() if value == 'Buy']
            sell_signals = [value for key, value in signals.items() if value == 'Sell']
            
            if len(buy_signals) > len(sell_signals):
                decision = 'Go Long'
            elif len(sell_signals) > len(buy_signals):
                decision = 'Go Short'
            else:
                decision = 'Neutral'
            
            return decision, data['Close'].iloc[-1]

        decision, entry_point = generate_perpetual_options_decision(indicators, moving_averages)

        # Define take profit and stop loss percentages
        take_profit_pct = 0.03  # Example: 3%
        stop_loss_pct = 0.02    # Example: 2%

        # Calculate take profit and stop loss levels
        take_profit_level = entry_point * (1 + take_profit_pct)
        stop_loss_level = entry_point * (1 - stop_loss_pct)

        # Display take profit and stop loss levels
        st.write(f"**Entry Point:** ${entry_point:.2f}")
        st.write(f"**Take Profit Level:** ${take_profit_level:.2f}")
        st.write(f"**Stop Loss Level:** ${stop_loss_level:.2f}")

        # Display Fear and Greed Index
        st.write(f"**Fear and Greed Index:** {fear_and_greed_value} ({fear_and_greed_classification})")

        # Display signal accuracy
        accuracy = calculate_signal_accuracy(logs, signals)
        st.write(f"**Signal Accuracy:** {accuracy}")

        # Display decisions
        st.write(f"**Decision:** {decision}")

        # Display indicators and moving averages
        st.write(f"**Technical Indicators:** {indicators}")
        st.write(f"**Moving Averages:** {moving_averages}")

        # Display signals
        st.write(f"**Signals:** {signals}")

        time.sleep(30)  # Sleep for 30 seconds

# Run the update_data function
update_data()
