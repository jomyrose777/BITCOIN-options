import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime
import plotly.graph_objects as go
from typing import Dict

# Define the ticker symbol for Bitcoin
ticker = 'BTC-USD'

# Define the timezone for EST
est = pytz.timezone('America/New_York')

# Function to convert datetime to EST
def to_est(dt):
    return dt.tz_convert(est) if dt.tzinfo else est.localize(dt)

# Function to calculate signal accuracy
def calculate_signal_accuracy(signals: Dict[str, str], actual: Dict[str, str] = {}) -> float:
    """Calculate accuracy of signals."""
    correct_signals = sum(1 for key in signals if signals[key] == actual.get(key, 'Neutral'))
    return correct_signals / len(signals) if signals else 0.0

# Fetch live data from Yahoo Finance
data = yf.download(ticker, period='1d', interval='1m')

# Check if data is empty
if data.empty:
    st.error("No data fetched from Yahoo Finance.")
else:
    # Convert index to EST if it's not already timezone-aware
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize(pytz.utc).tz_convert(est)
    else:
        data.index = data.index.tz_convert(est)

    # Calculate technical indicators using the ta library
    try:
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        data['MACD'] = ta.trend.MACD(data['Close']).macd()
        data['MACD_Signal'] = ta.trend.MACD(data['Close']).macd_signal()
        data['STOCH'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
        data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
        data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
        data['BULLBEAR'] = data['Close']  # Replace with actual sentiment if available
        data['UO'] = data['Close']  # Replace with actual UO if available
        data['ROC'] = ta.momentum.ROCIndicator(data['Close']).roc()
        data['WILLIAMSR'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")

    # Drop rows with NaN values
    data.dropna(inplace=True)

    # Check if data is still empty after dropping NaNs
    if data.empty:
        st.error("Data is empty after processing.")
    else:
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
                'BULLBEAR': data['BULLBEAR'].iloc[-1],
                'UO': data['UO'].iloc[-1],
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

        # Iron Condor Strategy
        def iron_condor_strategy(data):
            """Calculate Iron Condor strategy signals."""
            current_price = data['Close'].iloc[-1]
            # Define strike prices for Iron Condor
            put_strike_short = current_price - 5000  # Example strike prices
            put_strike_long = current_price - 6000
            call_strike_short = current_price + 5000
            call_strike_long = current_price + 6000

            # Entry signal
            entry_signal = 'Neutral'
            if put_strike_short < current_price < call_strike_short:
                entry_signal = 'Entry'

            # Stop-loss and Take-profit levels
            stop_loss = max(abs(current_price - put_strike_short), abs(call_strike_short - current_price))
            take_profit = min(abs(current_price - put_strike_long), abs(call_strike_long - current_price))

            return {
                'Entry Signal': entry_signal,
                'Stop-Loss': stop_loss,
                'Take-Profit': take_profit
            }

        # Execute Iron Condor Strategy
        iron_condor_signals = iron_condor_strategy(data)

        # Decision Logic for Going Long or Short
        def decision_logic(signals, iron_condor_signals):
            """Determine whether to go long or short based on signals."""
            decision = 'Neutral'
            if signals['RSI'] == 'Buy' and signals['MACD'] == 'Buy' and iron_condor_signals['Entry Signal'] == 'Entry':
                decision = 'Go Long'
            elif signals['RSI'] == 'Sell' and signals['MACD'] == 'Sell' and iron_condor_signals['Entry Signal'] == 'Entry':
                decision = 'Go Short'
            return decision

        trade_decision = decision_logic(signals, iron_condor_signals)

        # Display results in the Streamlit app
        st.subheader('Technical Indicators')
        for key, value in indicators.items():
            st.write(f"{key}: {value:.2f}")

        st.subheader('Moving Averages')
        for key, value in moving_averages.items():
            st.write(f"{key}: {value:.2f}")

        st.subheader('Trading Signals')
        for key, value in signals.items():
            st.write(f"{key}: {value}")

        st.subheader('Iron Condor Strategy Signals')
        st.write(f"Entry Signal: {iron_condor_signals['Entry Signal']}")
        st.write(f"Stop-Loss: {iron_condor_signals['Stop-Loss']:.2f}")
        st.write(f"Take-Profit: {iron_condor_signals['Take-Profit']:.2f}")

        st.subheader('Trade Decision')
        st.write(f"Trade Recommendation: {trade_decision}")

        # Assuming the Fear and Greed Index is fetched from an API or similar source
        try:
            fear_and_greed_value = 50  # Replace with actual value from API or data source
            fear_and_greed_classification = "Neutral"  # Replace with classification based on actual value
            st.subheader('Fear and Greed Index')
            st.write(f"Value: {fear_and_greed_value}")
            st.write(f"Classification: {fear_and_greed_classification}")
        except Exception as e:
            st.error(f"Error fetching Fear and Greed Index: {e}")

        # Calculate signal accuracy
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
        st.write(f"Accuracy Percentage: {accuracy_percentage:.2f}")

if __name__ == '__main__':
    main()
