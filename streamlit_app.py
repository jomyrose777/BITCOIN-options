
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pytz
from datetime import datetime

# Define your functions here
def to_est(dt):
    return dt.tz_convert(est) if dt.tzinfo else est.localize(dt)

def fetch_and_process_data():
    data = yf.download(ticker, period='1d', interval='1m')
    if data.empty:
        st.error("No data fetched from Yahoo Finance.")
        return None

    if data.index.tzinfo is None:
        data.index = data.index.tz_localize(pytz.utc).tz_convert(est)
    else:
        data.index = data.index.tz_convert(est)

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
        return None

    data.dropna(inplace=True)
    if data.empty:
        st.error("Data is empty after processing.")
        return None

    return data

def technical_indicators_summary(data):
    indicators = {
        'RSI': data['RSI'].iloc[-1],
        'MACD': data['MACD'].iloc[-1],
        'MACD_Signal': data['MACD_Signal'].iloc[-1],
        'STOCH': data['STOCH'].iloc[-1],
        'ADX': data['ADX'].iloc[-1],
        'CCI': data['CCI'].iloc[-1],
        'BULLBEAR': data['BULLBEAR'].iloc[-1],
        'UO': data['UO'].iloc[-1],
        'ROC': data['ROC'].iloc[-1],
        'WILLIAMSR': data['WILLIAMSR'].iloc[-1]
    }
    return indicators

def moving_averages_summary(data):
    moving_averages = {
        'MA5': data['Close'].rolling(window=5).mean().iloc[-1],
        'MA10': data['Close'].rolling(window=10).mean().iloc[-1],
        'MA20': data['Close'].rolling(window=20).mean().iloc[-1]
    }
    return moving_averages

def iron_condor_strategy(data):
    current_price = data['Close'].iloc[-1]
    put_strike_short = current_price - 5000
    put_strike_long = current_price - 6000
    call_strike_short = current_price + 5000
    call_strike_long = current_price + 6000

    entry_signal = 'Neutral'
    if put_strike_short < current_price < call_strike_short:
        entry_signal = 'Entry'

    stop_loss = max(abs(current_price - put_strike_short), abs(call_strike_short - current_price))
    take_profit = max(abs(current_price - put_strike_long), abs(call_strike_long - current_price))

    return {
        'Entry Signal': entry_signal,
        'Stop-Loss': stop_loss,
        'Take-Profit': take_profit
    }

def decision_logic(signals, iron_condor_signals):
    decision = 'Neutral'
    if signals['RSI'] == 'Buy' and signals['MACD'] == 'Buy' and iron_condor_signals['Entry Signal'] == 'Entry':
        decision = 'Go Long'
    elif signals['RSI'] == 'Sell' and signals['MACD'] == 'Sell' and iron_condor_signals['Entry Signal'] == 'Entry':
        decision = 'Go Short'
    return decision

def main():
    data = fetch_and_process_data()
    if data is None:
        return

    high = data['High'].max()
    low = data['Low'].min()
    fib_levels = fibonacci_retracement(high, low)
    data = detect_doji(data)
    data = calculate_support_resistance(data)

    indicators = technical_indicators_summary(data)
    moving_averages = moving_averages_summary(data)
    signals = generate_signals(indicators, moving_averages, data)
    iron_condor_signals = iron_condor_strategy(data)
    trade_decision = decision_logic(signals, iron_condor_signals)

    st.title('Bitcoin Technical Analysis and Signal Summary')

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

    try:
        fear_and_greed_value = 50  # Replace with actual value
        fear_and_greed_classification = "Neutral"  # Replace with actual classification
        st.subheader('Fear and Greed Index')
        st.write(f"Value: {fear_and_greed_value}")
        st.write(f"Classification: {fear_and_greed_classification}")
    except Exception as e:
        st.error(f"Error fetching Fear and Greed Index: {e}")

    actual_signals = {'RSI': 'Buy', 'MACD': 'Buy', 'ADX': 'Buy', 'CCI': 'Buy', 'MA': 'Buy'}
    accuracy = calculate_signal_accuracy(signals, actual_signals)
    st.write(f"Signal Accuracy: {accuracy * 100:.2f}%")
    
    actual_outcomes = {'RSI': 'Buy', 'MACD': 'Buy', 'ADX': 'Buy', 'CCI': 'Buy', 'MA': 'Buy'}
    win_loss_ratio = 0
    for signal in signals:
        if signals[signal] == actual_outcomes.get(signal):
            win_loss_ratio += 1
        else:
            win_loss_ratio -= 1
    
    accuracy_percentage = (win_loss_ratio / len(signals)) * 100
    st.write(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
    st.write(f"Accuracy Percentage: {accuracy_percentage:.2f}")

if __name__ == '__main__':
    main()
