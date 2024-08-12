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
import threading

def refresh_app():
    while True:
        st.experimental_rerun()
        time.sleep(30)

if st.session_state.get("refresh_started", False) is False:
    threading.Thread(target=refresh_app, daemon=True).start()
    st.session_state.refresh_started = True

ticker = 'BTC-USD'
est = pytz.timezone('America/New_York')

def to_est(dt):
    if dt.tzinfo is None:
        dt = est.localize(dt)
    return dt.astimezone(est)

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

def calculate_indicators(data):
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['MACD_Signal'] = ta.trend.MACD(data['Close']).macd_signal()
    data['STOCH'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
    data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
    data['ROC'] = ta.momentum.ROCIndicator(data['Close']).roc()
    data['WILLIAMSR'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
    data.dropna(inplace=True)
    return data

def calculate_support_resistance(data, window=5):
    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
    return data

def detect_doji(data, threshold=0.1):
    data['Doji'] = np.where(
        (data['Close'] - data['Open']).abs() / (data['High'] - data['Low']) < threshold,
        'Yes',
        'No'
    )
    return data

def calculate_atr(data, window=14):
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=window).average_true_range()
    return data

def calculate_atr_based_levels(data, entry_point, atr_multiplier=1.5):
    atr_value = data['ATR'].iloc[-1]
    take_profit_level = entry_point + (atr_value * atr_multiplier)
    stop_loss_level = entry_point - (atr_value * atr_multiplier)
    return take_profit_level, stop_loss_level

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

def generate_weighted_signals(indicators, moving_averages, data):
    weights = {
        'RSI': 0.2,
        'MACD': 0.3,
        'ADX': 0.2,
        'CCI': 0.2,
        'MA': 0.1
    }
    
    signals = generate_signals(indicators, moving_averages, data)
    
    weighted_score = sum([weights[key] if value == 'Buy' else -weights[key] for key, value in signals.items()])
    
    if weighted_score > 0:
        return 'Go Long'
    elif weighted_score < 0:
        return 'Go Short'
    else:
        return 'Neutral'

def is_trend_confirmed(moving_averages, signals):
    trend_up = moving_averages['MA5'] > moving_averages['MA20']
    trend_down = moving_averages['MA5'] < moving_averages['MA20']
    
    if signals == 'Go Long' and trend_up:
        return True
    elif signals == 'Go Short' and trend_down:
        return True
    else:
        return False

def calculate_position_size(account_balance, entry_price, stop_loss_level, risk_percentage=1):
    risk_amount = account_balance * (risk_percentage / 100)
    position_size = risk_amount / abs(entry_price - stop_loss_level)
    return position_size

def generate_signals(indicators, moving_averages, data):
    signals = {}
    last_timestamp = to_est(data.index[-1]).strftime('%Y-%m-%d %I:%M:%S %p')
    signals['timestamp'] = last_timestamp
    
    # RSI Signal
    if indicators['RSI'] < 30:
        signals['RSI'] = 'Buy'
    elif indicators['RSI'] > 70:
        signals['RSI'] = 'Sell'
    else:
        signals['RSI'] = 'Neutral'
    
    # MACD Signal
    signals['MACD'] = 'Buy' if indicators['MACD'] > 0 else 'Sell'
    
    # ADX Signal
    signals['ADX'] = 'Buy' if indicators['ADX'] > 25 else 'Neutral'
    
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

def generate_perpetual_options_decision(indicators, moving_averages, data, account_balance):
    signals = generate_weighted_signals(indicators, moving_averages, data)
    
    # Decision logic
    buy_signals = [value for key, value in signals.items() if value == 'Buy']
    sell_signals = [value for key, value in signals.items() if value == 'Sell']
    
    if len(buy_signals) > len(sell_signals):
        decision = 'Go Long'
        take_profit_pct = 0.02
        stop_loss_pct = 0.01
    elif len(sell_signals) > len(buy_signals):
        decision = 'Go Short'
        take_profit_pct = -0.02
        stop_loss_pct = 0.01
    else:
        decision = 'Neutral'
        take_profit_pct = 0
        stop_loss_pct = 0

    entry_point = data['Close'].iloc[-1]
    take_profit_level = entry_point * (1 + take_profit_pct)
    stop_loss_level = entry_point * (1 - stop_loss_pct)
    
    # Check if trend is confirmed
    if is_trend_confirmed(moving_averages, decision):
        position_size = calculate_position_size(account_balance, entry_point, stop_loss_level)
    else:
        position_size = 0
    
    return decision, entry_point, take_profit_level, stop_loss_level, position_size

def log_signals(signals, decision, entry_point, take_profit, stop_loss):
    log_file = 'signals_log.csv'
    try:
        logs = pd.read_csv(log_file)
    except FileNotFoundError:
        logs = pd.DataFrame(columns=['timestamp', 'RSI', 'MACD', 'ADX', 'CCI', 'MA', 'Entry Point', 'Take Profit', 'Stop Loss', 'Decision'])

    new_log = pd.DataFrame([{
        'timestamp': signals['timestamp'],
        'RSI': signals['RSI'],
        'MACD': signals['MACD'],
        'ADX': signals['ADX'],
        'CCI': signals['CCI'],
        'MA': signals['MA'],
        'Entry Point': entry_point,
        'Take Profit': take_profit,
        'Stop Loss': stop_loss,
        'Decision': decision
    }])
    
    logs = pd.concat([logs, new_log], ignore_index=True)
    logs.to_csv(log_file, index=False)

def fetch_fear_greed_index():
    try:
        response = requests.get('https://api.alternative.me/fng/?format=json')
        response.raise_for_status()
        data = response.json()
        return data['data'][0]['value'], data['data'][0]['value_classification']
    except Exception as e:
        st.error(f"Error fetching Fear and Greed Index: {e}")
        return None, None

def plot_chart(data):
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Candlestick'))

    fig.update_layout(title='BTC-USD Price Chart',
                      xaxis_title='Date',
                      yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)

def main():
    data = fetch_data(ticker)
    
    if data is None:
        st.stop()
    
    data = calculate_indicators(data)
    data = calculate_support_resistance(data)
    data = detect_doji(data)
    data = calculate_atr(data)

    indicators = technical_indicators_summary(data)
    moving_averages = moving_averages_summary(data)
    
    decision, entry_point, take_profit, stop_loss, position_size = generate_perpetual_options_decision(indicators, moving_averages, data, 10000)
    
    log_signals(generate_signals(indicators, moving_averages, data), decision, entry_point, take_profit, stop_loss)
    
    fear_greed_value, fear_greed_classification = fetch_fear_greed_index()

    st.write(f"Fear and Greed Index: {fear_greed_value} ({fear_greed_classification})")

    plot_chart(data)
    
    st.write(f"Decision: {decision}")
    st.write(f"Entry Point: ${entry_point:.2f}")
    st.write(f"Take Profit Level: ${take_profit:.2f}")
    st.write(f"Stop Loss Level: ${stop_loss:.2f}")
    st.write(f"Position Size: {position_size:.2f} units")

if __name__ == "__main__":
    main()
