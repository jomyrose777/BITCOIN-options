import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime
import pytz

# Define the ticker symbol for Bitcoin
ticker = 'BTC-USD'

# Define the timezone for EST
est = pytz.timezone('America/New_York')

def to_est(dt):
    if dt.tzinfo is None:
        dt = est.localize(dt)
        return dt.astimezone(est)
    else:
        return dt.astimezone(est)

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

# Function to calculate technical indicators
def calculate_indicators(data):
    try:
        data['SMA_20'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
        bb = ta.volatility.BollingerBands(data['Close'])
        data['BB_Middle'] = bb.bollinger_mavg()
        data['BB_Upper'] = bb.bollinger_hband()
        data['BB_Lower'] = bb.bollinger_lband()
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Hist'] = macd.macd_diff()
        data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        data['Fib_0.236'] = data['Close'].rolling(window=50).max() * 0.236
        data['Fib_0.382'] = data['Close'].rolling(window=50).max() * 0.382
        data['Fib_0.618'] = data['Close'].rolling(window=50).max() * 0.618
        data['SAR'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar()
        data['MFI'] = ta.volume.MFIIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=14).money_flow_index()
        stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
        data['Stoch_K'] = stoch.stoch()
        data['Stoch_D'] = stoch.stoch_signal()
        data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
        ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
        data['Ichimoku_A'] = ichimoku.ichimoku_a()
        data['Ichimoku_B'] = ichimoku.ichimoku_b()
        data['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        data['Ichimoku_Lead'] = ichimoku.ichimoku_a().shift(26)
        data['VWAP'] = ta.volume.VolumeWeightedAveragePrice(data['High'], data['Low'], data['Close'], data['Volume']).volume_weighted_average_price()
        data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume']).chaikin_money_flow()
        data.dropna(inplace=True)
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
    return data

# Function to calculate summary of indicators
def technical_indicators_summary(data):
    indicators = {}
    try:
        indicators = {
            'RSI': data['RSI'].iloc[-1] if not data['RSI'].empty else None,
            'MACD': data['MACD'].iloc[-1] - data['MACD_Signal'].iloc[-1] if not data['MACD'].empty else None,
            'BB_Upper': data['BB_Upper'].iloc[-1] if not data['BB_Upper'].empty else None,
            'BB_Lower': data['BB_Lower'].iloc[-1] if not data['BB_Lower'].empty else None,
            'SMA_20': data['SMA_20'].iloc[-1] if not data['SMA_20'].empty else None,
            'EMA_20': data['EMA_20'].iloc[-1] if not data['EMA_20'].empty else None,
            'OBV': data['OBV'].iloc[-1] if not data['OBV'].empty else None,
            'Fib_0.236': data['Fib_0.236'].iloc[-1] if not data['Fib_0.236'].empty else None,
            'Fib_0.382': data['Fib_0.382'].iloc[-1] if not data['Fib_0.382'].empty else None,
            'Fib_0.618': data['Fib_0.618'].iloc[-1] if not data['Fib_0.618'].empty else None,
            'MFI': data['MFI'].iloc[-1] if not data['MFI'].empty else None,
            'Stoch_K': data['Stoch_K'].iloc[-1] if not data['Stoch_K'].empty else None,
            'Stoch_D': data['Stoch_D'].iloc[-1] if not data['Stoch_D'].empty else None,
            'ATR': data['ATR'].iloc[-1] if not data['ATR'].empty else None,
            'Ichimoku_A': data['Ichimoku_A'].iloc[-1] if not data['Ichimoku_A'].empty else None,
            'Ichimoku_B': data['Ichimoku_B'].iloc[-1] if not data['Ichimoku_B'].empty else None,
            'Ichimoku_Base': data['Ichimoku_Base'].iloc[-1] if not data['Ichimoku_Base'].empty else None,
            'Ichimoku_Lead': data['Ichimoku_Lead'].iloc[-1] if not data['Ichimoku_Lead'].empty else None,
            'SAR': data['SAR'].iloc[-1] if not data['SAR'].empty else None,
            'VWAP': data['VWAP'].iloc[-1] if not data['VWAP'].empty else None,
            'CMF': data['CMF'].iloc[-1] if not data['CMF'].empty else None
        }
    except IndexError as e:
        st.error(f"Error accessing indicator data: {e}")
    return indicators

# Function to generate trading signals and calculate entry, take profit, and stop loss
def generate_trading_decision(indicators, data):
    signals = {}
    entry_point = None
    take_profit = None
    stop_loss = None

    if len(data) > 0:
        entry_point = data['Close'].iloc[-1]

    if entry_point is None:
        return 'No data available for trading decision', None, None

    # Example logic for signal generation
    if indicators.get('RSI') and indicators['RSI'] < 30:
        signals['RSI'] = 'Buy'
    elif indicators.get('RSI') and indicators['RSI'] > 70:
        signals['RSI'] = 'Sell'
    else:
        signals['RSI'] = 'Neutral'

    if indicators.get('MACD') and indicators.get('MACD_Signal'):
        macd_diff = indicators['MACD'] - indicators['MACD_Signal']
        if macd_diff > 0:
            signals['MACD'] = 'Buy'
        elif macd_diff < 0:
            signals['MACD'] = 'Sell'
        else:
            signals['MACD'] = 'Neutral'

    if indicators.get('BB_Upper') and indicators.get('BB_Lower'):
        if entry_point > indicators['BB_Upper']:
            signals['BB'] = 'Sell'
        elif entry_point < indicators['BB_Lower']:
            signals['BB'] = 'Buy'
        else:
            signals['BB'] = 'Neutral'

    if indicators.get('MFI') and indicators['MFI'] < 30:
        signals['MFI'] = 'Buy'
    elif indicators.get('MFI') and indicators['MFI'] > 70:
        signals['MFI'] = 'Sell'
    else:
        signals['MFI'] = 'Neutral'

    if len(set(signals.values())) > 1:
        final_signal = 'Go Long' if 'Buy' in signals.values() else 'Go Short' if 'Sell' in signals.values() else 'Hold'
        take_profit = entry_point * 1.02 if final_signal == 'Go Long' else entry_point * 0.98
        stop_loss = entry_point * 0.98 if final_signal == 'Go Long' else entry_point * 1.02
    else:
        final_signal = 'Hold'
        take_profit = None
        stop_loss = None

    return final_signal, take_profit, stop_loss

# Streamlit App
st.title('Bitcoin Technical Analysis')

data = fetch_data(ticker)

if data is not None and not data.empty:
    st.write(f"Data fetched for {ticker}:")
    st.write(data.tail())

    data = calculate_indicators(data)
    indicators = technical_indicators_summary(data)
    
    st.write("Technical Indicators Summary:")
    st.write(indicators)

    final_signal, take_profit, stop_loss = generate_trading_decision(indicators, data)
    
    st.write(f"Trading Signal: {final_signal}")
    st.write(f"Take Profit: {take_profit}")
    st.write(f"Stop Loss: {stop_loss}")

else:
    st.write("No data available to display.")
