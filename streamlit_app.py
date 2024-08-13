import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
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

#Function to fetch live data from Yahoo Finance
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
    # Moving Averages
    data['SMA_20'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
    data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(data['Close'])
    data['BB_Middle'] = bb.bollinger_mavg()
    data['BB_Upper'] = bb.bollinger_hband()
    data['BB_Lower'] = bb.bollinger_lband()

    # MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Hist'] = macd.macd_diff()
    
    # OBV
    data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    
    # RSI
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    
    # Fibonacci Retracement (dummy levels for illustration)
    data['Fib_0.236'] = data['Close'].rolling(window=50).max() * 0.236
    data['Fib_0.382'] = data['Close'].rolling(window=50).max() * 0.382
    data['Fib_0.618'] = data['Close'].rolling(window=50).max() * 0.618
    
    # Intraday Momentum Index (IMI)
    try:
    data['IMI'] = ta.momentum.IntradayMomentumIndex(data['Close'], data['High'], data['Low'], window=14).intraday_momentum_index()
    except Exception as e:
    st.error(f"Error calculating IMI: {e}")
    
    # Money Flow Index (MFI)
    data['MFI'] = ta.volume.MFIIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=14).money_flow_index()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()
    
    # Average True Range (ATR)
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
    
    # Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
    data['Ichimoku_A'] = ichimoku.ichimoku_a()
    data['Ichimoku_B'] = ichimoku.ichimoku_b()
    data['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
    data['Ichimoku_Lead'] = ichimoku.ichimoku_a().shift(26)
    
    # Parabolic SAR
    data['SAR'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close'], acceleration=0.02, max_acceleration=0.2).psar()
    
    # VWAP
    data['VWAP'] = ta.volume.VolumeWeightedAveragePrice(data['High'], data['Low'], data['Close'], data['Volume']).volume_weighted_average_price()
    
    # Chaikin Money Flow (CMF)
    data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=20).chaikin_money_flow()
    
    data.dropna(inplace=True)
    return data

# Function to calculate summary of indicators
def technical_indicators_summary(data):
    indicators = {
        'RSI': data['RSI'].iloc[-1],
        'MACD': data['MACD'].iloc[-1] - data['MACD_Signal'].iloc[-1],
        'BB_Upper': data['BB_Upper'].iloc[-1],
        'BB_Lower': data['BB_Lower'].iloc[-1],
        'SMA_20': data['SMA_20'].iloc[-1],
        'EMA_20': data['EMA_20'].iloc[-1],
        'OBV': data['OBV'].iloc[-1],
        'Fib_0.236': data['Fib_0.236'].iloc[-1],
        'Fib_0.382': data['Fib_0.382'].iloc[-1],
        'Fib_0.618': data['Fib_0.618'].iloc[-1],
        'IMI': data['IMI'].iloc[-1],
        'MFI': data['MFI'].iloc[-1],
        'Stoch_K': data['Stoch_K'].iloc[-1],
        'Stoch_D': data['Stoch_D'].iloc[-1],
        'ATR': data['ATR'].iloc[-1],
        'Ichimoku_A': data['Ichimoku_A'].iloc[-1],
        'Ichimoku_B': data['Ichimoku_B'].iloc[-1],
        'Ichimoku_Base': data['Ichimoku_Base'].iloc[-1],
        'Ichimoku_Lead': data['Ichimoku_Lead'].iloc[-1],
        'SAR': data['SAR'].iloc[-1],
        'VWAP': data['VWAP'].iloc[-1],
        'CMF': data['CMF'].iloc[-1]
    }
    return indicators

# Function to generate trading signals and calculate entry, take profit, and stop loss
def generate_trading_decision(indicators, data):
    signals = {}
    entry_point = data['Close'].iloc[-1]
    take_profit = None
    stop_loss = None

    # Example logic for signal generation
    if indicators['RSI'] < 30:
        signals['RSI'] = 'Buy'
    elif indicators['RSI'] > 70:
        signals['RSI'] = 'Sell'
    else:
        signals['RSI'] = 'Neutral'

    if indicators['MACD'] > 0:
        signals['MACD'] = 'Buy'
    elif indicators['MACD'] < 0:
        signals['MACD'] = 'Sell'
    else:
        signals['MACD'] = 'Neutral'

    # Example: Use Bollinger Bands to determine breakout signals
    if entry_point > indicators['BB_Upper']:
        signals['BB'] = 'Sell'
    elif entry_point < indicators['BB_Lower']:
        signals['BB'] = 'Buy'
    else:
        signals['BB'] = 'Neutral'

    # Example logic for additional indicators
    if indicators['IMI'] < 30:
        signals['IMI'] = 'Buy'
    elif indicators['IMI'] > 70:
        signals['IMI'] = 'Sell'
    else:
        signals['IMI'] = 'Neutral'

    if indicators['MFI'] < 30:
        signals['MFI'] = 'Buy'
    elif indicators['MFI'] > 70:
        signals['MFI'] = 'Sell'
    else:
        signals['MFI'] = 'Neutral'
    
    if indicators['Stoch_K'] < indicators['Stoch_D']:
        signals['Stochastic'] = 'Sell'
    else:
        signals['Stochastic'] = 'Buy'
    
    if entry_point < indicators['VWAP']:
        signals['VWAP'] = 'Sell'
    else:
        signals['VWAP'] = 'Buy'
    
    if indicators['CMF'] > 0:
        signals['CMF'] = 'Buy'
    else:
        signals['CMF'] = 'Sell'

    # Combine signals to make final decision
    buy_signals = [value for value in signals.values() if value == 'Buy']
    sell_signals = [value for value in signals.values() if value == 'Sell']
    
    if len(buy_signals) > len(sell_signals):
        final_signal = 'Go Long'
        take_profit = entry_point * 1.02  # Example take profit at 2% above entry
        stop_loss = entry_point * 0.98  # Example stop loss at 2% below entry
    elif len(sell_signals) > len(buy_signals):
        final_signal = 'Go Short'
        take_profit = entry_point * 0.98  # Example take profit at 2% below entry
        stop_loss = entry_point * 1.02  # Example stop loss at 2% above entry
    else:
        final_signal = 'Hold'
    
    return final_signal, take_profit, stop_loss, signals
    
    # Function to add Fear and Greed Index
    @st.cache_data(ttl=1800)
    def fetch_fear_and_greed_index():
        try:
            # Dummy data; replace with actual data source
            index = np.random.randint(0, 100)
            return index
        except Exception as e:
            st.error(f"Error fetching Fear and Greed Index: {e}")
        return None

# Streamlit app
st.title('Bitcoin Technical Analysis and Trading Signals')

data = fetch_data(ticker)
if data is not None:
    data = calculate_indicators(data)
    indicators = technical_indicators_summary(data)
    final_signal, take_profit, stop_loss, signals = generate_trading_decision(indicators, data)

    st.subheader('Technical Indicators Summary')
    st.write(indicators)

    st.subheader('Trading Decision')
    st.write(f"Final Signal: {final_signal}")
    st.write(f"Take Profit Level: {take_profit}")
    st.write(f"Stop Loss Level: {stop_loss}")

    st.subheader('Signals')
    st.write(signals)

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], mode='lines', name='EMA 20'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20'))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], mode='lines', name='BB Upper'))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], mode='lines', name='BB Lower'))
    fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))
    st.plotly_chart(fig)

    # Fear and Greed Index
    fear_and_greed_index = fetch_fear_and_greed_index()
    st.subheader('Fear and Greed Index')
    st.write(f"Current Fear and Greed Index: {fear_and_greed_index}")

    # Additional information and charts
    st.subheader('Historical Data')
    st.write(data.head())

    st.subheader('Technical Indicators Visualization')
    fig_indicators = go.Figure()
    fig_indicators.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'))
    fig_indicators.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='MACD Signal'))
    fig_indicators.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'))
    st.plotly_chart(fig_indicators)

    # Display additional details for options trading
    st.subheader('Options Trading Decision')
    st.write("Considering perpetual options trading based on market conditions...")
