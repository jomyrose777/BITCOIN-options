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

# Function to convert datetime to EST
def to_est(dt):
    if dt.tzinfo is None:
        dt = est.localize(dt)
    return dt.astimezone(est)

# Function to fetch live data from Yahoo Finance
@st.cache_data(ttl=30)
def fetch_data(ticker):
    try:
        data = yf.download(ticker, period='1d', interval='1m')
        if data.empty:
            st.error("No data fetched from Yahoo Finance.")
            return None

        # Convert datetime index to EST timezone
        if data.index.tzinfo is None:
            data.index = data.index.tz_localize(pytz.utc).tz_convert(est)
        else:
            data.index = data.index.tz_convert(est)

        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_indicators(data):
    if len(data) < 14:
        st.error("Insufficient data to calculate indicators.")
        return data

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

    # Williams %R
    try:
        williams_r = ta.momentum.WilliamsRIndicator(high=data['High'], low=data['Low'], close=data['Close'])
        data['Williams %R'] = williams_r.williams_r()
    except Exception as e:
        st.error(f"Error calculating Williams %R: {e}")
        data['Williams %R'] = np.nan

    # Money Flow Index (MFI)
    mfi = ta.volume.MFIIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'])
    data['MFI'] = mfi.money_flow_index()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'], window=14, smooth_window=3)
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()

    # Average True Range (ATR)
    if len(data) >= 14:
        try:
            atr = ta.volatility.AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14)
            data['ATR'] = atr.average_true_range()
        except Exception as e:
            st.error(f"Error calculating ATR: {e}")
            data['ATR'] = np.nan
    else:
        data['ATR'] = np.nan

    # Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(high=data['High'], low=data['Low'], window1=9, window2=26, window3=52)
    data['Ichimoku_A'] = ichimoku.ichimoku_a()
    data['Ichimoku_B'] = ichimoku.ichimoku_b()
    data['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
    data['Ichimoku_Lead'] = ichimoku.ichimoku_a().shift(26)

    # Parabolic SAR
    try:
        sar = ta.trend.PSARIndicator(high=data['High'], low=data['Low'], close=data['Close'])
        data['SAR'] = sar.psar()
    except Exception as e:
        st.error(f"Error calculating PSAR: {e}")
        data['SAR'] = np.nan

    # VWAP
    vwap = ta.volume.VolumeWeightedAveragePrice(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'])
    data['VWAP'] = vwap.volume_weighted_average_price()

    # Chaikin Money Flow (CMF)
    cmf = ta.volume.ChaikinMoneyFlowIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=20)
    data['CMF'] = cmf.chaikin_money_flow()

    data.dropna(inplace=True)
    return data

def technical_indicators_summary(data):
    indicators = {}
    for col in [
        'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'SMA_20',
        'EMA_20', 'OBV', 'Fib_0.236', 'Fib_0.382', 'Fib_0.618',
        'Williams %R', 'MFI', 'Stoch_K', 'Stoch_D', 'ATR', 'Ichimoku_A',
        'Ichimoku_B', 'Ichimoku_Base', 'Ichimoku_Lead', 'SAR', 'VWAP', 'CMF'
    ]:
        if col in data.columns and not data[col].empty:
            indicators[col] = data[col].iloc[-1]
        else:
            indicators[col] = 'N/A'
    return indicators

def generate_trading_decision(indicators, data):
    signals = {}
    entry_point = data['Close'].iloc[-1]
    take_profit = None
    stop_loss = None

    # Example logic for signal generation
    if indicators['RSI'] != 'N/A':
        if indicators['RSI'] < 30:
            signals['RSI'] = 'Buy'
            take_profit = entry_point * 1.05  # Example Take Profit
            stop_loss = entry_point * 0.95   # Example Stop Loss
        elif indicators['RSI'] > 70:
            signals['RSI'] = 'Sell'
            take_profit = entry_point * 0.95  # Example Take Profit
            stop_loss = entry_point * 1.05   # Example Stop Loss
        else:
            signals['RSI'] = 'Neutral'
    
    if indicators['MACD'] != 'N/A':
        if indicators['MACD'] > 0:
            signals['MACD'] = 'Buy'
        elif indicators['MACD'] < 0:
            signals['MACD'] = 'Sell'
        else:
            signals['MACD'] = 'Neutral'
    
    if indicators['BB_Upper'] != 'N/A' and indicators['BB_Lower'] != 'N/A':
        if entry_point > indicators['BB_Upper']:
            signals['BB'] = 'Sell'
        elif entry_point < indicators['BB_Lower']:
            signals['BB'] = 'Buy'
        else:
            signals['BB'] = 'Neutral'
    
    if indicators['Williams %R'] != 'N/A':
        if indicators['Williams %R'] < -80:
            signals['Williams %R'] = 'Buy'
        elif indicators['Williams %R'] > -20:
            signals['Williams %R'] = 'Sell'
        else:
            signals['Williams %R'] = 'Neutral'
    
    if indicators['MFI'] != 'N/A':
        if indicators['MFI'] < 20:
            signals['MFI'] = 'Buy'
        elif indicators['MFI'] > 80:
            signals['MFI'] = 'Sell'
        else:
            signals['MFI'] = 'Neutral'
    
    if indicators['Stoch_K'] != 'N/A' and indicators['Stoch_D'] != 'N/A':
        if indicators['Stoch_K'] < indicators['Stoch_D']:
            signals['Stochastic'] = 'Sell'
        else:
            signals['Stochastic'] = 'Buy'
    
    if indicators['VWAP'] != 'N/A':
        if entry_point < indicators['VWAP']:
            signals['VWAP'] = 'Sell'
        else:
            signals['VWAP'] = 'Buy'
    
    if indicators['CMF'] != 'N/A':
        if indicators['CMF'] > 0:
            signals['CMF'] = 'Buy'
        else:
            signals['CMF'] = 'Sell'
    
    # Determine final signal
    buy_signals = [k for k, v in signals.items() if v == 'Buy']
    sell_signals = [k for k, v in signals.items() if v == 'Sell']
    
    if len(buy_signals) > len(sell_signals):
        final_signal = 'Go Long'
    elif len(sell_signals) > len(buy_signals):
        final_signal = 'Go Short'
    else:
        final_signal = 'Neutral'
    
    return signals, final_signal, take_profit, stop_loss

def plot_data(data):
    fig = go.Figure()

    # Plot the price data
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlestick'))
    
    # Plot moving averages
    if 'SMA_20' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20'))
    if 'EMA_20' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], mode='lines', name='EMA 20'))

    # Plot Bollinger Bands
    if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='green')))

    # Plot MACD
    if 'MACD' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
    if 'MACD_Signal' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='MACD Signal', line=dict(color='orange')))

    fig.update_layout(title='Bitcoin Price and Indicators', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

def main():
    st.title("Bitcoin Trading Analysis")

    # Fetch data
    data = fetch_data(ticker)
    if data is None:
        return

    # Calculate indicators
    data = calculate_indicators(data)

    # Display data
    st.write("### Latest Data")
    st.write(data.tail())

    # Display technical indicators summary
    indicators = technical_indicators_summary(data)
    st.subheader('Technical Indicators Summary')
    st.write(indicators)

    # Generate trading decision
    signals, final_signal, take_profit, stop_loss = generate_trading_decision(indicators, data)
    
    # Display trading decision
    st.subheader('Trading Decision')
    st.write(f"Final Signal: {final_signal}")
    st.write(f"Take Profit Level: {take_profit}")
    st.write(f"Stop Loss Level: {stop_loss}")

    # Display signals
    st.subheader('Signals')
    st.write(signals)

    # Plot data
    plot_data(data)

if __name__ == "__main__":
    main()
