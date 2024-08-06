import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import pytz
from datetime import datetime
import streamlit.components.v1 as components
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

# Download NLTK data for sentiment analysis
nltk.download('vader_lexicon')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Define the ticker symbol for Bitcoin
ticker = 'BTC-USD'

# Define the timezone for EST
est = pytz.timezone('America/New_York')

# Function to convert datetime to EST
def to_est(dt):
    return dt.tz_convert(est) if dt.tzinfo else est.localize(dt)

# Fetch live data from Yahoo Finance
@st.cache
def fetch_data(ticker):
    data = yf.download(ticker, period='1d', interval='1m')
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize(pytz.utc).tz_convert(est)
    else:
        data.index = data.index.tz_convert(est)
    return data

data = fetch_data(ticker)

# Calculate technical indicators
def calculate_indicators(data):
    data['RSI'] = data['Close'].rolling(window=14).apply(lambda x: (x/x.shift(1)-1).mean())
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Stoch_OSC'] = (data['Close'] - data['Close'].rolling(window=14).min()) / (data['Close'].rolling(window=14).max() - data['Close'].rolling(window=14).min())
    data['Force_Index'] = data['Close'].diff() * data['Volume']
    data['Sentiment'] = data['Close'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    data.dropna(inplace=True)
    return data

data = calculate_indicators(data)

# Define machine learning model
X = pd.concat([data['Close'], data['RSI'], data['BB_Middle'], data['BB_Upper'], data['BB_Lower'], data['MACD'], data['Stoch_OSC'], data['Force_Index'], data['Sentiment']], axis=1)
y = data['Close'].shift(-1).dropna()
X = X.iloc[:len(y)]  # Ensure X and y have the same number of rows
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)
buy_sell_signals = np.where(predictions > X_test['Close'], 'GO LONG', 'GO SHORT')

# Create a DataFrame for the signals
signals_df = pd.DataFrame({
    'Date': X_test.index,
    'Signal': buy_sell_signals,
    'Actual_Close': y_test,
    'Predicted_Close': predictions
})
signals_df['Date'] = signals_df['Date'].apply(to_est)  # Convert dates to EST
signals_df['True_Label'] = np.where(signals_df['Actual_Close'].shift(-1) > signals_df['Actual_Close'], 'GO LONG', 'GO SHORT')
accuracy = np.mean(signals_df['Signal'] == signals_df['True_Label'])
signals_df = signals_df.sort_values(by='Date', ascending=False)

# Fetch Fear and Greed Index from Alternative.me
def fetch_fear_and_greed():
    try:
        url = 'https://alternative.me/crypto/fear-and-greed-index/'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        value = soup.find('div', {'class': 'fng-circle'}).text.strip()
        return int(value)
    except Exception as e:
        st.error(f"Error fetching Fear and Greed Index: {e}")
        return None

fear_and_greed_index = fetch_fear_and_greed()

# Display the title
st.title('BTC-Derivatives Technical Analysis')

# Plot the price chart with support and resistance
st.write('### Bitcoin Price Chart with Support and Resistance')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Close'], label='Close')
ax.plot(data['BB_Middle'], label='BB Middle')
ax.plot(data['BB_Upper'], label='BB Upper', linestyle='--')
ax.plot(data['BB_Lower'], label='BB Lower', linestyle='--')
ax.axhline(y=support, color='r', linestyle='-', label='Support')
ax.axhline(y=resistance, color='g', linestyle='-', label='Resistance')
ax.set_title('Bitcoin Price Chart with Support and Resistance')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.legend(loc='upper left')
st.pyplot(fig)

# Plot the 1-hour chart
st.write('### 1-Hour Bitcoin Price Chart')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Close'].resample('1h').mean())
ax.set_title('1-Hour Bitcoin Price Chart')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
st.pyplot(fig)

# Display decision section
st.write('### Final Decision:')

# Initialize counters for bullish and bearish signals
bullish_signals = 0
bearish_signals = 0

# Check the latest signal and sentiment
latest_signal = signals_df.iloc[0]['Signal']
latest_sentiment = data['Sentiment'].iloc[-1]

# Count signals
if latest_signal == 'GO LONG':
    bullish_signals += 1
elif latest_signal == 'GO SHORT':
    bearish_signals += 1

# Analyze sentiment
if latest_sentiment > 0:
    bullish_signals += 1
elif latest_sentiment < 0:
    bearish_signals += 1

# Include Fear and Greed Index in the decision
if fear_and_greed_index is not None:
    if fear_and_greed_index > 50:
        bullish_signals += 1
    elif fear_and_greed_index < 50:
        bearish_signals += 1

# Define support and resistance (example values; adjust as needed)
support = data['BB_Lower'].iloc[-1]
resistance = data['BB_Upper'].iloc[-1]

# Decision based on the count of signals, sentiment, and Fear and Greed Index
if bullish_signals > bearish_signals:
    decision = "GO LONG "
    reason = "The model suggests a buy signal, sentiment is positive, and the Fear and Greed Index indicates greed."
    stop_loss = support
elif bearish_signals > bullish_signals:
    decision = "GO SHORT "
    reason = "The model suggests a sell signal, sentiment is negative, and the Fear and Greed Index indicates fear."
    stop_loss = resistance
else:
    decision = "HOLD OPTION "
    reason = "The signals are mixed; it's best to hold the current position."
    stop_loss = None

# Day trading decision
if data['RSI'].iloc[-1] < 30 and data['MACD'].iloc[-1] > 0:
    day_trading_decision = "BUY FOR DAY TRADING "
elif data['RSI'].iloc[-1] > 70 and data['MACD'].iloc[-1] < 0:
    day_trading_decision = "SELL FOR DAY TRADING "
else:
    day_trading_decision = "HOLD FOR DAY TRADING "

st.write(f"### Decision: {decision}")
st.write(f"**Reason:** {reason}")
if stop_loss is not None:
    st.write(f"**Stop Loss:** {stop_loss:.3f}")
st.write(f"### Day Trading Decision: {day_trading_decision}")

# Display buy/sell signals in a table
st.write('### Buy/Sell Signals:')
st.dataframe(signals_df[['Date', 'Signal', 'Actual_Close', 'Predicted_Close']])

# Fetch latest news using BeautifulSoup
st.write('### Latest News:')
try:
    url = 'https://www.coindesk.com/feed/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'xml')
    articles = soup.find_all('item')
    for article in articles[:5]:  # Limit to latest 5 news items
        title = article.title.text
        link = article.link.text
        pub_date = article.pubDate.text
        st.write(f"**{title}**")
        st.write(f"Published on: {pub_date}")
        st.write(f"Link: [Read more]({link})")
        st.write("---")
except Exception as e:
    st.error(f"Error fetching latest news: {e}")

# Add JavaScript to auto-refresh the Streamlit app every 60 seconds
components.html("""
<script>
setTimeout(function(){
    window.location.reload();
}, 60000);  // Refresh every 60 seconds
</script>
""", height=0)
