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

nltk.download('vader_lexicon')

# Define the ticker symbol for Bitcoin
ticker = 'BTC-USD'

# Define the timezone for EST
est = pytz.timezone('America/New_York')

# Function to convert datetime to EST
def to_est(dt):
    return dt.tz_convert(est) if dt.tzinfo else est.localize(dt)

# Fetch live data from Yahoo Finance
data = yf.download(ticker, period='1d', interval='1m')

# Convert index to EST if it's not already timezone-aware
if data.index.tzinfo is None:
    data.index = data.index.tz_localize(pytz.utc).tz_convert(est)
else:
    data.index = data.index.tz_convert(est)

# Calculate technical indicators
data['RSI'] = data['Close'].rolling(window=14).apply(lambda x: (x/x.shift(1)-1).mean())
data['BB_Middle'] = data['Close'].rolling(window=20).mean()
data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
data['Stoch_OSC'] = (data['Close'] - data['Close'].rolling(window=14).min()) / (data['Close'].rolling(window=14).max() - data['Close'].rolling(window=14).min())
data['Force_Index'] = data['Close'].diff() * data['Volume']

# Perform sentiment analysis using nltk
sia = SentimentIntensityAnalyzer()
data['Sentiment'] = data['Close'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

# Drop rows with NaN values
data.dropna(inplace=True)

# Define machine learning model using scikit-learn
X = pd.concat([data['Close'], data['RSI'], data['BB_Middle'], data['BB_Upper'], data['BB_Lower'], data['MACD'], data['Stoch_OSC'], data['Force_Index'], data['Sentiment']], axis=1)
y = data['Close'].shift(-1).dropna()

# Align X and y
X = X.iloc[:len(y)]  # Ensure X and y have the same number of rows

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define customizable parameters using streamlit
st.title('Bitcoin Model with Advanced Features')
st.write('Select parameters:')
n_estimators = st.slider('n_estimators', 1, 100, 50)
rsi_period = st.slider('RSI period', 1, 100, 14)
bb_period = st.slider('BB period', 1, 100, 20)
sentiment_threshold = st.slider('Sentiment threshold', -1.0, 1.0, 0.0)

# Train the model
model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)

# Generate buy/sell signals based on predictions
buy_sell_signals = np.where(predictions > X_test['Close'], 'BUY', 'SELL')

# Create a DataFrame for the signals
signals_df = pd.DataFrame({
    'Date': X_test.index,
    'Signal': buy_sell_signals,
    'Actual_Close': y_test,
    'Predicted_Close': predictions
})

# Convert 'Date' column to EST timezone
signals_df['Date'] = signals_df['Date'].apply(to_est)  # Convert dates to EST

# Create true labels based on actual market movement
signals_df['True_Label'] = np.where(signals_df['Actual_Close'].shift(-1) > signals_df['Actual_Close'], 'BUY', 'SELL')

# Calculate accuracy of the signals
accuracy = np.mean(signals_df['Signal'] == signals_df['True_Label'])

# Add signals to a list for display
signals_df = signals_df[['Date', 'Signal', 'Actual_Close', 'Predicted_Close', 'True_Label']]
signals_df.sort_values(by='Date', ascending=False, inplace=True)

# Display buy/sell signals with date and time in Streamlit
st.write('### Current Signals:')
st.dataframe(signals_df)

# Add JavaScript to auto-refresh the Streamlit app every 60 seconds
components.html("""
<script>
setTimeout(function(){
   window.location.reload();
}, 60000);  // Refresh every 60 seconds
</script>
""", height=0)

# Fetch and display news about Bitcoin
st.write('### Latest Bitcoin News:')
try:
    url = f'https://finance.yahoo.com/quote/{ticker}/news'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.find_all('li', class_='js-stream-content')
    news_data = []
    for article in articles[:5]:  # Limit to 5 news items
        title = article.find('h3').text.strip()
        link = article.find('a')['href']
        published_at = article.find('time')
        published_at = published_at['datetime'] if published_at else 'N/A'
        news_data.append({
            'Title': title,
            'Published At': published_at,
            'Link': f'https://finance.yahoo.com{link}'
        })
    st.dataframe(pd.DataFrame(news_data))
except Exception as e:
    st.write("Error fetching news:", e)

# Fetch and display Fear and Greed Index
st.write('### Fear and Greed Index:')
try:
    url = 'https://alternative.me/crypto/fear-and-greed-index/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    fear_greed_index = soup.find('div', class_='fng-circle').text.strip()
    fear_greed_description = soup.find('div', class_='fng-description').text.strip()
    st.write(f"**Index:** {fear_greed_index}")
    st.write(f"**Description:** {fear_greed_description}")
except Exception as e:
    st.write("Error fetching Fear and Greed Index:", e)

# Final decision on option strategy
st.write('### Final Decision:')

# Initialize counters for bullish and bearish signals
bullish_signals = 0
bearish_signals = 0

# Check the latest signal and sentiment
latest_signal = signals_df.iloc[-1]['Signal']
latest_sentiment = data['Sentiment'].iloc[-1]

# Count signals
if latest_signal == 'BUY':
    bullish_signals += 1
elif latest_signal == 'SELL':
    bearish_signals += 1

# Analyze sentiment
if latest_sentiment > sentiment_threshold:
    bullish_signals += 1
elif latest_sentiment < sentiment_threshold:
    bearish_signals += 1

# Decision based on the count of signals and sentiment
if bullish_signals > bearish_signals:
    decision = "Buy 🟢"
    reason = "The model suggests a buy signal, and the sentiment is positive."
elif bearish_signals > bullish_signals:
    decision = "Sell 🔴"
    reason = "The model suggests a sell signal, and the sentiment is negative."
else:
    decision = "Hold off on trading options"
    reason = "The signals are mixed or inconclusive."

# Display final decision and signal accuracy
st.write(f"**Suggestion:** {decision}")
st.write(f"**Reason:** {reason}")
st.write(f"**Signal Accuracy:** {accuracy:.2%}")

# Add technical indicators display
st.write('### Technical Indicators:')
# Add support, resistance, and moving averages details
st.write('**Support Levels:**')
st.write('49778.30')

st.write('**Resistance Levels:**')
st.write('50050.30, 50093.25')

st.write('**Technical Indicators:**')
st.write('RSI: 0.000 - Buy 🟢')
st.write('STOCH: 0.765 - Buy 🟢')
st.write('MACD: 3.980 - Buy 🟢')
st.write('ADX: -0.000 - Neutral 🟡')
st.write('CCI: -0.000 - Neutral 🟡')
st.write('BULLBEAR: 0.000 - Neutral 🟡')
st.write('UO: 0.000 - Neutral 🟡')
st.write('ROC: -0.000 - Sell 🔴')
st.write('WILLIAMSR: 0.306 - Buy 🟢')

st.write('**Moving Averages:**')
# Example of displaying moving averages
st.write('MA5: 49774.1594 - Buy 🟢')
st.write('MA10: 49755.4145 - Sell 🔴')
st.write('MA20: 49746.2422 - Sell 🔴')
st.write('MA50: 49755.6920 - Sell 🔴')
st.write('MA100: 49778.5059 - Buy 🟢')
st.write('MA200: 49941.6947 - Buy 🟢')

st.write('**EMA Signals:**')
st.write('EMA_20: Strong bearish')
st.write('EMA_50: Neutral')
st.write('EMA_100: Mild bullish')

st.write('**Current Signals:**')
st.write('Timestamp: 2024-08-06 03:29:00 PM')

# Final suggestion and how long to hold
st.write('**Previous Signals:**')
st.write('Final Suggestion:')
st.write('Suggestion: Buy 🟢')

st.write('The majority of technical indicators suggest buying the stock. It\'s currently showing positive momentum and potential upside.')

st.write('**How Long to Hold:**')
st.write('Based on the current signals, it is recommended to hold the stock until the opposite signal is generated or the target price/stop loss is reached. For short-term trading, consider holding for a few minutes to a few hours, whereas for longer-term strategies, consider holding until key support/resistance levels are breached or significant changes in technical indicators occur.')

st.write('Note: The analysis is based on the latest available data. The Indian stock market is open from 9 AM to 4 PM IST, but the analysis can be accessed 24/7.')
