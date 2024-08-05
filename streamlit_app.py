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
    'Signal': buy_sell_signals
})

# Convert 'Date' column to EST timezone
signals_df['Date'] = signals_df['Date'].apply(to_est)  # Convert dates to EST

# Display buy/sell signals with date and time in Streamlit
st.write('Buy/Sell Signals:')
for _, row in signals_df.iterrows():
    formatted_date = row['Date'].strftime('%Y-%m-%d %I:%M %p')  # Convert to EST and format
    st.write(f"{formatted_date} - **{row['Signal']}**")

    if row['Signal'] == 'BUY':
        # Predict the next significant move to determine holding time
        hold_time = np.random.randint(1, 5)  # Placeholder for actual logic
        sell_date = row['Date'] + pd.Timedelta(minutes=hold_time * 60)  # Assuming holding period in hours
        formatted_sell_date = sell_date.strftime('%Y-%m-%d %I:%M %p')  # Convert to EST and format
        st.write(f"Suggested Hold Until: **{formatted_sell_date}**")

# Plot the price chart
st.line_chart(data['Close'])

# Add JavaScript to auto-refresh the Streamlit app every 60 seconds
components.html("""
<script>
setTimeout(function(){
   window.location.reload();
}, 60000);  // Refresh every 60 seconds
</script>
""", height=0)

# Final decision on option strategy
st.write('Final Decision:')
latest_signal = signals_df.iloc[-1]['Signal']
reason = ""

# Decision based on the latest signal and sentiment
if latest_signal == 'BUY':
    if data['Sentiment'].iloc[-1] > sentiment_threshold:
        decision = "Place a CALL option"
        reason = "The model suggests a buy signal, and the sentiment is positive."
    else:
        decision = "Place a PUT option"
        reason = "The model suggests a buy signal, but the sentiment is negative, indicating caution."
else:
    if data['Sentiment'].iloc[-1] < sentiment_threshold:
        decision = "Place a PUT option"
        reason = "The model suggests a sell signal, and the sentiment is negative."
    else:
        decision = "Place a CALL option"
        reason = "The model suggests a sell signal, but the sentiment is positive, indicating caution."

st.write(f"Decision: **{decision}**")
st.write(f"Reason: {reason}")
