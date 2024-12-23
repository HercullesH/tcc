# Import necessary libraries
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

stock_data = pd.read_csv("petr4.csv")

# Step 2: Visualizing Stock Prices History
plt.figure(figsize=(12, 6))
plt.title("PETR4 Stock Prices Over Time")
plt.plot(stock_data['Close'], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.savefig('PETR4_overtime.png')

# Step 3: Data Preprocessing
closing_prices = stock_data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(closing_prices)

# Prepare training set
train_size = int(len(normalized_data) * 0.8)
train_data = normalized_data[:train_size]

# Create sequences for training
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Prepare test set
test_data = normalized_data[train_size - 60:]
x_test, y_test = [], []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Step 3.3: Setting Up LSTM Network Architecture
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 3.4: Training LSTM Model
model.fit(x_train, y_train, batch_size=1, epochs=3)

# Step 3.5: Model Evaluation
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Step 3.6: Visualizing the Predicted Prices
train = stock_data[:train_size]
valid = stock_data[train_size:]
valid['Predictions'] = predictions

plt.figure(figsize=(12, 6))
plt.title(f"PETR4 Stock Price Prediction")
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.plot(train['Close'], label='Training Data')
plt.plot(valid[['Close', 'Predictions']], label='Actual and Predicted Prices')
plt.legend()

plt.savefig('ltsm-petr4.png')