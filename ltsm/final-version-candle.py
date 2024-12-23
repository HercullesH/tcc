# Import necessary libraries
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load stock data from CSV file
stock_data = pd.read_csv("stock_v5.csv")

# Filter by symbol TAEE4
symbol = 'TAEE4'
stock_data = stock_data[stock_data['Symbol'] == symbol]

# Drop the 'Symbol' column as it's no longer needed
stock_data.drop('Symbol', axis=1, inplace=True)

# Add the requested columns
columns_to_include = ['usd_brl', 'Iee_Ultimo', 'Iee_Abertura', 'Iee_Maxima', 'Iee_Minima', 'Iee_Variacao', 'no_mes', '12_meses']
for col in columns_to_include:
    stock_data[col] = stock_data[col].fillna(0)  # Handle missing data

# Step 2: Preprocessing the data
stock_data['mm7'] = stock_data['Adj Close'].rolling(7).mean()
stock_data['mm21d'] = stock_data['Adj Close'].rolling(21).mean()

# Shift 'Adj Close' to avoid data leakage
stock_data['Adj Close'] = stock_data['Adj Close'].shift(-1)
stock_data.dropna(inplace=True)

# Step 3: Data Preprocessing for LSTM
closing_prices = stock_data['Adj Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(closing_prices)

# Prepare training and test set
train_size = int(len(normalized_data) * 0.8)
train_data = normalized_data[:train_size]
test_data = normalized_data[train_size - 60:]  # Subtract 60 to keep continuity

# Create sequences for training
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Create sequences for testing
x_test, y_test = [], []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Step 4: Building the LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train the model
model.fit(x_train, y_train, batch_size=1, epochs=3)

# Step 6: Model Evaluation
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate errors
mae = mean_absolute_error(y_test_scaled, predictions)
mse = mean_squared_error(y_test_scaled, predictions)
rmse = np.sqrt(mse)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Adding predicted values to a new dataframe
df_predictions = stock_data[train_size:].copy()
df_predictions['Adj Close'] = predictions
df_predictions['Close'] = predictions
df_predictions['Open'] = df_predictions['Close'].shift(1)
df_predictions['High'] = df_predictions[['Open', 'Close']].max(axis=1)
df_predictions['Low'] = df_predictions[['Open', 'Close']].min(axis=1)
df_predictions.dropna(inplace=True)

# Step 7: Create candlestick charts

# Get the last 60 days for both actual and predicted data
df_completo = stock_data.tail(60)
df_predictions = df_predictions.tail(60)

# Candlestick chart for original data with moving averages
candlestick_df = df_completo[['Open', 'High', 'Low', 'Close', 'Volume']]
candlestick_df.index = pd.to_datetime(df_completo['Date'])

# Candlestick chart for predicted data with moving averages
candlestick_df_predictions = df_predictions[['Open', 'High', 'Low', 'Close', 'Volume']]
candlestick_df_predictions.index = pd.to_datetime(df_predictions['Date'])

# Plot original candlestick with moving averages
mpf.plot(candlestick_df, type='candle', style='charles', volume=True, ylabel='Preço', 
         title=f'Últimos 60 registros - Candlestick - {symbol} (Original', 
         mav=(7, 21), savefig=f'candle_com_mm_detalhado_{symbol}.png', tight_layout=True)

# Plot predicted candlestick with moving averages
mpf.plot(candlestick_df_predictions, type='candle', style='charles', volume=True, ylabel='Preço',
         title=f'Últimos 60 registros - Candlestick - {symbol} (Previsto)',
         mav=(7, 21), savefig=f'candle_com_mm_detalhado_{symbol}_previsto.png', tight_layout=True)

# Candlestick chart for original data without moving averages
mpf.plot(candlestick_df, type='candle', style='charles', volume=True, ylabel='Preço',
         title=f'Últimos 60 registros - Candlestick - {symbol} (Original',
         savefig=f'candle_sem_mm_detalhado_{symbol}.png', tight_layout=True)

# Candlestick chart for predicted data without moving averages
mpf.plot(candlestick_df_predictions, type='candle', style='charles', volume=True, ylabel='Preço',
         title=f'Últimos 60 registros - Candlestick - {symbol}',
         savefig=f'candle_sem_mm_detalhado_{symbol}_previsto.png', tight_layout=True)

# Show the plots
plt.show()


# # Criar DataFrame para armazenar as previsões dos últimos 60 dias
# resultado_previsao_lstm = df_predictions[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

# # Adicionar coluna Date e valores previstos (Close) para clareza
# resultado_previsao_lstm['Date'] = df_completo['Date'].values  # Utiliza as datas originais
# resultado_previsao_lstm['Adj Close Predicted'] = df_predictions['Close'].values

# # Reorganizar as colunas
# resultado_previsao_lstm = resultado_previsao_lstm[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close Predicted']]

# # Salvar o DataFrame em um arquivo CSV
# csv_file_name = f'previsao_ultimos_60_dias_lstm_{symbol}.csv'
# resultado_previsao_lstm.to_csv(csv_file_name, index=False)

# print(f"Arquivo CSV com previsões dos últimos 60 dias salvo como '{csv_file_name}'.")


# Comparar valores originais e previstos
comparison_df = df_completo.copy()

# Adicionar valores previstos
comparison_df['Close_Prev'] = df_predictions['Close'].values
comparison_df['Close_Diff'] = ((comparison_df['Close_Prev'] - comparison_df['Adj Close']) / comparison_df['Adj Close']) * 100

comparison_df['Open_Prev'] = df_predictions['Open'].values
comparison_df['Open_Diff'] = ((comparison_df['Open_Prev'] - comparison_df['Open']) / comparison_df['Open']) * 100

comparison_df['High_Prev'] = df_predictions['High'].values
comparison_df['High_Diff'] = ((comparison_df['High_Prev'] - comparison_df['High']) / comparison_df['High']) * 100

comparison_df['Low_Prev'] = df_predictions['Low'].values
comparison_df['Low_Diff'] = ((comparison_df['Low_Prev'] - comparison_df['Low']) / comparison_df['Low']) * 100

# Criar o DataFrame com as colunas organizadas em pares
comparison_clean = pd.DataFrame({
    'Close': comparison_df['Adj Close'],
    'Close_Prev': comparison_df['Close_Prev'],
    'Close_Diff': comparison_df['Close_Diff'],
    'Open': comparison_df['Open'],
    'Open_Prev': comparison_df['Open_Prev'],
    'Open_Diff': comparison_df['Open_Diff'],
    'High': comparison_df['High'],
    'High_Prev': comparison_df['High_Prev'],
    'High_Diff': comparison_df['High_Diff'],
    'Low': comparison_df['Low'],
    'Low_Prev': comparison_df['Low_Prev'],
    'Low_Diff': comparison_df['Low_Diff']
})

# Salvar o DataFrame comparativo em um CSV
comparison_clean.to_csv(f'comparacao_{symbol}.csv', index=True)
print("Arquivo CSV de comparação gerado com sucesso para LSTM.")

# Exibir as primeiras linhas do DataFrame comparativo
print(comparison_clean.head())
