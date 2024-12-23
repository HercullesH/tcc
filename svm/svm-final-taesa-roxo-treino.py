import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import mplfinance as mpf  # Para candlesticks
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Carregar os dados
data = pd.read_csv("stock_v5.csv")

# Filtrar apenas as ações com o símbolo TAEE4
symbol = "TAEE4"
data_taee4 = data[data['Symbol'] == symbol].copy()

# Converter a coluna Date para datetime e ordenar por data
data_taee4.loc[:, 'Date'] = pd.to_datetime(data_taee4['Date'])
data_taee4 = data_taee4.sort_values(by='Date')

# Preencher valores ausentes (NaN) com a média da coluna
data_taee4.fillna(data_taee4.mean(numeric_only=True), inplace=True)

# Selecionar as últimas 60 entradas para análise
last_60_days = data_taee4.tail(60)

# Preparar os dados para o modelo
X = data_taee4.drop(columns=['Date', 'Symbol', 'Adj Close'])
y = data_taee4['Adj Close']

# Escalar os dados
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Treinar o modelo SVM
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train, y_train)

# Fazer previsões
y_pred_scaled = svr.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Prever os últimos 60 dias
X_last_60_scaled = scaler_X.transform(last_60_days.drop(columns=['Date', 'Symbol', 'Adj Close']))
y_last_60_pred_scaled = svr.predict(X_last_60_scaled)
y_last_60_pred = scaler_y.inverse_transform(y_last_60_pred_scaled.reshape(-1, 1)).ravel()



# Dados originais para candlesticks (últimos 60 dias)
candlestick_df = last_60_days[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
candlestick_df.set_index('Date', inplace=True)

# Calcular métricas para os dados de teste
mae_test = mean_absolute_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred.reshape(-1, 1))
mse_test = mean_squared_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred.reshape(-1, 1))
rmse_test = np.sqrt(mse_test)

print(f"MAE (Teste): {mae_test:.6f}")
print(f"MSE (Teste): {mse_test:.6f}")
print(f"RMSE (Teste): {rmse_test:.6f}")

# Ajustar os candles previstos com base no comportamento dos dados originais
candlestick_df_previsao = candlestick_df.copy()
candlestick_df_previsao['Close'] = y_last_60_pred
candlestick_df_previsao['Open'] = candlestick_df['Open'] * (candlestick_df_previsao['Close'] / candlestick_df['Close'])
candlestick_df_previsao['High'] = np.maximum(candlestick_df_previsao['Close'], candlestick_df_previsao['Open']) * 1.01
candlestick_df_previsao['Low'] = np.minimum(candlestick_df_previsao['Close'], candlestick_df_previsao['Open']) * 0.99

# Gráfico de candlestick sem média móvel (original)
mpf.plot(candlestick_df, type='candle', style='charles', volume=True, ylabel='Preço', show_nontrading=False,
         title=f'Últimos 60 registros - Candlestick - {symbol} (Original)',
         savefig=f'candle_sem_mm_detalhado_{symbol}.png', tight_layout=True)
print("Gráfico original sem média móvel salvo.")

# Gráfico de candlestick com média móvel (original)
mpf.plot(candlestick_df, type='candle', style='charles', volume=True, ylabel='Preço', show_nontrading=False,
         title=f'Últimos 60 registros - Candlestick - {symbol} (Originall)', mav=(7, 21),
         savefig=f'candle_com_mm_detalhado_{symbol}.png', tight_layout=True)
print("Gráfico original com média móvel salvo.")

# Gráfico de candlestick sem média móvel (previsto)
mpf.plot(candlestick_df_previsao, type='candle', style='charles', volume=True, ylabel='Preço', show_nontrading=False,
         title=f'Últimos 60 registros - Candlestick - {symbol} (Previsto)',
         savefig=f'candle_sem_mm_detalhado_{symbol}_previsto.png', tight_layout=True)
print("Gráfico previsto sem média móvel salvo.")

# Gráfico de candlestick com média móvel (previsto)
mpf.plot(candlestick_df_previsao, type='candle', style='charles', volume=True, ylabel='Preço', show_nontrading=False,
         title=f'Últimos 60 registros - Candlestick - {symbol} (Previsto)', mav=(7, 21),
         savefig=f'candle_com_mm_detalhado_{symbol}_previsto.png', tight_layout=True)
print("Gráfico previsto com média móvel salvo.")


# # Criar um DataFrame para os resultados previstos dos últimos 60 dias
# resultado_previsao = last_60_days[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
# resultado_previsao['Close_Prev'] = y_last_60_pred

# # Salvar os dados em um arquivo CSV
# resultado_previsao.to_csv(f'previsao_ultimos_60_dias_{symbol}.csv', index=False)
# print(f"Arquivo CSV com previsões dos últimos 60 dias salvo como 'previsao_ultimos_60_dias_{symbol}.csv'.")

# Comparar valores originais e previstos
comparison_df = candlestick_df.copy()

# Calcula as diferenças em porcentagem e adiciona os valores previstos
comparison_df['Close_Prev'] = candlestick_df_previsao['Close']
comparison_df['Close_Diff'] = ((candlestick_df_previsao['Close'] - comparison_df['Close']) / comparison_df['Close']) * 100

comparison_df['Open_Prev'] = candlestick_df_previsao['Open']
comparison_df['Open_Diff'] = ((candlestick_df_previsao['Open'] - comparison_df['Open']) / comparison_df['Open']) * 100

comparison_df['High_Prev'] = candlestick_df_previsao['High']
comparison_df['High_Diff'] = ((candlestick_df_previsao['High'] - comparison_df['High']) / comparison_df['High']) * 100

comparison_df['Low_Prev'] = candlestick_df_previsao['Low']
comparison_df['Low_Diff'] = ((candlestick_df_previsao['Low'] - comparison_df['Low']) / comparison_df['Low']) * 100

# Criar o DataFrame com as colunas organizadas em pares
comparison_clean = pd.DataFrame({
    'Close': comparison_df['Close'],
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
print("Arquivo CSV de comparação gerado com sucesso.")

# Exibir as primeiras linhas do DataFrame comparativo
print(comparison_clean.head())

