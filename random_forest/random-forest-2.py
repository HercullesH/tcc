import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import mplfinance as mpf

# Carregar os dados
file_path = 'stock_v5.csv'  # Atualize para o caminho correto
data = pd.read_csv(file_path)

# Garantir que os nomes das colunas não tenham espaços extras
data.columns = data.columns.str.strip()

# Filtrar por uma ação específica e ordenar por data
symbol = 'TAEE4'  # Escolha um símbolo específico
data['Date'] = pd.to_datetime(data['Date'])
data = data[data['Symbol'] == symbol].sort_values(by='Date')

# Adicionar colunas adicionais para enriquecer os dados
columns_to_include = ['usd_brl', 'Iee_Ultimo', 'Iee_Abertura', 'Iee_Maxima', 'Iee_Minima', 'Iee_Variacao', 'no_mes', '12_meses']
for col in columns_to_include:
    data[col] = data[col].fillna(0)  # Substituir valores nulos por 0

# Adicionar médias móveis
data['mm7'] = data['Adj Close'].rolling(7).mean()
data['mm21d'] = data['Adj Close'].rolling(21).mean()

# Remover valores nulos gerados pelas médias móveis
data.dropna(inplace=True)

# Definir as features e os targets
features = ['Close', 'Open', 'Volume'] + columns_to_include + ['mm7', 'mm21d']
targets = ['Adj Close', 'Open', 'Close']

X = data[features]
Y = data[targets]

# Dividir os dados em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Treinar o modelo Random Forest
models = {col: RandomForestRegressor(n_estimators=100, random_state=42) for col in targets}
for col in targets:
    models[col].fit(X_train, Y_train[col])

# Fazer previsões
predictions = {col: models[col].predict(X_test) for col in targets}

# Adicionar previsões ao dataframe de teste
last_60_days = data.iloc[-60:].copy()
pred_last_60 = {col: models[col].predict(last_60_days[features]) for col in targets}

# Adicionar previsões ao dataframe
last_60_days['Adj Close Predicted'] = pred_last_60['Adj Close']
last_60_days['Close_Prev'] = pred_last_60['Close']
last_60_days['Open_Prev'] = pred_last_60['Open']

# Comparar valores originais e previstos
last_60_days['Close_Diff'] = ((last_60_days['Close_Prev'] - last_60_days['Adj Close']) / last_60_days['Adj Close']) * 100
last_60_days['Open_Diff'] = ((last_60_days['Open_Prev'] - last_60_days['Open']) / last_60_days['Open']) * 100

# Criar o DataFrame comparativo
comparison_clean = pd.DataFrame({
    'Date': last_60_days['Date'],
    'Close': last_60_days['Adj Close'],
    'Close_Prev': last_60_days['Close_Prev'],
    'Close_Diff': last_60_days['Close_Diff'],
    'Open': last_60_days['Open'],
    'Open_Prev': last_60_days['Open_Prev'],
    'Open_Diff': last_60_days['Open_Diff'],
    'Volume': last_60_days['Volume']
})

# Salvar o DataFrame comparativo em um CSV
csv_file_name = f'comparacao_{symbol}.csv'
comparison_clean.to_csv(csv_file_name, index=False)
print(f"Arquivo CSV de comparação gerado com sucesso para Random Forest: {csv_file_name}")

# Gerar os gráficos (removendo High e Low)
last_60_days.set_index('Date', inplace=True)
pred_df = last_60_days[['Open_Prev', 'Close_Prev', 'Volume']]
pred_df.columns = ['Open', 'Close', 'Volume']

# 1. Gráfico original
mpf.plot(last_60_days[['Open', 'Close', 'Volume']], type='candle', style='charles', volume=True,
         title=f'Últimos 60 registros - Candlestick - {symbol} (Original)',
         savefig=f'candle_original_{symbol}.png')

# 2. Gráfico previsto
mpf.plot(pred_df, type='candle', style='charles', volume=True,
         title=f'Últimos 60 registros - Candlestick - {symbol} (Previsto)',
         savefig=f'candle_previsto_{symbol}.png')

print("Gráficos gerados e salvos com sucesso sem High e Low.")
