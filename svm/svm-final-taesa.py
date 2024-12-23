import pandas as pd
from sklearn.svm import SVC 
import numpy as np 
import matplotlib.pyplot as plt

# Carregar o arquivo CSV em um DataFrame
df = pd.read_csv("stock.csv")

# Encontrar os valores únicos na coluna 'Symbol'
#unique_symbols = df['Symbol'].unique()

unique_symbols = ['TAEE11']

# Lista para armazenar símbolos sem registros suficientes
symbols_without_enough_records = []

# Loop para criar um gráfico para cada símbolo e verificar o número de registros
for symbol in unique_symbols:
    # Filtrar o DataFrame para o símbolo atual
    df_filtered = df[df['Symbol'] == symbol]

    # Verificar se há dados suficientes para treinamento
    if len(df_filtered) >= 2:  # Mude para 1 se você deseja pelo menos uma amostra
        # Remover informações de fuso horário (assumindo que não são necessárias)
        df_filtered['Date'] = df_filtered['Date'].str.split(' ').str[0]

        # Converter a coluna 'Date' para o formato de data "%Y-%m-%d"
        df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], format="%Y-%m-%d")
        df_filtered.index = pd.to_datetime(df_filtered['Date'])

        # drop The original date column
        df_filtered = df_filtered.drop(['Date'], axis='columns')

        df_filtered['Open-Close'] = df_filtered.Open - df_filtered.Close
        df_filtered['High-Low'] = df_filtered.High - df_filtered.Low

        # Store all predictor variables in a variable X
        X = df_filtered[['Open-Close', 'High-Low']]

        # Target variables
        y = np.where(df_filtered['Close'].shift(-1) > df_filtered['Close'], 1, 0)

        split_percentage = 0.8
        split = int(split_percentage * len(df_filtered))

        # Train data set
        X_train = X[:split]
        y_train = y[:split]

        # Test data set
        X_test = X[split:]
        y_test = y[split:]

        # Support vector classifier
        cls = SVC().fit(X_train, y_train)

        df_filtered['Predicted_Signal'] = cls.predict(X)

        df_filtered['Return'] = df_filtered.Close.pct_change()
        # Calculate strategy returns
        df_filtered['Strategy_Return'] = df_filtered.Return * df_filtered.Predicted_Signal.shift(1)

        # Calculate Cumulative returns
        df_filtered['Cum_Ret'] = df_filtered['Return'].cumsum()

        # Plot Strategy Cumulative returns
        df_filtered['Cum_Strategy'] = df_filtered['Strategy_Return'].cumsum()

        # Plotando o gráfico
        plt.plot(df_filtered['Cum_Ret'], color='red', label='Original')
        plt.plot(df_filtered['Cum_Strategy'], color='blue', label='Previsto')
        # plt.plot(df_filtered['Cum_Ret'] + df_filtered['Strategy_Return'].fillna(0).cumsum(), color='green', label='Cumulative Return (With Strategy)')  # Linha verde

        # Configurações adicionais do gráfico
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.title(f'Cumulative Returns Comparison for {symbol}')
        plt.savefig(f"{symbol}_comparison.png")

        # Limpar o gráfico para a próxima iteração
        plt.clf()

    else:
        # Adicionar o símbolo à lista de símbolos sem registros suficientes
        symbols_without_enough_records.append(symbol)

# Exibir o gráfico na tela
#plt.show()

# Escrever a lista de símbolos sem registros suficientes em um arquivo de texto
with open("symbols_without_enough_records.txt", "w") as file:
    file.write("\n".join(symbols_without_enough_records))
