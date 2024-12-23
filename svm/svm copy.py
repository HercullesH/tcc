import pandas as pd
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 

# For data manipulation 
import numpy as np 

# To plot 
import matplotlib.pyplot as plt

# Carregar o arquivo CSV em um DataFrame
df = pd.read_csv("stock.csv")

df = df[df['Symbol'] == 'MMXM3']
print(df.head())
# Remover informações de fuso horário (assumindo que não são necessárias)
df['Date'] = df['Date'].str.split(' ').str[0]

# Converter a coluna 'Date' para o formato de data "%Y-%m-%d"
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
df.index = pd.to_datetime(df['Date'])

# drop The original date column
df = df.drop(['Date'], axis='columns')

# Filtrar para incluir apenas as ações com o símbolo 'MMXM3'


df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low
  
# Store all predictor variables in a variable X
X = df[['Open-Close', 'High-Low']]

# Target variables
y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

split_percentage = 0.8
split = int(split_percentage * len(df))
  
# Train data set
X_train = X[:split]
y_train = y[:split]
  
# Test data set
X_test = X[split:]
y_test = y[split:]

# Support vector classifier
cls = SVC().fit(X_train, y_train)

df['Predicted_Signal'] = cls.predict(X)

df['Return'] = df.Close.pct_change()
# Calculate strategy returns
df['Strategy_Return'] = df.Return * df.Predicted_Signal.shift(1)

# Calculate Cumulative returns
df['Cum_Ret'] = df['Return'].cumsum()

# Plot Strategy Cumulative returns 
df['Cum_Strategy'] = df['Strategy_Return'].cumsum()

plt.plot(df['Cum_Ret'], color='red', label='Cumulative Return')
plt.plot(df['Cum_Strategy'], color='blue', label='Cumulative Strategy Return')
plt.legend()

# Salvar o gráfico em um arquivo chamado "herculles.png"
plt.savefig("herculles.png")

# Exibir o gráfico na tela
plt.show()