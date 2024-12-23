import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import r2_score

# Defina o símbolo do estoque aqui
symbol = 'TAEE4'

df = pd.read_csv("stock_v5.csv")

df = df[df['Symbol'] == symbol]

df.drop('Symbol', axis=1, inplace=True)

# Criando novos campos de médias móveis
df['mm7'] = df['Adj Close'].rolling(7).mean()
df['mm21d'] = df['Adj Close'].rolling(21).mean()

# Vou colar um dia pra frente o df porque senão eu vou ver o valor da previsão já dado,
# ou seja vou rodar o modelo já sabendo o valor kkkk
df['Adj Close'] = df['Adj Close'].shift(-1)

# Vamos apagar os dados nulos porque o modelo não vai ler essas infos NaN
df.dropna(inplace=True)

# Verificando quantidade de linhas
qtd_linhas = len(df)
qtd_linhas_treino = round(0.50 * qtd_linhas)
qtd_linhas_teste = 30  # Reduzi para 30 para melhor visualização dos gráficos

info = (
    f"linhas treino= 0:{qtd_linhas_treino}"
    f" linhas teste= {qtd_linhas_treino}:{qtd_linhas_teste + qtd_linhas_treino - 1}"
)

df['Date'] = df['Date'].str.split(' ').str[0]

# Converter a coluna 'Date' para o formato de data "%Y-%m-%d"
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
df.index = pd.to_datetime(df['Date'])

df_completo = df

# Separando as features e labels
features = df.drop(['Close', 'Adj Close', 'Date'], axis=1)
labels = df['Adj Close']

# Agora vamos escolher as melhores variáveis para nossa base de dados com Kbest
features_list = ('Open', 'High', 'Low', 'Volume', 'mm7', 'mm21d')

k_best_features = SelectKBest(k='all')
k_best_features.fit_transform(features, labels)
k_best_features_scores = k_best_features.scores_
raw_pairs = zip(features_list[1:], k_best_features_scores)
ordered_pairs = list(reversed(sorted(raw_pairs, key=lambda x: x[1])))

k_best_features_final = dict(ordered_pairs[:15])
best_features = k_best_features_final.keys()

# Separa os dados de treino teste e validação
X_train = features[:qtd_linhas_treino]
X_test = features[qtd_linhas_treino:qtd_linhas_teste + qtd_linhas_treino]
y_train = labels[:qtd_linhas_treino]
y_test = labels[qtd_linhas_treino:qtd_linhas_teste + qtd_linhas_treino]

print(len(X_train), len(y_train))
print(len(X_test), len(y_test))

# Normalizando os dados de entrada (features)
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

# Treinamento usando regressão linear
lr = linear_model.LinearRegression()
lr.fit(X_train_scale, y_train)
pred = lr.predict(X_test_scale)

# Calculando o coeficiente de determinação
cd = r2_score(y_test, pred)
print(f'Coeficiente de determinação: {cd * 100:.2f}')

# Previsão para todo o conjunto de dados
previsao = scaler.transform(features)
pred = lr.predict(previsao)

# Criação de um novo DataFrame para os valores previstos
df_previsao = df_completo.copy()
df_previsao['Adj Close'] = pred
df_previsao['Close'] = pred
df_previsao['Open'] = df_previsao['Close'].shift(1)
df_previsao['High'] = df_previsao[['Open', 'Close']].max(axis=1)
df_previsao['Low'] = df_previsao[['Open', 'Close']].min(axis=1)
df_previsao.dropna(inplace=True)

# Selecionando apenas os últimos 60 registros para visualização
df_completo = df_completo.tail(60)
df_previsao = df_previsao.tail(60)

print("Datas dos últimos 60 registros:")
print(df_completo['Date'])

# Ajustando o tamanho da figura para maior detalhamento
plt.rcParams['figure.figsize'] = (40, 20)

# Criando um dataframe para o gráfico de candlestick original
candlestick_df = df_completo[['Open', 'High', 'Low', 'Close', 'Volume']]
candlestick_df.index = df_completo.index

# Criando um dataframe para o gráfico de candlestick previsto
candlestick_df_previsao = df_previsao[['Open', 'High', 'Low', 'Close', 'Volume']]
candlestick_df_previsao.index = df_previsao.index

# Plotando os gráficos de candlestick separadamente e salvando como PNG

# Gráfico de candlestick com média móvel
mpf.plot(candlestick_df, type='candle', style='charles', volume=True, ylabel='Preço', show_nontrading=False,
         title=f'Últimos 60 registros - Candlestick - {symbol} (Original - Com Média Móvel)', mav=(7, 21),
         savefig=f'candle_com_mm_detalhado_{symbol}.png', tight_layout=True)
plt.show()

# Gráfico de candlestick sem média móvel
mpf.plot(candlestick_df, type='candle', style='charles', volume=True, ylabel='Preço', show_nontrading=False,
         title=f'Últimos 60 registros - Candlestick - {symbol} (Original - Sem Média Móvel)',
         savefig=f'candle_sem_mm_detalhado_{symbol}.png', tight_layout=True)
plt.show()

# Gráfico de candlestick previsto com média móvel
mpf.plot(candlestick_df_previsao, type='candle', style='charles', volume=True, ylabel='Preço', show_nontrading=False,
         title=f'Últimos 60 registros - Candlestick - {symbol} (Previsto - Com Média Móvel)', mav=(7, 21),
         savefig=f'candle_com_mm_detalhado_{symbol}_previsto.png', tight_layout=True)
plt.show()

# Gráfico de candlestick previsto sem média móvel
mpf.plot(candlestick_df_previsao, type='candle', style='charles', volume=True, ylabel='Preço', show_nontrading=False,
         title=f'Últimos 60 registros - Candlestick - {symbol} (Previsto - Sem Média Móvel)',
         savefig=f'candle_sem_mm_detalhado_{symbol}_previsto.png', tight_layout=True)
plt.show()
