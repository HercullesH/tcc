import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (18, 8)
df = pd.read_csv("stock.csv")

df = df[df['Symbol'] == 'TAEE4']

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
qtd_linhas_teste = 300
qtd_linhas_validacao = qtd_linhas - 1

info = (
    f"linhas treino= 0:{qtd_linhas_treino}"
    f" linhas teste= {qtd_linhas_treino}:{qtd_linhas_treino + qtd_linhas_teste - 1}"
    f" linhas validação= {qtd_linhas_validacao}"
)

df['Date'] = df['Date'].str.split(' ').str[0]

# Converter a coluna 'Date' para o formato de data "%Y-%m-%d"
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
df.index = pd.to_datetime(df['Date'])

df_completo = df

# Separando as features e labels
features = df.drop(['Close', 'Adj Close', 'Date'],axis=1)
labels = df['Adj Close']

# Agora vamos escolher as melhores variáveis para nossa base de dados com Kbest
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

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
X_test = features[qtd_linhas_treino:qtd_linhas_treino + qtd_linhas_teste - 1]
y_train = labels[:qtd_linhas_treino]
y_test = labels[qtd_linhas_treino:qtd_linhas_treino + qtd_linhas_teste - 1]

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
cd = r2_score(y_test, pred)
f'Coeficiente de determinação: {cd * 100:.2f}'

valor_novo = features
previsao = scaler.transform(valor_novo)
pred = lr.predict(previsao)

Date_full = df_completo['Date']
Date = Date_full

res_full = df_completo['Adj Close']
res = res_full

df = pd.DataFrame({'Date': Date, 'real': res, 'previsao': pred})
df.set_index('Date', inplace=True)
df.head()
plt.title("Preço da Ação R$")
plt.plot(df["real"],label = "Real", color = "blue", marker = 'o')
plt.plot(df["previsao"],label = "Previsto", color = "red")
plt.xlabel("Data")
plt.ylabel("Valor fechamento")
plt.legend()
plt.savefig("TAEE4.png")
for index, row in df.iterrows():
    df.loc[index,'diferenca'] = (df.loc[index,'real'] - df.loc[index,'previsao']);

df.head()
print(df.head(100))

