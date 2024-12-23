import pandas as pd

# Carregar os datasets
ipca = pd.read_csv('ipca.csv')
stock_v4 = pd.read_csv('stock_v4.csv')

# Converter a coluna Date para datetime para extrair ano e mês
stock_v4['Date'] = pd.to_datetime(stock_v4['Date'])
stock_v4['ano'] = stock_v4['Date'].dt.year
stock_v4['mes'] = stock_v4['Date'].dt.strftime('%b').str.upper()

# Renomear coluna "mes" do ipca para coincidir com o formato
mes_dict = {
    'JAN': 'JAN', 'FEV': 'FEB', 'MAR': 'MAR', 'ABR': 'APR', 'MAI': 'MAY', 
    'JUN': 'JUN', 'JUL': 'JUL', 'AGO': 'AUG', 'SET': 'SEP', 'OUT': 'OCT', 
    'NOV': 'NOV', 'DEZ': 'DEC'
}
ipca['mes'] = ipca['mes'].map(mes_dict)

# Realizar o merge temporário
merged_data = pd.merge(stock_v4, ipca[['ano', 'mes', 'no_mes', '12_meses']], on=['ano', 'mes'], how='left')

# Remover as colunas de ano e mês temporárias
merged_data.drop(columns=['ano', 'mes'], inplace=True)

# Salvar o resultado no novo arquivo
merged_data.to_csv('stock_v5.csv', index=False)

print("Arquivo stock_v5.csv criado com sucesso!")
