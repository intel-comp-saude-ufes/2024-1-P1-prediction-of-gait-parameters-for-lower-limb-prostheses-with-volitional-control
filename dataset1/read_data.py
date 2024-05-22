import pandas as pd
import matplotlib.pyplot as plt

# Leia os dados do arquivo CSV
data = pd.read_csv('data/1Nmar.csv')

# Verifique as primeiras linhas do dataframe para entender a estrutura
print(data.head())

# Assumindo que as colunas são nomeadas como "Recto Femoral", "Biceps Femoral", "Vasto Medial", "EMG Semitendinoso" e "Flexo Extension"
# Se os nomes das colunas forem diferentes, atualize essa lista conforme necessário.
colunas = ['Recto Femoral', 'Biceps Femoral', 'Vasto Medial', 'EMG Semitendinoso', 'Flexo Extension']

# Renomeia as colunas para facilitar a manipulação (caso não estejam já nomeadas assim)
data.columns = colunas

# Troca o sinal dos valores da última coluna
# data['Flexo Extension'] = -data['Flexo Extension']

# Crie um vetor de tempo com base no número de amostras
tempo = range(len(data))
print(len(data))

print(len(data['Flexo Extension']))

# Plote os valores EMG para cada músculo
plt.figure(figsize=(15, 10))

for i in range(4):
    plt.subplot(5, 1, i+1)
    plt.plot(tempo, data[colunas[i]], label=colunas[i])
    plt.xlabel('Tempo')
    plt.ylabel('Microvolts')
    plt.legend()

# Plote os valores da angulação da junta do joelho
plt.subplot(5, 1, 5)
plt.plot(tempo, data['Flexo Extension'], label='Flexo Extension', color='red')
plt.xlabel('Tempo')
plt.ylabel('Graus')
plt.legend()

# Ajusta o layout para evitar sobreposição de gráficos
plt.tight_layout()
plt.show()