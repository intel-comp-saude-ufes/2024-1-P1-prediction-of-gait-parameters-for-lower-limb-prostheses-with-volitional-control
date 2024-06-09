import pandas as pd
import matplotlib.pyplot as plt

# Leia os dados do arquivo CSV
data = pd.read_csv('data/P15/V1/R/Angles/T1.txt', delimiter='\t')

colunas = data.columns
colunas = ['St1_Knee_X']
qtd_colunas = len(colunas)



# Crie um vetor de tempo com base no número de amostras
tempo = range(len(data))

# Plote os valores EMG para cada músculo
plt.figure(figsize=(15, 10))

for i in range(qtd_colunas):
    plt.subplot(qtd_colunas, 1, i+1)
    plt.plot(tempo, data[colunas[i]], label=colunas[i])
    plt.xlabel('Tempo')
    plt.ylabel('degress')
    plt.legend()

# Ajusta o layout para evitar sobreposição de gráficos
plt.tight_layout()
plt.show()