import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

# Função para aplicar filtros passa-banda e obter o envelope do sinal
def process_emg_signal(signal, lowcut=20.0, highcut=450.0, fs=1000.0, order=4):
    # Filtro passa-banda
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    
    # Retificação
    rectified_signal = np.abs(filtered_signal)
    
    # Envelope usando a transformada de Hilbert
    envelope = np.abs(hilbert(rectified_signal))
    
    return envelope

# Função para processar e plotar os sinais EMG
def plot_emg_signals(data):
    # Nome das colunas
    colunas = ['RF', 'BF', 'VM', 'ST', 'FX']
    
    # Crie um vetor de tempo com base no número de amostras
    tempo = range(len(data))
    
    # Inicialize a figura
    plt.figure(figsize=(15, 10))
    
    # Processa e plota cada sinal EMG
    for i in range(4):
        signal = data[colunas[i]]
        envelope = process_emg_signal(signal)
        plt.subplot(5, 1, i+1)
        plt.plot(tempo, envelope, label=f'Envelope {colunas[i]}')
        plt.xlabel('Tempo')
        plt.ylabel('Microvolts')
        plt.legend()
    
    # Plote os valores da angulação da junta do joelho
    plt.subplot(5, 1, 5)
    plt.plot(tempo, data['FX'], label='Flexo Extension', color='red')
    plt.xlabel('Tempo')
    plt.ylabel('Graus')
    plt.legend()
    
    # Ajusta o layout para evitar sobreposição de gráficos
    plt.tight_layout()
    plt.show()

# Leia os dados do arquivo CSV
data = pd.read_csv('data/1Nmar.csv')

# Renomeia as colunas para facilitar a manipulação
data.columns = ['RF', 'BF', 'VM', 'ST', 'FX']

# Processa e plota os sinais EMG
plot_emg_signals(data)