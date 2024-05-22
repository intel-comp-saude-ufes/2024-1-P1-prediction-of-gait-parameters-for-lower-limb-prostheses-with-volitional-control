import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf

# Função para aplicar filtros passa-banda e obter o envelope do sinal
def process_emg_signal(signal, lowcut=20.0, highcut=300.0, fs=1000.0, order=4):
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

# Leia os dados do arquivo CSV
data = pd.read_csv('data/1Nmar.csv')

# Renomeia as colunas para facilitar a manipulação
data.columns = ['RF', 'BF', 'VM', 'ST', 'FX']

# Processa os sinais EMG para obter os envelopes
for col in ['RF', 'BF', 'VM', 'ST']:
    data[col] = process_emg_signal(data[col])

# Verifique as primeiras linhas do dataframe para entender a estrutura
print(data.head())

# Separe os dados em características (X) e alvo (y)
X = data.drop(columns=['FX'])
y = data['FX']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Definir o modelo
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(20, activation='linear'),
    tf.keras.layers.Dense(10, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Compilar o modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='mean_squared_error')

# Treinar o modelo
model.fit(X_scaled, y_scaled, epochs=100, verbose=0)

# Fazer previsões
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

# Calcule métricas de avaliação
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Imprima as métricas
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2%}")
print(f"R^2 Score: {r2:.2f}")

# Função para plotar os sinais EMG e a angulação do joelho
def plot_emg_signals(data):
    # Nome das colunas
    colunas = ['RF', 'BF', 'VM', 'ST', 'FX']
    
    # Crie um vetor de tempo com base no número de amostras
    tempo = range(len(data))
    
    # Inicialize a figura
    plt.figure(figsize=(15, 10))
    
    # Processa e plota cada sinal EMG
    for i in range(4):
        plt.subplot(5, 1, i+1)
        plt.plot(tempo, data[colunas[i]], label=f'Envelope {colunas[i]}')
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

# Plote os sinais processados e a angulação da junta do joelho
plot_emg_signals(data)

# Função para plotar a angulação real versus predita
def plot_real_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(y_true)), y_true, label='Real', color='blue')
    plt.plot(range(len(y_pred)), y_pred, label='Predito', color='red', linestyle='dashed')
    plt.xlabel('Amostras')
    plt.ylabel('Angulação do Joelho (Graus)')
    plt.legend()
    plt.title('Comparação entre Angulação Real e Predita')
    plt.show()

# Plote a comparação entre angulação real e predita
plot_real_vs_predicted(y, y_pred)
