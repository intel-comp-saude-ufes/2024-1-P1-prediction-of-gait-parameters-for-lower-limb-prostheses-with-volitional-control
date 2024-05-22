import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf

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

# Leia os dados do arquivo CSV
data = pd.read_csv('data/1Nmar.csv')

# Verifique as primeiras linhas do dataframe para entender a estrutura
print(data.head())

# Renomeia as colunas para facilitar a manipulação
data.columns = ['RF', 'BF', 'VM', 'ST', 'FX']

# Processa os sinais EMG para obter os envelopes
for col in ['RF', 'BF', 'VM', 'ST']:
    data[col] = process_emg_signal(data[col])

# Separe os dados em características (X) e alvo (y)
X = data.drop(columns=['FX'])
y = data['FX']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=20)

# Definir o modelo
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=100, verbose=0)

# Fazer previsões
y_pred_train_scaled = model.predict(X_train)
y_pred_train = scaler.inverse_transform(y_pred_train_scaled).flatten()

y_pred_test_scaled = model.predict(X_test)
y_pred_test = scaler.inverse_transform(y_pred_test_scaled).flatten()

# Avaliar o desempenho do modelo nos dados de treino
mae_mlnn_train = mean_absolute_error(y_train, y_pred_train)
mse_mlnn_train = mean_squared_error(y_train, y_pred_train)
rmse_mlnn_train = np.sqrt(mse_mlnn_train)
r2_mlnn_train = r2_score(y_train, y_pred_train)

# Avaliar o desempenho do modelo nos dados de teste
mae_mlnn_test = mean_absolute_error(y_test, y_pred_test)
mse_mlnn_test = mean_squared_error(y_test, y_pred_test)
rmse_mlnn_test = np.sqrt(mse_mlnn_test)
r2_mlnn_test = r2_score(y_test, y_pred_test)

# Imprimir as métricas
print("Resultados com Rede Neural Multicamadas (MLNN):")
print("Dados de Treino:")
print(f"Mean Absolute Error: {mae_mlnn_train:.2f}")
print(f"Mean Squared Error: {mse_mlnn_train:.2f}")
print(f"Root Mean Squared Error: {rmse_mlnn_train:.2f}")
print(f"R^2 Score: {r2_mlnn_train:.2f}")
print("\nDados de Teste:")
print(f"Mean Absolute Error: {mae_mlnn_test:.2f}")
print(f"Mean Squared Error: {mse_mlnn_test:.2f}")
print(f"Root Mean Squared Error: {rmse_mlnn_test:.2f}")
print(f"R^2 Score: {r2_mlnn_test:.2f}")

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
# plot_emg_signals(data)

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
y_test = scaler.inverse_transform(y_test).flatten()
plot_real_vs_predicted(y_test, y_pred_test)
