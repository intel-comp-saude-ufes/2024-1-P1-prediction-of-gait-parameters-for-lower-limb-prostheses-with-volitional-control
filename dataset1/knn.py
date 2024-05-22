import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


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



# Função para aplicar filtros passa-banda e obter o envelope do sinal
def process_emg_signal(signal, lowcut_bp=20.0, highcut_bp=300.0, fs=1000.0, order_bp=4, freq_notch=60.0, q_notch=30.0, lowcut_lp=2.0):
    # Filtro passa-banda
    nyquist = 0.5 * fs
    low_bp = lowcut_bp / nyquist
    high_bp = highcut_bp / nyquist
    b_bp, a_bp = butter(order_bp, [low_bp, high_bp], btype='band')
    filtered_signal_bp = filtfilt(b_bp, a_bp, signal)
    
    # Filtro notch
    freq_notch /= nyquist
    b_notch, a_notch = iirnotch(freq_notch, q_notch)
    filtered_signal_notch = filtfilt(b_notch, a_notch, filtered_signal_bp)
    
    # Retificação
    rectified_signal = np.abs(filtered_signal_notch)
    
    # Filtro passa-baixa
    low_lp = lowcut_lp / nyquist
    b_lp, a_lp = butter(4, low_lp, btype='low')
    filtered_signal_lp = filtfilt(b_lp, a_lp, rectified_signal)
    
    return filtered_signal_lp



# Leia os dados do arquivo CSV
data = pd.read_csv('data/1Nmar.csv')

# Renomeia as colunas para facilitar a manipulação
data.columns = ['RF', 'BF', 'VM', 'ST', 'FX']

# Processa os sinais EMG para obter os envelopes
for col in ['RF', 'BF', 'VM', 'ST']:
    data[col] = process_emg_signal(data[col])

# Verifique as primeiras linhas do dataframe para entender a estrutura
# print(data.head())


plot_emg_signals(data)

# Separe os dados em características (X) e alvo (y)
X = data.drop(columns=['ST', 'FX'])
y = data['FX']

# Inicialize o modelo KNN para regressão
knn_model = KNeighborsRegressor(n_neighbors=500, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
knn_model.fit(X, y)



data_test = pd.read_csv('data/2Nmar.csv')

# Renomeia as colunas para facilitar a manipulação
data_test.columns = ['RF', 'BF', 'VM', 'ST', 'FX']

# Processa os sinais EMG para obter os envelopes
for col in ['RF', 'BF', 'VM', 'ST']:
    data_test[col] = process_emg_signal(data_test[col])

# Verifique as primeiras linhas do dataframe para entender a estrutura
# print(data_test.head())

plot_emg_signals(data_test)

# Separe os dados em características (X) e alvo (y)
X_test = data_test.drop(columns=['ST', 'FX'])
y_test = data_test['FX']

y_pred = knn_model.predict(X_test)

# Calcule métricas de avaliação
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprima as métricas
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2%}")
print(f"R^2 Score: {r2:.2f}")


# Plote a comparação entre angulação real e predita
plot_real_vs_predicted(y_test, y_pred)