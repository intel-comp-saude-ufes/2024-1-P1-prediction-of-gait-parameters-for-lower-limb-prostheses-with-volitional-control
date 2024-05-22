from scipy.signal import butter, filtfilt, iirnotch, hilbert, butter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit



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
        plt.plot(tempo, data[colunas[i]], label=f'{colunas[i]}')
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


# Função para segmentar os dados em janelas temporais e extrair características
def extract_features(data, window_size=100, overlap=50):
    features = []
    labels = []
    for i in range(0, len(data) - window_size + 1, overlap):
        window = data.iloc[i:i+window_size]
        features.append([
            np.mean(window['RF']),
            np.std(window['RF']),
            np.max(window['RF']),
            np.min(window['RF']),
            np.polyfit(np.arange(len(window)), window['RF'], 1)[0]
        ])
        labels.append(np.mean(window['FX']))  # Agora calculamos a média dos valores de FX na janela
    return np.array(features), np.array(labels)


def sliding_window_cv(X, y, n_splits=5, test_size=0.2):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_index, test_index in tscv.split(X):
        train_size = int(len(train_index) * (1 - test_size))
        train_indices = train_index[:train_size]
        val_indices = train_index[train_size:]
        splits.append((train_indices, val_indices, test_index))
    return splits


# Leia os dados do arquivo CSV
data = pd.read_csv('data/1Nmar.csv')

for col in ['RF', 'BF', 'VM', 'ST']:
    data[col] = process_emg_signal(data[col])

# plot_emg_signals(data)

# Segmentar os dados em janelas temporais e extrair características
window_size = 100  # Tamanho da janela em amostras
overlap = 50  # Quantidade de sobreposição entre as janelas em amostras
# X, y = extract_features(data, window_size=window_size, overlap=overlap)
X = data.drop(columns=['BF', 'VM', 'ST', 'FX'])
y = data['FX']

# Normalizar os dados
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# Exibir as características extraídas e seus rótulos
print("Características Extraídas:")
print(X)
print("Rótulos:")
print(y)


# Dividir os dados em janelas de tempo
splits = sliding_window_cv(X, y)

# Treinar e avaliar modelos em cada janela de tempo
for i, (train_index, val_index, test_index) in enumerate(splits):
    X_train, X_val, X_test = X[train_index], X[val_index], X[test_index]
    y_train, y_val, y_test = y[train_index], y[val_index], y[test_index]
    
    # Treinar e avaliar modelos usando X_train, y_train, X_val, y_val, X_test, y_test
    # Definir os hiperparâmetros a serem testados
    param_grid = {
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Instanciar o modelo de árvore de decisão
    tree_model = DecisionTreeRegressor(random_state=42)

    # Instanciar o objeto GridSearchCV
    grid_search = GridSearchCV(estimator=tree_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

    # Realizar a busca em grade
    grid_search.fit(X_train, y_train)

    # Melhores hiperparâmetros encontrados
    best_params = grid_search.best_params_
    print("Melhores Hiperparâmetros:")
    print(best_params)

    # Melhor modelo encontrado
    best_model = grid_search.best_estimator_

    # Fazer previsões no conjunto de teste com o melhor modelo
    y_pred_test_best = best_model.predict(X_test)

    # Avaliar o desempenho do melhor modelo
    mse_test_best = mean_squared_error(y_test, y_pred_test_best)
    r2_test_best = r2_score(y_test, y_pred_test_best)

    # Exibir as métricas de avaliação do melhor modelo
    print("Desempenho do Melhor Modelo:")
    print(f"Erro Quadrático Médio (MSE) - Teste: {mse_test_best:.2f}")
    print(f"Coeficiente de Determinação (R^2) - Teste: {r2_test_best:.2f}")
    
    print(f"Janela de tempo {i+1}:")
    print(f"Tamanho dos conjuntos de treino, validação e teste: {len(train_index)}, {len(val_index)}, {len(test_index)}")

    print("\n\n")