
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import PolynomialFeatures




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


# Função para carregar dados de treino de todas as subpastas
def load_train_data(train_folder):
    count_emg = 0
    count_torques = 0
    count_grf = 0
    count_angles = 0

    for root, dirs, files in os.walk(train_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f'Loading file: {file_path}')
                if 'emg_filtered' in file_path:
                    if count_emg == 0:
                        data_emg = pd.read_csv(file_path)
                    else:
                        data_emg = pd.concat([data_emg, pd.read_csv(file_path)], axis=0)
                    count_emg += 1
                elif 'torques.' in file_path: # '.' to not include 'torques_norm.csv'
                    if count_torques == 0:
                        data_torques = pd.read_csv(file_path)
                    else:
                        data_torques = pd.concat([data_torques, pd.read_csv(file_path)], axis=0)
                    count_torques += 1
                elif 'grf' in file_path:
                    if count_grf == 0:
                        data_grf = pd.read_csv(file_path)
                    else:
                        data_grf = pd.concat([data_grf, pd.read_csv(file_path)], axis=0)
                    count_grf += 1
                elif 'angles' in file_path:
                    if count_angles == 0:
                        data_angles = pd.read_csv(file_path)
                    else:
                        data_angles = pd.concat([data_angles, pd.read_csv(file_path)], axis=0)
                    count_angles += 1
            
    return data_emg, data_torques, data_grf, data_angles

# Leia os dados de treino de todas as subpastas
train_folder = 'data/train'
data_emg, data_torques, data_grf, data_angles = load_train_data(train_folder)

print(data_emg.shape)
print(data_torques.shape)
print(data_grf.shape)
print(data_angles.shape)


# Concatenar todos os dados de treino em um único DataFrame
X = pd.concat([data_emg, data_torques, data_grf], axis=1)
y = data_angles['St1_Knee_X']


# Inicialize o modelo KNN para regressão
knn_model = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
knn_model.fit(X, y)


# Leia os dados do arquivo CSV
data_emg_test = pd.read_csv('data/test/P1/V1/T3/emg_filtered.csv')
data_torques_test = pd.read_csv('data/test/P1/V1/T3/torques.csv')
data_grf_test = pd.read_csv('data/test/P1/V1/T3/grf.csv')
X_test = pd.concat([data_emg_test, data_torques_test, data_grf_test], axis=1)

data_angles_test = pd.read_csv('data/test/P1/V1/T3/angles.csv')
y = data_angles_test['St1_Knee_X']

y_pred = knn_model.predict(X_test)

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


# Plote a comparação entre angulação real e predita
plot_real_vs_predicted(y, y_pred)