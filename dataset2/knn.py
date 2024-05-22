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



# Leia os dados do arquivo CSV
data_emg = pd.read_csv('data/train/emg_filtered.csv')
data_torques = pd.read_csv('data/train/torques.csv')
data_grf = pd.read_csv('data/train/grf.csv')
X = pd.concat([data_emg, data_torques, data_grf], axis=1)

data_angles = pd.read_csv('data/train/angles.csv')
y = data_angles['St1_Knee_X']

# Inicialize o modelo KNN para regressão
knn_model = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
knn_model.fit(X, y)



# Leia os dados do arquivo CSV
data_emg_test = pd.read_csv('data/test/emg_filtered.csv')
data_torques_test = pd.read_csv('data/test/torques.csv')
data_grf_test = pd.read_csv('data/test/grf.csv')
X_test = pd.concat([data_emg_test, data_torques_test, data_grf_test], axis=1)

data_angles_test = pd.read_csv('data/test/angles.csv')
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