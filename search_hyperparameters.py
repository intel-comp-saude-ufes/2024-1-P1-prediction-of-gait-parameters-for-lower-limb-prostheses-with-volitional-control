import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

from utils import load_data


if __name__ == '__main__':
    train_folder = 'data/P5'
    train_files = ['T1.txt', 'T2.txt', 'T3.txt', 'T4.txt', 'T5.txt', 'T6.txt', 'T7.txt', 'T8.txt', 'T9.txt']
    metadata, data_angles, data_emg_envelope, data_emg_filtered, data_grf, data_torques, data_torques_norm = load_data(train_folder, train_files)

    St = 'St1'
    data_emg_columns = [St+'_VL', St+'_BF']
    data_torques_columns = [St+'_Pelvis_X', St+'_Pelvis_Y', St+'_Pelvis_Z',
                            St+'_Hip_X',    St+'_Hip_Y',    St+'_Hip_Z',
                            St+'_Knee_X',   St+'_Knee_Y',   St+'_Knee_Z',
                            St+'_Ankle_X',  St+'_Ankle_Y',  St+'_Ankle_Z']
    data_grf_columns = [St+'_GRF_X', St+'_GRF_Y', St+'_GRF_Z']
    data_angles_columns = [St+'_Knee_X']

    data_angles = data_angles[data_angles_columns]
    data_emg_envelope = data_emg_envelope[data_emg_columns]
    data_emg_filtered = data_emg_filtered[data_emg_columns]
    data_grf = data_grf[data_grf_columns]
    data_torques = data_torques[data_torques_columns]
    data_torques_norm = data_torques_norm[data_torques_columns]

    X_train = pd.concat([data_emg_envelope], axis=1) # Concatenate the input model data
    y_train = data_angles # Get the target data

    model = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)

    grid_params = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [30, 40, 50],
        'p': [1, 2],
        'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
    }

    grid_search = GridSearchCV(model, grid_params, scoring='r2', cv=5)
    grid_search.fit(X_train, y_train)
    print("Grid search results:")
    print(f'Best score: {grid_search.best_score_}')
    print(f'Best params: {grid_search.best_params_}')
    print(f'Best estimator: {grid_search.best_estimator_}')
    print("\n")