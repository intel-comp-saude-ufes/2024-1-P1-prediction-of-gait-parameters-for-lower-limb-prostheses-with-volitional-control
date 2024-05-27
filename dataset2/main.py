########################################################################################################################
############################################## IMPORTS #################################################################
########################################################################################################################
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate as interp
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter

from scipy.signal import butter, filtfilt, iirnotch
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

import seaborn as sns
from lazypredict.Supervised import LazyRegressor

from animation import create_animation



########################################################################################################################
############################################## DEFINING FUNCTIONS ######################################################
########################################################################################################################

def load_data(folder_path, files_to_load):
    '''
    Function to load train/test data from a folder

    INPUT:
        folder_path (str): Path to the folder containing the train/test data

    OUTPUT:
        data_angles (pd.DataFrame): Angles data
        data_emg_envelope (pd.DataFrame): EMG envelope data
        data_emg_filtered (pd.DataFrame): EMG filtered data
        data_grf (pd.DataFrame): GRF data
        data_torques (pd.DataFrame): Torques data
        data_torques_norm (pd.DataFrame): Normalized torques data
    '''

    # Counters to check if the data was already loaded at least once or nots
    count_angles = 0
    count_emg_envelope = 0
    count_emg_filtered = 0
    count_grf = 0
    count_torques = 0
    count_torques_norm = 0

    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            if dir == 'Angles':
                # Read the files inside this folder
                for file in os.listdir(os.path.join(root, dir)):
                    if file in files_to_load:
                        if count_angles == 0:
                            data_angles = pd.read_csv(os.path.join(root, dir, file), delimiter='\t')
                            count_angles += 1
                        else:
                            data_angles = pd.concat([data_angles, pd.read_csv(os.path.join(root, dir, file), delimiter='\t')], axis=0)

            elif dir == 'EMG envelope':
                # Read the files inside this folder
                for file in os.listdir(os.path.join(root, dir)):
                    if file in files_to_load:
                        if count_emg_envelope == 0:
                            data_emg_envelope = pd.read_csv(os.path.join(root, dir, file), delimiter='\t')
                            count_emg_envelope += 1
                        else:
                            data_emg_envelope = pd.concat([data_emg_envelope, pd.read_csv(os.path.join(root, dir, file), delimiter='\t')], axis=0)

            elif dir == 'EMG filtered':
                # Read the files inside this folder
                for file in os.listdir(os.path.join(root, dir)):
                    if file in files_to_load:
                        if count_emg_filtered == 0:
                            data_emg_filtered = pd.read_csv(os.path.join(root, dir, file), delimiter='\t')
                            count_emg_filtered += 1
                        else:
                            data_emg_filtered = pd.concat([data_emg_filtered, pd.read_csv(os.path.join(root, dir, file), delimiter='\t')], axis=0)

            elif dir == 'GRF':
                # Read the files inside this folder
                for file in os.listdir(os.path.join(root, dir)):
                    if file in files_to_load:
                        if count_grf == 0:
                            data_grf = pd.read_csv(os.path.join(root, dir, file), delimiter='\t')
                            count_grf += 1
                        else:
                            data_grf = pd.concat([data_grf, pd.read_csv(os.path.join(root, dir, file), delimiter='\t')], axis=0)

            elif dir == 'Torques':
                # Read the files inside this folder
                for file in os.listdir(os.path.join(root, dir)):
                    if file in files_to_load:
                        if count_torques == 0:
                            data_torques = pd.read_csv(os.path.join(root, dir, file), delimiter='\t')
                            count_torques += 1
                        else:
                            data_torques = pd.concat([data_torques, pd.read_csv(os.path.join(root, dir, file), delimiter='\t')], axis=0)

            elif dir == 'Torques_Norm':
                # Read the files inside this folder
                for file in os.listdir(os.path.join(root, dir)):
                    if file in files_to_load:
                        if count_torques_norm == 0:
                            data_torques_norm = pd.read_csv(os.path.join(root, dir, file), delimiter='\t')
                            count_torques_norm += 1
                        else:
                            data_torques_norm = pd.concat([data_torques_norm, pd.read_csv(os.path.join(root, dir, file), delimiter='\t')], axis=0)
            
    return data_angles, data_emg_envelope, data_emg_filtered, data_grf, data_torques, data_torques_norm


def plot_comparisons(y_true, predictions, metrics):
    '''
    Function to plot the comparison between the real values and the predicted values

    INPUT:
        y_true (pd.Series): Real values
        predictions (dict): Dictionary containing the predictions of each model
        metrics (dict): Dictionary containing the R^2 score of each model

    OUTPUT:
        None
    '''

    plt.figure(figsize=(18, 10))
    
    for i, (model_name, y_pred) in enumerate(predictions.items(), 1):
        plt.subplot(6, 2, i)
        plt.plot(range(len(y_true)), y_true, label='Real', color='blue')
        plt.plot(range(len(y_pred)), y_pred, label=model_name, linestyle='dashed', color='red')
        plt.xlabel('Samples')
        plt.ylabel('Knee Ang (Deg)')
        plt.legend()
        plt.title(f'{model_name} - R^2 Score: {metrics[model_name]:.2f}')
    
    plt.tight_layout()
    plt.show()


# Curve smoothing functions
def moving_average(y_pred, window_size=5):
    y_pred = np.ravel(y_pred)
    pad_size = window_size // 2
    y_padded = np.pad(y_pred, pad_size, mode='edge')
    y_smooth = np.convolve(y_padded, np.ones(window_size)/window_size, mode='same')
    return y_smooth[pad_size:-pad_size]

def smooth_spline(y_pred, s=1):
    y_pred = np.ravel(y_pred)
    x = np.arange(len(y_pred))
    spline = interp.UnivariateSpline(x, y_pred, k=1, s=s)
    return spline(x)

def loess_smoothing(y_pred, frac=0.2):
    y_pred = np.ravel(y_pred)
    x = np.arange(len(y_pred))
    loess_result = lowess(y_pred, x, frac=frac)
    y_smooth = np.interp(x, loess_result[:, 0], loess_result[:, 1])
    return y_smooth

def kalman_filter(y_pred):
    y_pred = np.ravel(y_pred)
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    state_means, _ = kf.smooth(y_pred)
    return state_means.flatten()



########################################################################################################################
################################################## MAIN PROGRAM ########################################################
########################################################################################################################

if __name__ == '__main__':

    # Prepare the train data
    train_folder = 'data/P5'
    train_files = ['T1.txt', 'T2.txt', 'T3.txt', 'T4.txt', 'T5.txt', 'T6.txt', 'T7.txt', 'T8.txt', 'T9.txt']
    data_angles, data_emg_envelope, data_emg_filtered, data_grf, data_torques, data_torques_norm = load_data(train_folder, train_files)

    St = 'St1'

    # St1_VL	St2_VL	St1_BF	St2_BF	St1_TA	St2_TA	St1_GAL	St2_GAL
    # data_emg_columns = [St+'_VL', St+'_BF', St+'_TA', St+'_GAL']
    data_emg_columns = [St+'_VL', St+'_BF']

    # St1_Pelvis_X	St1_Pelvis_Y	St1_Pelvis_Z	St2_Pelvis_X	St2_Pelvis_Y	St2_Pelvis_Z    ...
    # St1_Hip_X	    St1_Hip_Y	    St1_Hip_Z	    St2_Hip_X	    St2_Hip_Y	    St2_Hip_Z	    ...
    # St1_Knee_X	St1_Knee_Y	    St1_Knee_Z	    St2_Knee_X	    St2_Knee_Y	    St2_Knee_Z      ...
    # St1_Ankle_X	St1_Ankle_Y	    St1_Ankle_Z	    St2_Ankle_X	    St2_Ankle_Y	    St2_Ankle_Z
    # data_torques_columns = [St+'_Pelvis_X', St+'_Pelvis_Y', St+'_Pelvis_Z',
    #                         St+'_Hip_X',    St+'_Hip_Y',    St+'_Hip_Z',
    #                         St+'_Knee_X',   St+'_Knee_Y',   St+'_Knee_Z',
    #                         St+'_Ankle_X',  St+'_Ankle_Y',  St+'_Ankle_Z']
    data_torques_columns = [St+'_Knee_X']
    
    # St1_GRF_X	    St1_GRF_Y	    St1_GRF_Z	    St2_GRF_X	    t2_GRF_Y	    St2_GRF_Z
    # data_grf_columns = [St+'_GRF_X', St+'_GRF_Y', St+'_GRF_Z']
    data_grf_columns = [St+'_GRF_X']

    # St1_Pelvis_X	St1_Pelvis_Y	St1_Pelvis_Z	St2_Pelvis_X	St2_Pelvis_Y	St2_Pelvis_Z    ...
    # St1_Hip_X	    St1_Hip_Y	    St1_Hip_Z	    St2_Hip_X	    St2_Hip_Y	    St2_Hip_Z       ...   
    # St1_Knee_X	St1_Knee_Y	    St1_Knee_Z	    St2_Knee_X	    St2_Knee_Y	    St2_Knee_Z      ...
    # St1_Ankle_X	St1_Ankle_Y	    St1_Ankle_Z	    St2_Ankle_X	    St2_Ankle_Y	    St2_Ankle_Z
    # data_angles_columns = [St+'_Knee_X']
    data_angles_columns = [St+'_Knee_X']

    data_angles = data_angles[data_angles_columns]
    data_emg_envelope = data_emg_envelope[data_emg_columns]
    data_emg_filtered = data_emg_filtered[data_emg_columns]
    data_grf = data_grf[data_grf_columns]
    data_torques = data_torques[data_torques_columns]
    data_torques_norm = data_torques_norm[data_torques_columns]


    X = pd.concat([data_emg_envelope], axis=1) # Concatenate the input model data
    y = data_angles # Get the target data


    # Prepare the test data
    test_folder = 'data/P5'
    test_files = ['T10.txt']
    data_angles_test, data_emg_envelope_test, data_emg_filtered_test, data_grf_test, data_torques_test, data_torques_norm_test = load_data(test_folder, test_files)

    data_angles_test = data_angles_test[data_angles_columns]
    data_emg_envelope_test = data_emg_envelope_test[data_emg_columns]
    data_emg_filtered_test = data_emg_filtered_test[data_emg_columns]
    data_grf_test = data_grf_test[data_grf_columns]
    data_torques_test = data_torques_test[data_torques_columns]
    data_torques_norm_test = data_torques_norm_test[data_torques_columns]

    X_test = pd.concat([data_emg_envelope_test], axis=1) # Concatenate the input model data
    y_test = data_angles_test # Get the target data

    # Defining the models
    models = {
        'KNN': KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None),

        'Decision Tree': DecisionTreeRegressor(criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0, monotonic_cst=None),

        'SVM': SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1),

        'Random Forest': RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None),
        
        'Gradient Boosting': GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0),

        'AdaBoost': AdaBoostRegressor(estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None),

        'Ridge': Ridge(alpha=1.0, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto', positive=False, random_state=None),

        'Lasso': Lasso(alpha=1.0, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'),

        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'),
        
        'Extra Trees Regressor' : ExtraTreesRegressor(n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None),

        'MLP Regressor' : MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000),

        'XGBRegressor' : XGBRegressor()
    }

    # Dictionary to store the predictions and metrics to plot after
    predictions = {}
    metrics = {}

    # Use lazy predict to get a holistc view about the result of a lot os models
    # lazy_model = LazyRegressor()
    # models_lazy, predictions_lazy = lazy_model.fit(X, X_test, y, y_test)
    # print(models_lazy)

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}")

        model.fit(X, y)
        y_pred = model.predict(X_test)

        # applyng smoth filter
        # y_pred = moving_average(y_pred, window_size=75)
        # y_pred = smooth_spline(y_pred)
        # y_pred = loess_smoothing(y_pred, frac=0.09)
        # y_pred = kalman_filter(y_pred)

        predictions[model_name] = y_pred

        # Calculate assessment metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics[model_name] = r2

        # Print the metrics
        print(f"{model_name} - Mean Absolute Error: {mae:.2f}")
        print(f"{model_name} - Mean Squared Error: {mse:.2f}")
        print(f"{model_name} - Root Mean Squared Error: {rmse:.2f}")
        print(f"{model_name} - Mean Absolute Percentage Error: {mape:.2%}")
        print(f"{model_name} - R^2 Score: {r2:.2f}")
        print() # Blank line

    plot_comparisons(y_test, predictions, metrics)

    # Finding the best model
    best_model = max(metrics, key=metrics.get)

    # Prepare the data to create the animation
    emg_anim = data_emg_envelope_test['St1_BF'].to_numpy().reshape(-1, 1)
    y_test_anim = y_test.to_numpy().ravel()

    # Run a animation with the best model    
    create_animation(emg_anim, y_test.to_numpy().flatten(), predictions[best_model].flatten())