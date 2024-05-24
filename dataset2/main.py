########################################################################################################################
############################################## IMPORTS #################################################################
########################################################################################################################
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor



########################################################################################################################
############################################## DEFINING FUNCTIONS ######################################################
########################################################################################################################

def load_data(train_folder):
    '''
    Function to load train/test data from a folder

    INPUT:
        train_folder (str): Path to the folder containing the train/test data

    OUTPUT:
        data_emg (pd.DataFrame): EMG data
        data_torques (pd.DataFrame): Torques data
        data_grf (pd.DataFrame): GRF data
        data_angles (pd.DataFrame): Angles data
    '''

    # Counters to check if the data was already loaded at least once or nots
    count_emg = 0
    count_torques = 0
    count_grf = 0
    count_angles = 0

    for root, dirs, files in os.walk(train_folder):
        for file in files:
            # Pass from each file in the folder
            if file.endswith('.csv'):
                file_path = os.path.join(root, file) # Get the hole file path
                if 'emg_filtered' in file_path:
                    if count_emg == 0: # If the data was not loaded yet
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
        plt.subplot(5, 2, i)
        plt.plot(range(len(y_true)), y_true, label='Real', color='blue')
        plt.plot(range(len(y_pred)), y_pred, label=model_name, linestyle='dashed', color='red')
        plt.xlabel('Samples')
        plt.ylabel('Knee Angulation (Degrees)')
        plt.legend()
        plt.title(f'{model_name} - R^2 Score: {metrics[model_name]:.2f}')
    
    plt.tight_layout()
    plt.show()



########################################################################################################################
################################################## MAIN PROGRAM ########################################################
########################################################################################################################

if __name__ == '__main__':

    # Prepare the train data
    train_folder = 'data/train'
    data_emg, data_torques, data_grf, data_angles = load_data(train_folder)
    X = pd.concat([data_torques], axis=1) # Concatenate the input model data
    y = data_angles['St1_Knee_X'] # Get the target data

    # Prepare the test data
    test_folder = 'data/test'
    data_emg_test, data_torques_test, data_grf_test, data_angles_test = load_data(test_folder)
    X_test = pd.concat([data_torques], axis=1) # Concatenate the input model data
    y_test = data_angles_test['St1_Knee_X'] # Get the target data

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
    }

    # Dictionary to store the predictions and metrics to plot
    predictions = {}
    metrics = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}")
        model.fit(X, y)
        y_pred = model.predict(X_test)
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