from itertools import combinations
import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from output_filters import loess_smoothing
from utils import load_data
from lazypredict.Supervised import LazyRegressor

r2_vec = {}
r2_vec['KNN'] = 0
r2_vec['Random Forest'] = 0
r2_vec['Gradient Boosting'] = 0
r2_vec['AdaBoost'] = 0
r2_vec['Extra Trees Regressor'] = 0
r2_vec['MLP Regressor'] = 0
r2_vec['XGBRegressor'] = 0
r2_vec['Voting Regressor'] = 0

mae_vec = {}
mae_vec['KNN'] = 0
mae_vec['Random Forest'] = 0
mae_vec['Gradient Boosting'] = 0
mae_vec['AdaBoost'] = 0
mae_vec['Extra Trees Regressor'] = 0
mae_vec['MLP Regressor'] = 0
mae_vec['XGBRegressor'] = 0
mae_vec['Voting Regressor'] = 0
    

def train_and_validate_patients(data_path, train_patients, val_patients):
    '''
    Function to train a model and validate it with validation patients

    INPUT:
        data_path (str): Path to the data folder
        train_patients (list): Patients to use for training
        val_patients (str): Patients to use for validation

    OUTPUT:
        best_model (Model): Best model to use
    '''

    # Load the train data
    for patient in train_patients:
        # Get all test files of the patient
        data_path = f'data/{patient}'
        print(data_path)
        data_path_patient = f'{data_path}/V1/R/Angles'
        data_files = [f for f in os.listdir(data_path_patient) if f.endswith('.txt')]
        print(data_files)

        # Load the data
        metadata, data_angles, data_emg_envelope, data_emg_filtered, data_grf, data_torques, data_torques_norm = load_data(data_path, data_files)

        St = 'St1'
        data_angles_columns = [St+'_Knee_X']
        data_emg_columns = [St+'_VL', St+'_BF']
        data_grf_columns = [St+'_GRF_X']
        data_torques_columns = [St+'_Knee_X']

        # Select the columns to use
        data_angles = data_angles[data_angles_columns]
        data_emg_envelope = data_emg_envelope[data_emg_columns]
        data_emg_filtered = data_emg_filtered[data_emg_columns]
        data_grf = data_grf[data_grf_columns]
        data_torques = data_torques[data_torques_columns]
        data_torques_norm = data_torques_norm[data_torques_columns]

        # Prepare the input and target to train the models
        curr_X_train = pd.concat([data_emg_envelope], axis=1)
        curr_y_train = data_angles
        
        # Concatenate the data of the patients
        if patient == train_patients[0]: # First patient
            prev_X_train = curr_X_train
            prev_y_train = curr_y_train
        else:
            prev_X_train = pd.concat([prev_X_train, curr_X_train], axis=0)
            prev_y_train = pd.concat([prev_y_train, curr_y_train], axis=0)


    # Load the validation data
    for patient in val_patients:
        # Get all test files of the patient
        data_path = f'data/{patient}'
        print(data_path)
        data_path_patient = f'{data_path}/V1/R/Angles'
        data_files = [f for f in os.listdir(data_path_patient) if f.endswith('.txt')]
        print(data_files)

        # Load the data
        metadata, data_angles, data_emg_envelope, data_emg_filtered, data_grf, data_torques, data_torques_norm = load_data(data_path, data_files)

        St = 'St1'
        data_angles_columns = [St+'_Knee_X']
        data_emg_columns = [St+'_VL', St+'_BF']
        data_grf_columns = [St+'_GRF_X']
        data_torques_columns = [St+'_Knee_X']

        # Select the columns to use
        data_angles = data_angles[data_angles_columns]
        data_emg_envelope = data_emg_envelope[data_emg_columns]
        data_emg_filtered = data_emg_filtered[data_emg_columns]
        data_grf = data_grf[data_grf_columns]
        data_torques = data_torques[data_torques_columns]
        data_torques_norm = data_torques_norm[data_torques_columns]

        # Prepare the input and target to train the models
        curr_X_val = pd.concat([data_emg_envelope], axis=1)
        curr_y_val = data_angles
        
        # Concatenate the data of the patients
        if patient == val_patients[0]: # First patient
            prev_X_val = curr_X_val
            prev_y_val = curr_y_val
        else:
            prev_X_val = pd.concat([prev_X_val, curr_X_val], axis=0)
            prev_y_val = pd.concat([prev_y_val, curr_y_val], axis=0)


    # Defining the models
    models = {
        'KNN': KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None),

        'Random Forest': RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None),
        
        'Gradient Boosting': GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0),

        'AdaBoost': AdaBoostRegressor(estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None),
        
        'Extra Trees Regressor' : ExtraTreesRegressor(n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None),

        'MLP Regressor' : MLPRegressor(hidden_layer_sizes=(50,50,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000),

        'XGBRegressor' : XGBRegressor()
    }

    # Create the voting regressor based on some of the models
    voting_reg = VotingRegressor(estimators=[ ('Random Forest', models['Random Forest']), ('MLP Regressor', models['MLP Regressor']), ('XGBRegressor', models['XGBRegressor'])])

    # Add the voting regressor to the models
    models['Voting Regressor'] = voting_reg

    # Dictionary to store the predictions and metrics to plot after
    predictions = {}
    metrics = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}")

        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])

        pipeline.fit(prev_X_train, prev_y_train)
        y_pred = pipeline.predict(prev_X_val)

        # applyng smoth filter
        # y_pred = moving_average(y_pred, window_size=75)
        # y_pred = smooth_spline(y_pred)
        y_pred = loess_smoothing(y_pred, frac=0.09)
        # y_pred = kalman_filter(y_pred)

        predictions[model_name] = y_pred

        # Calculate assessment metrics
        mae = mean_absolute_error(prev_y_val, y_pred)
        mse = mean_squared_error(prev_y_val, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(prev_y_val, y_pred)
        r2 = r2_score(prev_y_val, y_pred)
        metrics[model_name] = r2

        # To calculate the media after
        r2_vec[model_name] += r2 
        mae_vec[model_name] += mae

        # Print the metrics
        print(f"{model_name} - Mean Absolute Error: {mae:.2f}")
        print(f"{model_name} - Mean Squared Error: {mse:.2f}")
        print(f"{model_name} - Root Mean Squared Error: {rmse:.2f}")
        print(f"{model_name} - Mean Absolute Percentage Error: {mape:.2%}")
        print(f"{model_name} - R^2 Score: {r2:.2f}")
        print() # Blank line

    

if __name__ == '__main__':
    # Define the data path and patients
    data_path = 'data'
    data_patients = ['P1', 'P2', 'P3', 'P4', 'P5', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16']

    n = 1 # Number of patients used for validation

    # Train the model lefting n patients for validation
    # Train the model lefting n files for validation
    list_comb = combinations(data_patients, n)
    num_comb = len(list(list_comb))
    comb = combinations(data_patients, n)
    for val_patients in comb:
        train_patients = [f for f in data_patients if f not in val_patients]
        print(f'\nTraining with {train_patients}, validating with {val_patients}')
        train_and_validate_patients(data_path, train_patients, val_patients)

    for model in r2_vec:
        r2_vec[model] = r2_vec[model]/num_comb
        mae_vec[model] = mae_vec[model]/num_comb
        print(f'{model} - Mean R^2 Score: {r2_vec[model]:.2f}')
        print(f'{model} - Mean Mean Absolute Error: {mae_vec[model]:.2f}')