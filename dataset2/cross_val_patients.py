from itertools import combinations
import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from utils import load_data
from lazypredict.Supervised import LazyRegressor
    

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


    # Use lazy predict to get a holistc view about the result of a lot os models
    lazy_model = LazyRegressor()
    models_lazy, predictions_lazy = lazy_model.fit(prev_X_train, prev_X_val, prev_y_train, prev_y_val)
    print(models_lazy)

    # Find the best model
    best_model = models_lazy['R2'].idxmax()
    return best_model

    

if __name__ == '__main__':
    # Define the data path and patients
    data_path = 'data'
    data_patients = ['P1', 'P2', 'P3', 'P4', 'P5', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16']

    n = 1 # Number of patients used for validation

    # Train the model lefting n patients for validation
    comb = combinations(data_patients, n)
    for val_patients in comb:
        train_patients = [f for f in data_patients if f not in val_patients]
        print(f'\nTraining with {train_patients}, validating with {val_patients}')
        model = train_and_validate_patients(data_path, train_patients, val_patients)