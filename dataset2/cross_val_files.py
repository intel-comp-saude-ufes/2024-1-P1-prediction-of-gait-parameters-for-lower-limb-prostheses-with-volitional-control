from itertools import combinations
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from utils import load_data
from lazypredict.Supervised import LazyRegressor
    

def train_and_validate_files(data_path, train_files, val_file):
    '''
    Function to train a model and validate it with validation files

    INPUT:
        data_path (str): Path to the data folder
        train_files (list): Files to use for training
        val_file (str): Files to use for validation

    OUTPUT:
        best_model (Model): Best model to use
    '''

    # Load the data
    metadata, data_angles, data_emg_envelope, data_emg_filtered, data_grf, data_torques, data_torques_norm = load_data(data_path, train_files)

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
    X_train = pd.concat([data_emg_envelope], axis=1) # Concatenate the input model data
    y_train = data_angles # Get the target data

    # Load the validation data
    metadata_val, data_angles_val, data_emg_envelope_val, data_emg_filtered_val, data_grf_val, data_torques_val, data_torques_norm_val = load_data(data_path, val_file)

    # Select the columns to use
    data_angles_val = data_angles_val[data_angles_columns]
    data_emg_envelope_val = data_emg_envelope_val[data_emg_columns]
    data_emg_filtered_val = data_emg_filtered_val[data_emg_columns]
    data_grf_val = data_grf_val[data_grf_columns]
    data_torques_val = data_torques_val[data_torques_columns]
    data_torques_norm_val = data_torques_norm_val[data_torques_columns]

    # Prepare the input and target to validate the models
    X_val = pd.concat([data_emg_envelope_val], axis=1)
    y_val = data_angles_val

    # Use lazy predict to get a holistc view about the result of a lot os models
    lazy_model = LazyRegressor()
    models_lazy, predictions_lazy = lazy_model.fit(X_train, X_val, y_train, y_val)
    print(models_lazy)

    # Find the best model
    best_model = models_lazy['R2'].idxmax()
    return best_model
    

if __name__ == '__main__':
    # Define the data path and files
    data_path = 'data/P5'
    data_files = ['T1.txt', 'T2.txt', 'T3.txt', 'T4.txt', 'T5.txt', 'T6.txt', 'T7.txt', 'T8.txt', 'T9.txt', 'T10.txt']

    n = 1 # Number of files used for validation

    # Train the model lefting n files for validation
    comb = combinations(data_files, n)
    for val_files in comb:
        train_files = [f for f in data_files if f not in val_files]
        print(f'\nTraining with {train_files}, validating with {val_files}')
        model = train_and_validate_files(data_path, train_files, val_files)