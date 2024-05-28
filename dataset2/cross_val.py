import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from utils import load_data
    

def train_and_validate(data_path, train_files, val_file):
    '''
    Function to train a model and validate it with a validation file

    INPUT:
        data_path (str): Path to the data folder
        train_files (list): List of files to use for training
        val_file (str): File to use for validation

    OUTPUT:
        model (XGBRegressor): Trained model
    '''

    # Load the data
    data_angles, data_emg_envelope, data_emg_filtered, data_grf, data_torques, data_torques_norm = load_data(data_path, train_files)

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

    # Train the model
    model = XGBRegressor()
    model.fit(X_train, y_train)

    # Load the validation data
    data_angles_val, data_emg_envelope_val, data_emg_filtered_val, data_grf_val, data_torques_val, data_torques_norm_val = load_data(data_path, val_file)

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

    # Validate the model
    y_val_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)
    print(f'Validation r2 for {val_file}: {r2}')
    
    return model


if __name__ == '__main__':
    # Define the data path and files
    data_path = 'data/P5'
    data_files = ['T1.txt', 'T2.txt', 'T3.txt', 'T4.txt', 'T5.txt', 'T6.txt', 'T7.txt', 'T8.txt', 'T9.txt', 'T10.txt']

    # Validate by leaving a file out
    for i in range(len(data_files)):
        train_files = [f for j, f in enumerate(data_files) if j != i]
        val_file = data_files[i]
        print(f'\nTraining with {train_files}, validating with {val_file}')
        model = train_and_validate(data_path, train_files, val_file)