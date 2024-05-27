import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Função para carregar os dados dos arquivos
def load_data(folder_path, files_to_load):
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
    

# Função para treinar e validar o modelo
def train_and_validate(data_path, train_files, val_file):
    data_angles, data_emg_envelope, data_emg_filtered, data_grf, data_torques, data_torques_norm = load_data(data_path, train_files)

    St = 'St1'
    data_emg_columns = [St+'_VL', St+'_BF']
    data_torques_columns = [St+'_Knee_X']
    data_grf_columns = [St+'_GRF_X']
    data_angles_columns = [St+'_Knee_X']

    data_emg_envelope = data_emg_envelope[data_emg_columns]
    data_emg_filtered = data_emg_filtered[data_emg_columns]
    data_grf = data_grf[data_grf_columns]
    data_angles = data_angles[data_angles_columns]
    data_torques = data_torques[data_torques_columns]
    data_torques_norm = data_torques_norm[data_torques_columns]

    X_train = pd.concat([data_emg_envelope], axis=1) # Concatenate the input model data
    y_train = data_angles # Get the target data

    # Treinar o modelo
    model = XGBRegressor()
    model.fit(X_train, y_train)


    data_angles_val, data_emg_envelope_val, data_emg_filtered_val, data_grf_val, data_torques_val, data_torques_norm_val = load_data(data_path, val_file)

    data_emg_envelope_val = data_emg_envelope_val[data_emg_columns]
    data_emg_filtered_val = data_emg_filtered_val[data_emg_columns]
    data_grf_val = data_grf_val[data_grf_columns]
    data_angles_val = data_angles_val[data_angles_columns]
    data_torques_val = data_torques_val[data_torques_columns]
    data_torques_norm_val = data_torques_norm_val[data_torques_columns]

    X_val = pd.concat([data_emg_envelope_val], axis=1)
    y_val = data_angles_val


    # Validar o modelo
    y_val_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)
    print(f'Validation r2 for {val_file}: {r2}')
    
    return model



if __name__ == '__main__':
    # Lista dos arquivos de dados
    data_path = 'data/P5'
    data_files = ['T1.txt', 'T2.txt', 'T3.txt', 'T4.txt', 'T5.txt', 'T6.txt', 'T7.txt', 'T8.txt', 'T9.txt', 'T10.txt']

    # Validar deixando um arquivo de fora
    for i in range(len(data_files)):
        train_files = [f for j, f in enumerate(data_files) if j != i]
        val_file = data_files[i]
        print(f'\nTraining with {train_files}, validating with {val_file}')
        model = train_and_validate(data_path, train_files, val_file)