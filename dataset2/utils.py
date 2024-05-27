import os
import pandas as pd


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
                    if file in files_to_load: # Check if the file is in the list of files to load
                        if count_angles == 0: # If the data was not loaded yet
                            data_angles = pd.read_csv(os.path.join(root, dir, file), delimiter='\t')
                            count_angles += 1
                        else:
                            data_angles = pd.concat([data_angles, pd.read_csv(os.path.join(root, dir, file), delimiter='\t')], axis=0)

            elif dir == 'EMG envelope':
                for file in os.listdir(os.path.join(root, dir)):
                    if file in files_to_load:
                        if count_emg_envelope == 0:
                            data_emg_envelope = pd.read_csv(os.path.join(root, dir, file), delimiter='\t')
                            count_emg_envelope += 1
                        else:
                            data_emg_envelope = pd.concat([data_emg_envelope, pd.read_csv(os.path.join(root, dir, file), delimiter='\t')], axis=0)

            elif dir == 'EMG filtered':
                for file in os.listdir(os.path.join(root, dir)):
                    if file in files_to_load:
                        if count_emg_filtered == 0:
                            data_emg_filtered = pd.read_csv(os.path.join(root, dir, file), delimiter='\t')
                            count_emg_filtered += 1
                        else:
                            data_emg_filtered = pd.concat([data_emg_filtered, pd.read_csv(os.path.join(root, dir, file), delimiter='\t')], axis=0)

            elif dir == 'GRF':
                for file in os.listdir(os.path.join(root, dir)):
                    if file in files_to_load:
                        if count_grf == 0:
                            data_grf = pd.read_csv(os.path.join(root, dir, file), delimiter='\t')
                            count_grf += 1
                        else:
                            data_grf = pd.concat([data_grf, pd.read_csv(os.path.join(root, dir, file), delimiter='\t')], axis=0)

            elif dir == 'Torques':
                for file in os.listdir(os.path.join(root, dir)):
                    if file in files_to_load:
                        if count_torques == 0:
                            data_torques = pd.read_csv(os.path.join(root, dir, file), delimiter='\t')
                            count_torques += 1
                        else:
                            data_torques = pd.concat([data_torques, pd.read_csv(os.path.join(root, dir, file), delimiter='\t')], axis=0)

            elif dir == 'Torques_Norm':
                for file in os.listdir(os.path.join(root, dir)):
                    if file in files_to_load:
                        if count_torques_norm == 0:
                            data_torques_norm = pd.read_csv(os.path.join(root, dir, file), delimiter='\t')
                            count_torques_norm += 1
                        else:
                            data_torques_norm = pd.concat([data_torques_norm, pd.read_csv(os.path.join(root, dir, file), delimiter='\t')], axis=0)
            
    return data_angles, data_emg_envelope, data_emg_filtered, data_grf, data_torques, data_torques_norm