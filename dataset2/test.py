import os
import pandas as pd

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
        for dir in dirs:
            print(dir)
            if dir == 'Angles':
                # Read the files inside this folder
                for file in os.listdir(os.path.join(root, dir)):
                    print(file)
                    if count_angles == 0:
                        data_angles = pd.read_csv(os.path.join(root, dir, file), delimiter='\t')
                        count_angles += 1
                    else:
                        data_angles = pd.concat([data_angles, pd.read_csv(os.path.join(root, dir, file), delimiter='\t')], axis=0)

            elif dir == 'EMG filtered':
                # Read the files inside this folder
                for file in os.listdir(os.path.join(root, dir)):
                    print(file)
                    if count_emg == 0:
                        data_emg = pd.read_csv(os.path.join(root, dir, file), delimiter='\t')
                        count_emg += 1
                    else:
                        data_emg = pd.concat([data_emg, pd.read_csv(os.path.join(root, dir, file), delimiter='\t')], axis=0)

            elif dir == 'Torques':
                # Read the files inside this folder
                for file in os.listdir(os.path.join(root, dir)):
                    print(file)
                    if count_torques == 0:
                        data_torques = pd.read_csv(os.path.join(root, dir, file), delimiter='\t')
                        count_torques += 1
                    else:
                        data_torques = pd.concat([data_torques, pd.read_csv(os.path.join(root, dir, file), delimiter='\t')], axis=0)

            elif dir == 'GRF':
                # Read the files inside this folder
                for file in os.listdir(os.path.join(root, dir)):
                    print(file)
                    if count_grf == 0:
                        data_grf = pd.read_csv(os.path.join(root, dir, file), delimiter='\t')
                        count_grf += 1
                    else:
                        data_grf = pd.concat([data_grf, pd.read_csv(os.path.join(root, dir, file), delimiter='\t')], axis=0)
            
    return data_emg, data_torques, data_grf, data_angles


files_path = 'data/test'
data_emg, data_torques, data_grf, data_angles = load_data(files_path)
print(data_angles.head())
print(data_angles.shape)
print()
print(data_emg.head())
print(data_emg.shape)
print()
print(data_torques.head())
print(data_torques.shape)
print()
print(data_grf.head())
print(data_grf.shape)
print()