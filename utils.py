import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert


def load_data(folder_path, files_to_load):
    '''
    Function to load train/test data from a folder

    INPUT:
        folder_path (str): Path to the folder containing the train/test data

    OUTPUT:
        metadata (pd.DataFrame): Pacient info data
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
        metadata = pd.read_csv(f'{folder_path}/Metadata.txt', delimiter='\t') 
        
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
            
    return metadata, data_angles, data_emg_envelope, data_emg_filtered, data_grf, data_torques, data_torques_norm


def process_emg_signal(signal, lowcut=20.0, highcut=450.0, fs=1000.0, order=4):
    '''
    Function to process an EMG signal

    INPUT:
        signal (np.array): EMG signal
        lowcut (float): Lowcut frequency
        highcut (float): Highcut frequency
        fs (float): Sampling frequency
        order (int): Filter order

    OUTPUT:
        envelope (np.array): EMG envelope
    '''

    # Bandpass filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    
    # Rectification
    rectified_signal = np.abs(filtered_signal)
    
    # Envelope using Hilbert transform
    envelope = np.abs(hilbert(rectified_signal))
    
    return envelope