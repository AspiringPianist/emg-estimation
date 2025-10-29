# data_preparation.py

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

FS_EMG = 100
SEGMENT_DURATION = 2
WINDOW_SAMPLES = int(FS_EMG * SEGMENT_DURATION)
DATA_DIR = 'C:/Users/rajes/Desktop/IIITB/emg-emg/emg_data/'

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    return butter(order, [low, high], btype='band')

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, cutoff / nyq, btype='low')

def apply_filter(data, b, a):
    return filtfilt(b, a, data, axis=0)

def preprocess_emg(data):
    b_band, a_band = butter_bandpass(20, 45, FS_EMG, order=4)
    filtered_data = apply_filter(data, b_band, a_band)
    rectified_data = np.abs(filtered_data)
    b_low, a_low = butter_lowpass(6, FS_EMG, order=4)
    return apply_filter(rectified_data, b_low, a_low)

def zscore_normalize(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    std[std == 0] = 1
    return (data - mean) / std, mean, std

def create_sequences(input_data, target_data, sequence_len):
    X, Y = [], []
    num_windows = len(input_data)
    for i in range(num_windows - sequence_len):
        X.append(input_data[i:i + sequence_len])
        Y.append(target_data[i:i + sequence_len])
    return np.array(X), np.array(Y)

def prepare_all_emg(sequence_len, input_channels, target_channel, data_dir=DATA_DIR):
    all_sequences_X, all_sequences_Y = [], []
    num_input_channels = len(input_channels)
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for filename in file_list:
        try:
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath)
            
            input_df, target_df = df[input_channels], df[[target_channel]]
            
            processed_input = preprocess_emg(input_df.values)
            processed_target = preprocess_emg(target_df.values)
            
            norm_input, _, _ = zscore_normalize(processed_input)
            norm_target, _, _ = zscore_normalize(processed_target)
            
            num_samples = len(norm_input)
            num_windows = num_samples // WINDOW_SAMPLES
            
            if num_windows < sequence_len:
                # ADDED: More informative warning
                # print(f"  [INFO] Skipping file {filename}: needs at least {sequence_len * SEGMENT_DURATION} seconds of data, but file only has ~{num_samples / FS_EMG:.2f} seconds.")
                continue

            windowed_input = norm_input[:num_windows * WINDOW_SAMPLES].reshape(num_windows, WINDOW_SAMPLES, num_input_channels)
            windowed_target = norm_target[:num_windows * WINDOW_SAMPLES].reshape(num_windows, WINDOW_SAMPLES, 1)
            
            X, Y = create_sequences(windowed_input, windowed_target, sequence_len)
            
            if X.shape[0] > 0:
                all_sequences_X.append(X)
                all_sequences_Y.append(Y)
        except Exception:
            continue
    
    if not all_sequences_X:
        return np.array([]), np.array([])

    final_X = np.concatenate(all_sequences_X, axis=0)
    final_Y = np.concatenate(all_sequences_Y, axis=0).reshape(len(final_X), -1)
    
    return final_X, final_Y