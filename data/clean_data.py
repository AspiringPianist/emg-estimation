import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Design a Butterworth bandpass filter."""
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass(cutoff, fs, order=1):
    """Design a Butterworth low-pass filter."""
    nyq = fs / 2.0
    cutoff = cutoff / nyq
    b, a = butter(order, cutoff, btype='low')
    return b, a

def preprocess_eeg(eeg_signals, fs, lowcut=0.1, highcut=45.0):
    """Preprocess EEG signals: bandpass filter and ICA for artifact removal."""
    # Apply 0.1-45 Hz bandpass filter
    b, a = butter_bandpass(lowcut, highcut, fs)
    filtered_eeg = np.array([filtfilt(b, a, eeg_signals[:, i]) for i in range(eeg_signals.shape[1])]).T
    
    # Apply ICA for artifact removal
    n_components = min(8, eeg_signals.shape[1])  # Limit to 8 components or available channels
    ica = FastICA(n_components=n_components, random_state=42)
    ica_components = ica.fit_transform(filtered_eeg)
    
    # Select components (simplified: assume first 4 are motion-related, as per paper)
    selected_components = ica_components[:, :4]
    # Reconstruct EEG signals from selected components
    reconstructed_eeg = np.dot(selected_components, ica.mixing_[:, :4].T)
    
    return reconstructed_eeg, ica_components, ica.mixing_

def preprocess_emg(emg_signal, fs, lowcut=0.7, recovery_factor=2):
    """Preprocess EMG signal: full-wave rectification, moving average, and low-pass filter."""
    # Full-wave rectification
    rectified_emg = np.abs(emg_signal)
    
    # 20-point moving average
    window_size = 20
    moving_avg = np.convolve(rectified_emg, np.ones(window_size)/window_size, mode='valid')
    # Pad to match original length
    pad_length = len(emg_signal) - len(moving_avg)
    moving_avg = np.pad(moving_avg, (pad_length//2, pad_length - pad_length//2), mode='edge')
    
    # Apply 0.7 Hz low-pass filter
    b, a = butter_lowpass(lowcut, fs)
    smoothed_emg = filtfilt(b, a, moving_avg)
    
    # Apply recovery factor to restore amplitude
    smoothed_emg *= recovery_factor
    
    return smoothed_emg

def clean_and_save_data(eeg_data, emg_data, eeg_output_path, emg_output_path, ica_output_path=None):
    """Clean EEG and EMG data, ensuring 10-second duration, and save to CSV."""
    # Parse timestamps
    eeg_timestamps = pd.to_datetime(eeg_data['timestamp'], format='%H:%M:%S.%f')
    emg_timestamps = pd.to_datetime(emg_data['timestamp'], format='%H:%M:%S.%f')
    
    # Sampling rates and expected duration
    fs_eeg = 250  # EEG at 250 Hz
    fs_emg = 100  # EMG at 100 Hz
    duration = 10.0  # Expected duration in seconds
    expected_eeg_samples = int(fs_eeg * duration)  # 2500 samples
    expected_emg_samples = int(fs_emg * duration)  # 1000 samples
    
    # Calculate relative time from timestamps
    eeg_rel_time = (eeg_timestamps - eeg_timestamps.iloc[0]).dt.total_seconds().values
    emg_rel_time = (emg_timestamps - emg_timestamps.iloc[0]).dt.total_seconds().values
    
    # Trim or select data to exactly 10 seconds
    eeg_mask = (eeg_rel_time >= 0) & (eeg_rel_time <= duration)
    emg_mask = (emg_rel_time >= 0) & (emg_rel_time <= duration)
    
    eeg_data = eeg_data[eeg_mask].reset_index(drop=True)
    emg_data = emg_data[emg_mask].reset_index(drop=True)
    eeg_rel_time = eeg_rel_time[eeg_mask]
    emg_rel_time = emg_rel_time[emg_mask]
    
    # Generate synthetic time vectors to ensure correct sampling
    if len(eeg_data) != expected_eeg_samples or not np.allclose(eeg_rel_time[1:] - eeg_rel_time[:-1], 1/fs_eeg, rtol=0.1):
        print("EEG timestamps uneven or incorrect. Generating synthetic timestamps.")
        eeg_time = np.linspace(0, duration, expected_eeg_samples, endpoint=False)
        eeg_data = eeg_data.iloc[:expected_eeg_samples]  # Trim or pad if needed
        if len(eeg_data) < expected_eeg_samples:
            eeg_data = eeg_data.reindex(range(expected_eeg_samples)).fillna(method='ffill')
    else:
        eeg_time = eeg_rel_time[:expected_eeg_samples]
    
    if len(emg_data) != expected_emg_samples or not np.allclose(emg_rel_time[1:] - emg_rel_time[:-1], 1/fs_emg, rtol=0.1):
        print("EMG timestamps uneven or incorrect. Generating synthetic timestamps.")
        emg_time = np.linspace(0, duration, expected_emg_samples, endpoint=False)
        emg_data = emg_data.iloc[:expected_emg_samples]  # Trim or pad if needed
        if len(emg_data) < expected_emg_samples:
            emg_data = emg_data.reindex(range(expected_emg_samples)).fillna(method='ffill')
    else:
        emg_time = emg_rel_time[:expected_emg_samples]
    
    # Extract signal data
    eeg_signals = eeg_data[[col for col in eeg_data.columns if col != 'timestamp']].values
    emg_signals = emg_data[[col for col in emg_data.columns if col != 'timestamp']].values
    
    # Preprocess EEG
    eeg_processed, ica_components, mixing_matrix = preprocess_eeg(eeg_signals, fs_eeg)
    
    # Preprocess EMG (no interpolation, process at native 100 Hz)
    emg_processed = np.array([preprocess_emg(emg_signals[:, i], fs_emg) 
                             for i in range(emg_signals.shape[1])]).T
    
    # Save cleaned data
    cleaned_eeg = pd.DataFrame(eeg_processed, columns=[f'eeg{i+1}' for i in range(eeg_processed.shape[1])])
    cleaned_eeg['time'] = eeg_time
    cleaned_emg = pd.DataFrame(emg_processed, columns=[f'emg{i+1}' for i in range(emg_processed.shape[1])])
    cleaned_emg['time'] = emg_time
    
    cleaned_eeg.to_csv(eeg_output_path, index=False)
    cleaned_emg.to_csv(emg_output_path, index=False)
    
    # Save ICA components and mixing matrix if path provided
    if ica_output_path:
        ica_df = pd.DataFrame(ica_components, columns=[f'ica{i+1}' for i in range(ica_components.shape[1])])
        ica_df['time'] = eeg_time
        ica_df.to_csv(ica_output_path, index=False)
        
        mixing_df = pd.DataFrame(mixing_matrix, 
                                columns=[f'ica{i+1}' for i in range(mixing_matrix.shape[1])],
                                index=[f'eeg{i+1}' for i in range(mixing_matrix.shape[0])])
        mixing_df.to_csv(ica_output_path.replace('.csv', '_mixing.csv'))
    
    print(f"Data cleaned and aligned to 10 seconds. Saved as '{eeg_output_path}' and '{emg_output_path}'.")
    
    return ica_components, mixing_matrix if ica_output_path else None

if __name__ == "__main__":
    # Load data
    eeg_data = pd.read_csv('eeg_emg_trial/Trial_02/eeg_data.csv')
    emg_data = pd.read_csv('eeg_emg_trial/Trial_02/emg_data.csv')
    # Clean and save
    clean_and_save_data(eeg_data, emg_data, 'cleaned_eeg_data.csv', 'cleaned_emg_data.csv', 'ica_components.csv')