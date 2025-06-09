import pandas as pd
import numpy as np
from scipy.signal import iirnotch, filtfilt
import mne

def preprocess_eeg(eeg_signals, fs):
    """Remove only 50Hz powerline noise from EEG signals"""
    notch_freq = 50
    Q = 30  # Quality factor
    b_notch, a_notch = iirnotch(notch_freq, Q=Q, fs=fs)
    notch_filtered = np.array([filtfilt(b_notch, a_notch, eeg_signals[:, i]) 
                              for i in range(eeg_signals.shape[1])]).T
    return notch_filtered

def analyze_time_sync(eeg_times, emg_times):
    # Compute start and end time differences
    start_diff = eeg_times[0] - emg_times[0]
    end_diff = eeg_times[-1] - emg_times[-1]
    print(f"EEG starts {start_diff*1000:.2f} ms {'after' if start_diff>0 else 'before'} EMG.")
    print(f"EEG ends {end_diff*1000:.2f} ms {'after' if end_diff>0 else 'before'} EMG.")

    # Jitter calculation
    min_len = min(len(eeg_times), len(emg_times))
    time_jitter = eeg_times[:min_len] - emg_times[:min_len]
    print(f"Mean jitter: {np.mean(np.abs(time_jitter))*1000:.2f} ms")
    print(f"Max jitter: {np.max(np.abs(time_jitter))*1000:.2f} ms")
    print(f"Std of jitter: {np.std(time_jitter)*1000:.2f} ms")

def synchronize_emg_to_eeg(eeg_times, emg_times, emg_signals):
    """
    Interpolates EMG signals to EEG time base.
    eeg_times: np.array of EEG times (seconds)
    emg_times: np.array of EMG times (seconds)
    emg_signals: np.array, shape (n_samples, n_channels)
    Returns interpolated EMG signals at EEG time points.
    """
    emg_interp = np.zeros((len(eeg_times), emg_signals.shape[1]))
    for ch in range(emg_signals.shape[1]):
        emg_interp[:, ch] = np.interp(eeg_times, emg_times, emg_signals[:, ch])
    return emg_interp

def clean_and_save_data(eeg_data, emg_data, eeg_output_path, emg_output_path):
    # Parse timestamps
    eeg_timestamps = pd.to_datetime(eeg_data['timestamp'], format='%H:%M:%S.%f')
    emg_timestamps = pd.to_datetime(emg_data['timestamp'], format='%H:%M:%S.%f')
    eeg_rel_time = (eeg_timestamps - eeg_timestamps.iloc[0]).dt.total_seconds().values
    emg_rel_time = (emg_timestamps - emg_timestamps.iloc[0]).dt.total_seconds().values

    # Analyze original synchronization
    print("Before synchronization:")
    analyze_time_sync(eeg_rel_time, emg_rel_time)

    # Extract signals
    eeg_signals = eeg_data[[col for col in eeg_data.columns if col != 'timestamp']].values
    emg_signals = emg_data[[col for col in emg_data.columns if col != 'timestamp']].values

    # Preprocess EEG
    fs_eeg = 250
    eeg_processed = preprocess_eeg(eeg_signals, fs_eeg)

    # Synchronize EMG to EEG time base
    emg_synced = synchronize_emg_to_eeg(eeg_rel_time, emg_rel_time, emg_signals)

    # Analyze synchronization after interpolation
    print("After synchronization:")
    analyze_time_sync(eeg_rel_time, eeg_rel_time)  # Now both are on the EEG time base

    # Save cleaned and time-synced data
    cleaned_eeg = pd.DataFrame(eeg_processed, columns=[f'eeg{i+1}' for i in range(eeg_processed.shape[1])])
    cleaned_eeg['time'] = eeg_rel_time
    cleaned_emg = pd.DataFrame(emg_synced, columns=[f'emg{i+1}' for i in range(emg_synced.shape[1])])
    cleaned_emg['time'] = eeg_rel_time  # Now both are on the same time base

    cleaned_eeg.to_csv(eeg_output_path, index=False)
    cleaned_emg.to_csv(emg_output_path, index=False)

    print(f"Data processed and synchronized. Saved as '{eeg_output_path}' and '{emg_output_path}'.")
    print(f"EEG: Powerline noise removed, {len(cleaned_eeg)} samples at {fs_eeg}Hz")
    print(f"EMG: Interpolated to EEG time base, {len(cleaned_emg)} samples")

if __name__ == "__main__":
    # Load data
    eeg_data = pd.read_csv('eeg_emg_trial/Trial_02/eeg_data.csv')
    emg_data = pd.read_csv('eeg_emg_trial/Trial_02/emg_data.csv')
    # Clean and save
    clean_and_save_data(eeg_data, emg_data, 'cleaned_eeg_data.csv', 'cleaned_emg_data.csv')
