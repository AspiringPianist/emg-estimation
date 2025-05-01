import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, stft
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import FastICA
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D

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

def preprocess_emg(emg_signal, fs, lowcut=0.7, recovery_factor=2):
    """Preprocess EMG signal: low-pass filter and apply recovery factor."""
    # Apply 0.7 Hz low-pass filter
    b, a = butter_lowpass(lowcut, fs)
    smoothed_emg = filtfilt(b, a, emg_signal)
    
    # Apply recovery factor to restore amplitude
    smoothed_emg *= recovery_factor
    
    return smoothed_emg

def preprocess_eeg(eeg_signal, fs, lowcut=1.0, highcut=45.0):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, eeg_signal)

def compute_stft(signal, fs, nperseg=512, noverlap=0.99, n_samples=None):
    """Compute Short-Time Fourier Transform with specified number of samples."""
    nperseg = min(nperseg, len(signal))
    noverlap = int(nperseg * noverlap)
    if n_samples is not None and len(signal) > n_samples:
        signal = signal[:n_samples]
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
    return f, t, np.abs(Zxx)

def plot_eeg_emg_multi_view(eeg_signals, emg_raw, eeg_channels, eeg_time, emg_time, fs_eeg, fs_emg, save_path='eeg_emg_multi_view_analysis.png'):
    """
    Create multi-view plots for each EEG channel with EMG signals at bottom:
    1. 3D surface plot of EEG time-frequency-amplitude
    2. 2D heatmap of EEG time-frequency
    3. EMG time series (raw rectified in green, processed in red)
    """
    # Determine duration from data
    duration = min(eeg_time[-1], emg_time[-1]) if len(eeg_time) > 0 and len(emg_time) > 0 else 14.0
    
    # Calculate number of rows and columns for subplots
    n_channels = min(len(eeg_channels), 100)
    n_cols = n_channels
    
    # Create figure
    fig = plt.figure(figsize=(15, 15))
    
    # Create unified time grid based on actual data length
    unified_time_eeg = np.linspace(0, duration, len(eeg_time))
    unified_time_emg = np.linspace(0, duration, len(emg_time))
    
    # Preprocess EMG
    rectified_emg = np.abs(emg_raw)
    processed_emg = preprocess_emg(rectified_emg, fs_emg)
    
    #Preprocess EEG
    eeg_signals_filtered = np.array([preprocess_eeg(signal, fs_eeg) for signal in eeg_signals.T]).T
    
    # Resample EMG signals
    # if len(emg_time) != len(unified_time_emg):
    if False:
        rectified_emg_resampled = interp1d(emg_time.flatten(), rectified_emg.flatten(), 
                                          bounds_error=False, fill_value="extrapolate")(unified_time_emg)
        processed_emg_resampled = interp1d(emg_time.flatten(), processed_emg.flatten(), 
                                          bounds_error=False, fill_value="extrapolate")(unified_time_emg)
    else:
        rectified_emg_resampled = rectified_emg
        processed_emg_resampled = processed_emg
    
    # Create colormap
    cmap = cm.jet
    
    for col_idx, ch_idx in enumerate(range(n_channels)):
        # 1. 3D Surface plot
        ax_3d = fig.add_subplot(3, n_cols, col_idx + 1, projection='3d')
        f, t_stft, Sxx = compute_stft(eeg_signals_filtered[:, ch_idx], fs_eeg, n_samples=len(eeg_time))
        freq_mask = (f >= 0.7) & (f <= 12)
        T, F = np.meshgrid(t_stft, f[freq_mask])
        surf = ax_3d.plot_surface(T, F, Sxx[freq_mask], cmap=cmap, antialiased=True, linewidth=0)
        ax_3d.set_xlabel('Time')
        ax_3d.set_ylabel('Frequency')
        ax_3d.set_zlabel('Amplitude')
        ax_3d.set_title(eeg_channels[ch_idx])
        cbar = fig.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=5)
        ax_3d.view_init(elev=30, azim=-60)
        
        # 2. 2D Heatmap
        ax_2d = fig.add_subplot(3, n_cols, n_cols + col_idx + 1)
        im = ax_2d.imshow(Sxx[freq_mask], aspect='auto', 
                        extent=[0, duration, f[freq_mask][0], f[freq_mask][-1]],
                        origin='lower', cmap=cmap)
        ax_2d.set_xlabel('Time')
        ax_2d.set_ylabel('Frequency')
        cbar = fig.colorbar(im, ax=ax_2d)
        
        # 3. EMG Time Series
        ax_emg = fig.add_subplot(3, n_cols, 2*n_cols + col_idx + 1)
        if np.any(np.isnan(rectified_emg_resampled)) or np.any(np.isnan(processed_emg_resampled)):
            print("Warning: NaN values detected in EMG data")
        ax_emg.plot(unified_time_emg, rectified_emg_resampled, 'g-', alpha=0.7, linewidth=1, label='Rectified EMG')
        ax_emg.plot(unified_time_emg, processed_emg_resampled, 'r-', linewidth=2, label='Processed EMG')
        ax_emg.set_ylim([0, max(np.max(rectified_emg_resampled), np.max(processed_emg_resampled)) * 1.1])
        ax_emg.set_xlabel('Time(s)')
        ax_emg.set_ylabel('Amplitude(V)')
        if col_idx == 0:
            ax_emg.legend(loc='upper right')
        ax_emg.set_xlim(0, duration)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    # Load cleaned data
    eeg_data = pd.read_csv('cleaned_eeg_data.csv')
    emg_data = pd.read_csv('cleaned_emg_data.csv')
    
    # Extract signals and time
    fs_eeg = 250  # Adjust if actual sampling rate differs
    fs_emg = 100  # Adjust if actual sampling rate differs
    for j in range(1, 24, 5):
        eeg_channels = [f'eeg{i}'for i in range(j, j+5)]
        emg_channel = 'emg2'
        
        available_eeg_channels = [col for col in eeg_data.columns if col.startswith('eeg')]
        if not all(ch in eeg_data.columns for ch in eeg_channels):
            print(f"Warning: Some requested channels not found. Using available channels instead: {available_eeg_channels[:3]}")
            eeg_channels = available_eeg_channels[:3]
        
        eeg_signals = eeg_data[eeg_channels].values if all(ch in eeg_data.columns for ch in eeg_channels) else eeg_data[available_eeg_channels[:3]].values
        eeg_time = eeg_data['time'].values
        
        if emg_channel in emg_data.columns:
            emg_raw = emg_data[emg_channel].values
        else:
            emg_cols = [col for col in emg_data.columns if col.startswith('emg')]
            if emg_cols:
                emg_raw = emg_data[emg_cols[0]].values
                print(f"Warning: Using {emg_cols[0]} instead of {emg_channel}")
            else:
                raise ValueError("No suitable EMG data column found")
        
        emg_time = np.sort(emg_data['time'].values)
        
        print("EEG signals shape:", eeg_signals.shape)
        print("EEG time shape:", eeg_time.shape)
        print("EMG raw shape:", emg_raw.shape)
        print("EMG time shape:", emg_time.shape)
        print("EEG time sample:", eeg_time[:5])
        print("EMG time sample:", emg_time[:5])
        name=f"eeg_emg_multi_view_{j}_to_{j+5}.png"
        
        plot_eeg_emg_multi_view(eeg_signals, emg_raw, eeg_channels, eeg_time, emg_time, fs_eeg, fs_emg, name)

if __name__ == "__main__":
    main()