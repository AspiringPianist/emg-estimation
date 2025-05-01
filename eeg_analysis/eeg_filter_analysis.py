import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import mne
from mne.filter import filter_data
import tkinter as tk
from tkinter import filedialog

class Config:
    """Configuration class to handle paths and settings"""
    def __init__(self):
        self.data_dir = None
        self.output_dir = None
        self.excluded_channels = ['timestamp', 'datetime', 'time_seconds', 'ch25', 'ch26', 'ch27']
    
    def setup(self):
        """Setup paths through user input"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        print("Please select the directory containing EEG trials data...")
        self.data_dir = filedialog.askdirectory(title="Select Data Directory")
        
        print("Please select where to save the output visualizations...")
        self.output_dir = filedialog.askdirectory(title="Select Output Directory")
        
        if not self.data_dir or not self.output_dir:
            raise ValueError("Both data and output directories must be selected")
        
        return self

def load_eeg_data(file_path):
    """Load EEG data from CSV file with proper timestamp handling"""
    print(f"Reading EEG data from {file_path}...")
    data = pd.read_csv(file_path)
    # Convert timestamp strings to datetime objects
    data['datetime'] = pd.to_datetime(data['timestamp'])
    # Calculate actual sampling intervals
    data['time_seconds'] = (data['datetime'] - data['datetime'].iloc[0]).dt.total_seconds()
    return data

def apply_filters(data, ch_names, timestamps, sfreq=250):
    """Apply MNE filters with non-uniform sampling handling"""
    # Create MNE raw object with proper time points
    actual_sfreq = 1/np.mean(np.diff(timestamps))
    if actual_sfreq != sfreq:
        print(f"Warning: Actual sampling frequency ({actual_sfreq}) differs from expected ({sfreq}).")
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data.T, info)
    
    # Apply notch filter at 50Hz with careful bandwidth
    raw_notch = raw.copy().notch_filter(
        freqs=50,
        verbose=False
    )
    notch_data = raw_notch.get_data().T
    
    # # Calculate transition bandwidths (in Hz)
    # l_trans_bandwidth = 0.5  # wider transition for low frequencies
    # h_trans_bandwidth = 7.5  # 25% of the lowpass frequency
    
    return notch_data, notch_data

def compute_psd(data, fs=250, nperseg=4096):
    """Compute power spectral density"""
    f, pxx = signal.welch(data, fs=fs, nperseg=nperseg)
    return f, pxx

def plot_comparison(raw_signal, filtered_signal, fs=250, channel_name='EEG'):
    """Plot time domain and PSD comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time domain plots
    time = np.arange(len(raw_signal))/fs
    ax1.plot(time, raw_signal, 'b-', label='Raw', alpha=0.7)
    ax1.plot(time, filtered_signal, 'r-', label='Filtered', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (μV)')
    ax1.set_title(f'{channel_name} - Time Domain')
    ax1.legend()
    ax1.grid(True)

    # Zoomed time domain
    zoom_start = int(len(raw_signal)/2)
    zoom_duration = int(fs * 0.1)  # 250ms
    ax2.plot(time[zoom_start:zoom_start+zoom_duration], 
             raw_signal[zoom_start:zoom_start+zoom_duration], 
             'b-', label='Raw', alpha=0.7)
    ax2.plot(time[zoom_start:zoom_start+zoom_duration], 
             filtered_signal[zoom_start:zoom_start+zoom_duration], 
             'r-', label='Filtered', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude (μV)')
    ax2.set_title('Zoomed View (250ms)')
    ax2.legend()
    ax2.grid(True)

    # PSD plots
    f_raw, pxx_raw = compute_psd(raw_signal, fs)
    f_filtered, pxx_filtered = compute_psd(filtered_signal, fs)
    
    # Full spectrum
    ax3.semilogy(f_raw, pxx_raw, 'b-', label='Raw', alpha=0.7)
    ax3.semilogy(f_filtered, pxx_filtered, 'r-', label='Filtered', alpha=0.7)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power/Frequency (μV²/Hz)')
    ax3.set_title('Power Spectral Density')
    ax3.set_xlim(0, 250)
    ax3.legend()
    ax3.grid(True)

    # Zoomed PSD around 50Hz
    ax4.semilogy(f_raw, pxx_raw, 'b-', label='Raw', alpha=0.7)
    ax4.semilogy(f_filtered, pxx_filtered, 'r-', label='Filtered', alpha=0.7)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Power/Frequency (μV²/Hz)')
    ax4.set_title('PSD - Zoomed around 50Hz')
    ax4.set_xlim(45, 55)
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    return fig

def plot_all_eeg_channels(raw_data, filtered_data, channels, time_seconds, duration=10):  # Changed default from 5 to 10 seconds
    """Plot using actual timestamps"""
    n_channels = len(channels)
    n_rows = 6
    n_cols = 4
    
    # Plot raw data
    fig_raw = plt.figure(figsize=(20, 15))
    fig_raw.suptitle('Raw EEG Data - All Channels', fontsize=16)
    
    # Get time window for plotting
    end_time = duration if duration else time_seconds[-1]
    mask = time_seconds <= end_time
    
    for i, channel in enumerate(channels):
        print(channels)
        ax = plt.subplot(n_rows, n_cols, i+1)
        ax.plot(time_seconds[mask], raw_data[mask, i], 'b-', linewidth=0.5)
        ax.set_title(f'Channel {channel}')
        ax.set_xlabel('Time (s)' if i >= (n_channels-n_cols) else '')
        ax.set_ylabel('Amplitude (μV)')
        ax.grid(True)
    
    plt.tight_layout()
    
    # Plot filtered data
    fig_filtered = plt.figure(figsize=(20, 15))
    fig_filtered.suptitle('Filtered EEG Data - All Channels', fontsize=16)
    
    for i, channel in enumerate(channels):
        print(channels)
        ax = plt.subplot(n_rows, n_cols, i+1)
        ax.plot(time_seconds[mask], filtered_data[mask, i], 'r-', linewidth=0.5)
        ax.set_title(f'Channel {channel}')
        ax.set_xlabel('Time (s)' if i >= (n_channels-n_cols) else '')
        ax.set_ylabel('Amplitude (μV)')
        ax.grid(True)
    
    plt.tight_layout()
    return fig_raw, fig_filtered

def process_trial(trial_num, config):
    """Process a single trial"""
    eeg_file_path = os.path.join(config.data_dir, f'Trial_{trial_num:02d}', 'eeg_data.csv')
    data = load_eeg_data(eeg_file_path)
    
    # Get EEG channels
    eeg_channels = [col for col in data.columns if col not in config.excluded_channels]
    
    # Get raw signals and timestamps
    raw_signals = data[eeg_channels].values
    time_seconds = data['time_seconds'].values
    
    # Apply filters
    notch_filtered, bandpass_filtered = apply_filters(raw_signals, eeg_channels, time_seconds)
    
    # Create output directory
    trial_output_dir = os.path.join(config.output_dir, f'Trial_{trial_num:02d}')
    os.makedirs(trial_output_dir, exist_ok=True)
    
    # Plot and save
    fig_raw, fig_filtered = plot_all_eeg_channels(raw_signals, notch_filtered, eeg_channels, time_seconds)
    
    fig_raw.savefig(os.path.join(trial_output_dir, 'eeg_all_channels_raw.png'), 
                    dpi=300, bbox_inches='tight')
    fig_filtered.savefig(os.path.join(trial_output_dir, 'eeg_all_channels_filtered.png'), 
                        dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print(f"Saved plots for Trial {trial_num}")

def main():
    """Main processing function"""
    # Setup configuration
    config = Config().setup()
    
    # Get number of trials from user
    n_trials = int(input("Enter number of trials to process: "))
    
    # Process each trial
    for trial in range(1, n_trials + 1):
        print(f"\nProcessing trial {trial}...")
        process_trial(trial, config)

if __name__ == "__main__":
    main()
