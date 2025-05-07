import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import mne

def load_emg_data(file_path):
    """Load emg data from CSV file"""
    print(f"Reading emg data from {file_path}...")
    data = pd.read_csv(file_path)
    return data

def apply_filters(data, ch_names, sfreq=100):
    """Apply MNE filters to remove 50Hz power line noise and apply bandpass"""
    # Create MNE raw object for filtering
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='emg')
    raw = mne.io.RawArray(data.T, info)
    
    # Apply notch filter at 50Hz
    raw_notch = raw.copy().notch_filter(freqs=50,verbose=False)
    
    # Apply bandpass filter (optional)
    raw_filtered = raw_notch.copy().filter(l_freq=1, h_freq=40, verbose=False)
    
    return raw_notch.get_data().T, raw_filtered.get_data().T

def compute_psd(data, fs=100, nperseg=4096):
    """Compute power spectral density"""
    f, pxx = signal.welch(data, fs=fs, nperseg=nperseg)
    return f, pxx

def plot_comparison(raw_signal, filtered_signal, fs=100, channel_name='emg'):
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
    zoom_duration = int(fs * 0.1)  # 100ms
    ax2.plot(time[zoom_start:zoom_start+zoom_duration], 
             raw_signal[zoom_start:zoom_start+zoom_duration], 
             'b-', label='Raw', alpha=0.7)
    ax2.plot(time[zoom_start:zoom_start+zoom_duration], 
             filtered_signal[zoom_start:zoom_start+zoom_duration], 
             'r-', label='Filtered', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude (μV)')
    ax2.set_title('Zoomed View (100ms)')
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
    ax3.set_xlim(0, 100)
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

def plot_all_emg_channels(raw_data, filtered_data, channels, fs=100, duration=10):
    """Plot all emg channels in subplots - raw and filtered separately"""
    n_channels = len(channels)
    n_rows = 6
    n_cols = 4
    
    # Plot raw data
    fig_raw = plt.figure(figsize=(20, 15))
    fig_raw.suptitle('Raw emg Data - All Channels', fontsize=16)
    
    for i, channel in enumerate(channels):
        ax = plt.subplot(n_rows, n_cols, i+1)
        time = np.arange(len(raw_data[:, i]))/fs
        ax.plot(time[:int(duration*fs)], raw_data[:int(duration*fs), i], 'b-', linewidth=0.5)
        ax.set_title(f'Channel {channel}')
        ax.set_xlabel('Time (s)' if i >= (n_channels-n_cols) else '')
        ax.set_ylabel('Amplitude (μV)')
        ax.grid(True)
    
    plt.tight_layout()
    
    # Plot filtered data
    fig_filtered = plt.figure(figsize=(20, 15))
    fig_filtered.suptitle('Notch Filtered emg Data - All Channels', fontsize=16)
    
    for i, channel in enumerate(channels):
        ax = plt.subplot(n_rows, n_cols, i+1)
        time = np.arange(len(filtered_data[:, i]))/fs
        ax.plot(time[:int(duration*fs)], filtered_data[:int(duration*fs), i], 'r-', linewidth=0.5)
        ax.set_title(f'Channel {channel}')
        ax.set_xlabel('Time (s)' if i >= (n_channels-n_cols) else '')
        ax.set_ylabel('Amplitude (μV)')
        ax.grid(True)
    
    plt.tight_layout()
    return fig_raw, fig_filtered

def main():
    # Load emg data
    for trial in range(1, 26):
        print(f"Processing trial {trial}...")
        emg_file_path = os.path.join('./EEG_EMG_Trials_28thApril', f'Trial_{trial:02d}', 'emg_data.csv')
        data = load_emg_data(emg_file_path)
        
        # Get EMG channels (excluding timestamp and datetime)
        EMG_channels = [col for col in data.columns if col not in ['timestamp', 'datetime']]
        
        # Get raw signals
        raw_signals = data[EMG_channels].values
        
        # Apply MNE filters to all channels at once
        # notch_filtered, bandpass_filtered = apply_filters(raw_signals, EMG_channels, sfreq=100)
        notch_filtered = raw_signals.copy()
        
        # Plot all channels together
        fig_raw, fig_filtered = plot_all_emg_channels(raw_signals, notch_filtered, EMG_channels)
        
        # Save the multi-channel plots
        if not os.path.exists('28th_April_Visuals_EMG'):
            os.makedirs('28th_April_Visuals_EMG')
        if not os.path.exists(f'28th_April_Visuals_EMG/Trial_{trial:02d}'):
            os.makedirs(f'28th_April_Visuals_EMG/Trial_{trial:02d}')
        fig_raw.savefig(f'28th_April_Visuals_EMG/Trial_{trial:02d}/EMG_all_channels_raw.png', dpi=300, bbox_inches='tight')
        fig_filtered.savefig(f'28th_April_Visuals_EMG/Trial_{trial:02d}/EMG_all_channels_filtered.png', dpi=300, bbox_inches='tight')
        plt.close('all')
        
        print("Saved multi-channel plots")
        
        # # Original single-channel processing
        # for i, channel in enumerate(EMG_channels):
        #     print(f"\nProcessing channel: {channel}")
            
        #     # Get signals for this channel
        #     raw_signal = raw_signals[:, i]
        #     filtered_signal = notch_filtered[:, i]  # or use bandpass_filtered[:, i] if you want both filters
            
        #     # Plot and save comparison
        #     fig = plot_comparison(raw_signal, filtered_signal, fs=250, channel_name=channel)
        #     fig.savefig(f'28th_April_Visuals_EMG/Trial_{trial:02d}/filter_comparison_{channel}.png', dpi=300, bbox_inches='tight')
        #     plt.close(fig)
            
        #     print(f"Saved comparison plot for {channel}")

if __name__ == "__main__":
    main()
