#!/usr/bin/env python3
"""
Author: Unnath Chittimalla
Date: 10th April 2025

FFT Analysis for EEG and EMG Data

This script loads EEG and EMG data from CSV files, computes FFT, and displays
side-by-side plots of the magnitude spectra.

- EEG Data: Sampled at 250 Hz (Nyquist frequency = 125 Hz)
- EMG Data: Sampled at 100 Hz (Nyquist frequency = 50 Hz)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import windows
import os

def load_data(file_path):
    """
    Load data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def compute_fft(signal_data, fs, window_function=windows.hann):
    """
    Compute FFT with windowing.
    
    Args:
        signal_data (numpy.ndarray): Input time domain signal
        fs (float): Sampling frequency in Hz
        window_function (function): Window function to apply
        
    Returns:
        tuple: (frequencies, magnitude spectrum)
    """
    # Remove DC offset
    signal_data = signal_data - np.mean(signal_data)
    
    # Apply window function to reduce spectral leakage
    window = window_function(len(signal_data))
    windowed_signal = signal_data * window
    
    # Compute FFT
    fft_result = np.fft.rfft(windowed_signal)
    
    # Compute magnitude spectrum
    magnitude = np.abs(fft_result)
    
    # Compute frequency axis
    frequencies = np.fft.rfftfreq(len(signal_data), d=1/fs)
    
    return frequencies, magnitude

def plot_all_emg_channels(emg_data, fs=100, duration=5):
    """Plot all EMG channels in subplots"""
    # Get EMG channels (excluding timestamp and datetime)
    emg_channels = [col for col in emg_data.columns if col.startswith('emg')]
    n_channels = len(emg_channels)
    
    # Calculate subplot layout
    n_rows = int(np.ceil(n_channels/2))
    n_cols = 2
    
    fig = plt.figure(figsize=(15, 4*n_rows))
    fig.suptitle('EMG Data - All Channels', fontsize=16)
    
    for i, channel in enumerate(emg_channels):
        ax = plt.subplot(n_rows, n_cols, i+1)
        time = np.arange(len(emg_data[channel]))/fs
        ax.plot(time[:int(duration*fs)], emg_data[channel][:int(duration*fs)], 'g-', linewidth=0.5)
        ax.set_title(f'Channel {channel}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
    
    plt.tight_layout()
    return fig

def main():
    # Define file paths and sampling rates
    eeg_file = os.path.join('EEG_EMG_Trials_25April_Biceps', 'Trial_04', 'eeg_data.csv')
    emg_file = os.path.join('EEG_EMG_Trials_25April_Biceps', 'Trial_04', 'emg_data.csv')
    
    eeg_fs = 250  # Hz
    emg_fs = 100  # Hz
    
    # Load data
    eeg_data = load_data(eeg_file)
    emg_data = load_data(emg_file)
    
    if eeg_data is None or emg_data is None:
        print("Error: Failed to load required data.")
        return
    
    print(f"EEG data shape: {eeg_data.shape}")
    print(f"EMG data shape: {emg_data.shape}")
    
    # Extract channel data (ch1 for EEG, emg1 for EMG)
    try:
        eeg_ch1 = eeg_data['ch1'].values
        emg_ch1 = emg_data['emg1'].values
    except KeyError as e:
        print(f"Error accessing channel data: {e}")
        print("Available EEG columns:", eeg_data.columns.tolist())
        print("Available EMG columns:", emg_data.columns.tolist())
        return
    
    # Plot all EMG channels
    fig_emg = plot_all_emg_channels(emg_data)
    fig_emg.savefig('emg_all_channels.png', dpi=300, bbox_inches='tight')
    plt.close(fig_emg)
    
    print("Saved EMG multi-channel plot")
    
    # Compute FFT for both signals
    eeg_freq, eeg_mag = compute_fft(eeg_ch1, eeg_fs)
    emg_freq, emg_mag = compute_fft(emg_ch1, emg_fs)
    
    # Create subplots for side-by-side comparison
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot EEG FFT (frequency range: 0 to 125 Hz)
    axs[0].plot(eeg_freq, eeg_mag)
    axs[0].set_xlim(0, min(125, max(eeg_freq)))  # Limit to Nyquist frequency
    axs[0].set_title('EEG FFT Magnitude Spectrum')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Magnitude')
    axs[0].grid(True)
    
    # Plot EMG FFT (frequency range: 0 to 50 Hz)
    axs[1].plot(emg_freq, emg_mag)
    axs[1].set_xlim(0, min(50, max(emg_freq)))  # Limit to Nyquist frequency
    axs[1].set_title('EMG FFT Magnitude Spectrum')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Magnitude')
    axs[1].grid(True)
    
    # Adjust layout and display plot
    plt.tight_layout()
    plt.suptitle('FFT Analysis of EEG (250Hz) and EMG (100Hz) Signals', y=1.05)
    plt.show()
    
    print("FFT analysis complete. Plot displayed.")

if __name__ == "__main__":
    main()

