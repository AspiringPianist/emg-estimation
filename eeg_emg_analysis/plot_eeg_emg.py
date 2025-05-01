import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import os
import numpy as np
"""
Author: Unnath Chittimalla
Date: 10th April 2025

Purpose: Visualize EEG and EMG data from CSV files.
"""
# Set the paths for the data files
eeg_file_path = os.path.join('EEG_EMG_Trials_22nd_April','EEG_EMG_Trials', 'Trial_04', 'eeg_data.csv')
emg_file_path = os.path.join('EEG_EMG_Trials_22nd_April','EEG_EMG_Trials', 'Trial_04', 'emg_data.csv')

# Read the CSV files
print(f"Reading EEG data from {eeg_file_path}...")
eeg_data = pd.read_csv(eeg_file_path)
print(f"Reading EMG data from {emg_file_path}...")
emg_data = pd.read_csv(emg_file_path)

# Convert timestamps to datetime objects
# Since the timestamp only contains time information, we'll add a reference date
reference_date = dt.datetime.today().date()

# Function to convert time strings to datetime objects
def convert_time(time_str, ref_date):
    time_parts = time_str.split(':')
    hour = int(time_parts[0])
    minute = int(time_parts[1])
    # Handle seconds and microseconds
    second_parts = time_parts[2].split('.')
    second = int(second_parts[0])
    microsecond = int(second_parts[1]) * 1000 if len(second_parts) > 1 else 0
    
    return dt.datetime.combine(ref_date, dt.time(hour, minute, second, microsecond))

# Apply conversion to both datasets
eeg_data['datetime'] = eeg_data['timestamp'].apply(lambda x: convert_time(x, reference_date))
emg_data['datetime'] = emg_data['timestamp'].apply(lambda x: convert_time(x, reference_date))

# Print the first few rows of each dataset to understand their structure
print("\nEEG data preview:")
print(eeg_data.head())
print("\nEMG data preview:")
print(emg_data.head())

# Select columns for plotting (exclude timestamp and datetime)
eeg_columns = [col for col in eeg_data.columns if col not in ['timestamp', 'datetime']]
emg_columns = [col for col in emg_data.columns if col not in ['timestamp', 'datetime']]

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Plot EEG data on the left subplot
# Use a colormap to distinguish different channels
cmap_eeg = plt.cm.viridis
eeg_colors = [cmap_eeg(i) for i in np.linspace(0, 1, len(eeg_columns))]

for i, col in enumerate(eeg_columns):
    ax1.plot(eeg_data['datetime'], eeg_data[col], label=col, color=eeg_colors[i], linewidth=1, alpha=0.7)
    
# Format the x-axis to show timestamps properly
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax1.xaxis.set_major_locator(mdates.SecondLocator(interval=5))  # Adjust interval based on your data span
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add labels and title for EEG plot
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude (μV)')
ax1.set_title('EEG Data - Trial 01', fontsize=14, fontweight='bold')

# Add legend outside the plot area to avoid cluttering
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize='small')
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot EMG data on the right subplot
cmap_emg = plt.cm.plasma
emg_colors = [cmap_emg(i) for i in np.linspace(0, 1, len(emg_columns))]

for i, col in enumerate(emg_columns):
    ax2.plot(emg_data['datetime'], emg_data[col], label=col, color=emg_colors[i], linewidth=1.5)
    
# Format the x-axis to show timestamps properly
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax2.xaxis.set_major_locator(mdates.SecondLocator(interval=5))  # Adjust interval based on your data span
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add labels and title for EMG plot
ax2.set_xlabel('Time')
ax2.set_ylabel('Amplitude (mV)')
ax2.set_title('EMG Data - Trial 01', fontsize=14, fontweight='bold')

# Add legend outside the plot area
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize='small')
ax2.grid(True, linestyle='--', alpha=0.7)

# Add stats or additional information
ax1.text(0.02, 0.98, f"Channels: {len(eeg_columns)}\nSamples: {len(eeg_data)}", 
         transform=ax1.transAxes, fontsize=9, va='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax2.text(0.02, 0.98, f"Channels: {len(emg_columns)}\nSamples: {len(emg_data)}", 
         transform=ax2.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Add a main title
fig.suptitle('EEG and EMG Data Visualization - Subject 1, Trial 01', fontsize=18, fontweight='bold')

# Adjust the layout
plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.2)  # Make space for title and legends

# Add timestamp of analysis
plt.figtext(0.5, 0.01, f"Analysis performed: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
            ha='center', fontsize=9, style='italic')

# Save the figure with higher DPI for better quality
plt.savefig('eeg_emg_trial_01.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'eeg_emg_trial_01.png'")

# Show the plot
plt.show()

# Create a second visualization with stacked subplots for selected channels
# This is optional but can help in visualizing temporal correlations between EEG and EMG
print("\nCreating stacked channel visualization...")

# Select a subset of channels for better visibility
selected_eeg_channels = eeg_columns[:4]  # first 4 EEG channels
selected_emg_channels = emg_columns[:4]  # first 4 EMG channels

# Create a figure with stacked subplots
fig2, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Plot selected EEG channels on the top subplot
for i, col in enumerate(selected_eeg_channels):
    axs[0].plot(eeg_data['datetime'], eeg_data[col], label=col, color=cmap_eeg(i/len(selected_eeg_channels)), linewidth=1)

axs[0].set_ylabel('EEG Amplitude (μV)')
axs[0].set_title('Selected EEG Channels', fontweight='bold')
axs[0].legend(loc='upper right')
axs[0].grid(True, linestyle='--', alpha=0.7)

# Plot selected EMG channels on the bottom subplot
for i, col in enumerate(selected_emg_channels):
    axs[1].plot(emg_data['datetime'], emg_data[col], label=col, color=cmap_emg(i/len(selected_emg_channels)), linewidth=1.5)

axs[1].set_ylabel('EMG Amplitude (mV)')
axs[1].set_xlabel('Time')
axs[1].set_title('Selected EMG Channels', fontweight='bold')
axs[1].legend(loc='upper right')
axs[1].grid(True, linestyle='--', alpha=0.7)

# Format the x-axis to show timestamps properly
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
axs[1].xaxis.set_major_locator(mdates.SecondLocator(interval=2))
plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add a main title
fig2.suptitle('Stacked EEG and EMG Channel Comparison - Trial 01', fontsize=16, fontweight='bold')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save this figure as well
plt.savefig('eeg_emg_stacked_trial_01.png', dpi=300, bbox_inches='tight')
print("Stacked visualization saved as 'eeg_emg_stacked_trial_01.png'")

# Show the plot
plt.show()

