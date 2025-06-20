import os
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from csp_cleaned import prepare_csp_dataset, load_data
from mne.decoding import CSP

EEG_CHANNELS = 19
FS_EEG = 250
FS_EMG = 100
CSP_COMPONENTS = 4
SEGMENT_DURATION = 2
EEG_WINDOW = FS_EEG * SEGMENT_DURATION
DATA_DIR = '../data/cleaned_data'


def load_and_preprocess_csp_filter():
    eeg_dfs, emg_dfs, file_ids = load_data(DATA_DIR)
    X_spike, X_non_spike, file_id_labels, eeg_channel_names = prepare_csp_dataset(
        eeg_dfs, emg_dfs, file_ids, FS_EEG, FS_EMG
    )
    X_csp = np.concatenate([X_spike, X_non_spike], axis=0)
    y_csp = np.concatenate([np.ones(len(X_spike)), np.zeros(len(X_non_spike))])
    csp = CSP(n_components=CSP_COMPONENTS, reg='ledoit_wolf', log=False, norm_trace=False)
    csp.fit(X_csp, y_csp)
    return csp


def prepare_segments(eeg_df, emg_df, csp=None):
    eeg_data = eeg_df.iloc[:, :EEG_CHANNELS].values
    emg_data = emg_df.iloc[:, 2].values
    emg_envelope = np.abs(hilbert(emg_data))
    eeg_segments = []
    total_samples = FS_EEG * 10

    if len(eeg_data) < total_samples or len(emg_envelope) < FS_EMG * 10:
        return None, None

    for i in range(0, total_samples, EEG_WINDOW):
        eeg_window = eeg_data[i:i + EEG_WINDOW]
        if len(eeg_window) != EEG_WINDOW:
            continue
        if csp is not None:
            eeg_window = eeg_window.T  # (channels, time)
            csp_filtered = np.dot(csp.filters_[:CSP_COMPONENTS], eeg_window).T  # (time, components)
            eeg_segments.append(csp_filtered)
        else:
            eeg_segments.append(eeg_window)

    return np.array(eeg_segments), emg_envelope[:FS_EMG * 10]


def prepare_all():
    eeg_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('eeg_clean_')])
    all_eeg, all_emg = [], []
    # csp = load_and_preprocess_csp_filter()

    for eeg_file in eeg_files:
        idx = eeg_file.split('_')[-1].split('.')[0]
        emg_file = f'emg_clean_{idx}.csv'
        eeg_path = os.path.join(DATA_DIR, eeg_file)
        emg_path = os.path.join(DATA_DIR, emg_file)
 
        if not os.path.exists(emg_path):
            continue

        eeg_df = pd.read_csv(eeg_path)
        emg_df = pd.read_csv(emg_path)

        eeg_seg, emg_seg = prepare_segments(eeg_df, emg_df, None) #change back to csp
        if eeg_seg is not None:
            all_eeg.append(eeg_seg)
            all_emg.append(emg_seg)

    return np.array(all_eeg), np.array(all_emg)