import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from mne import create_info, EvokedArray
from mne.channels import make_dig_montage
from mne.viz import plot_topomap
from mne.decoding import CSP
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import glob
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress MNE warnings

def butter_lowpass(cutoff, fs, order=4):
    """Designs a Butterworth lowpass filter."""
    nyq = 0.5 * fs
    cutoff = cutoff / nyq
    b, a = butter(order, cutoff, btype='low')
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=4):
    """Applies a lowpass filter to the data."""
    b, a = butter_lowpass(cutoff, fs, order=order)
    filtered = filtfilt(b, a, data)
    return filtered

def load_data(data_dir):
    """Loads EEG and EMG CSV files from the specified directory."""
    eeg_files = sorted(glob.glob(os.path.join(data_dir, "eeg_clean_*.csv")))
    emg_files = sorted(glob.glob(os.path.join(data_dir, "emg_clean_*.csv")))
    
    if not eeg_files or not emg_files:
        raise ValueError(f"No EEG ({len(eeg_files)}) or EMG ({len(emg_files)}) files found in {data_dir}")
    
    if len(eeg_files) != len(emg_files):
        raise ValueError(f"Mismatch in number of EEG ({len(eeg_files)}) and EMG ({len(emg_files)}) files")
    
    eeg_dfs = []
    emg_dfs = []
    file_ids = []
    
    for eeg_file, emg_file in zip(eeg_files, emg_files):
        eeg_id = os.path.basename(eeg_file).replace("eeg_clean_", "").replace(".csv", "")
        emg_id = os.path.basename(emg_file).replace("emg_clean_", "").replace(".csv", "")
        if eeg_id != emg_id:
            raise ValueError(f"Mismatched file IDs: {eeg_file} and {emg_file}")
        
        eeg_df = pd.read_csv(eeg_file)
        emg_df = pd.read_csv(emg_file)
        
        print(f"Loaded file ID {eeg_id}:")
        print(f"  EEG shape: {eeg_df.shape}, columns: {eeg_df.columns.tolist()}, dtype: {eeg_df.dtypes.get('eeg1', 'N/A')}")
        print(f"  EMG shape: {emg_df.shape}, columns: {emg_df.columns.tolist()}, dtype: {emg_df.dtypes.get('emg1', 'N/A')}")
        
        eeg_dfs.append(eeg_df)
        emg_dfs.append(emg_df)
        file_ids.append(eeg_id)
    
    return eeg_dfs, emg_dfs, file_ids

def detect_spikes(emg_data, emg_time, emg_sfreq, lowcut=0.8, threshold_mult=2.0, min_distance=0.05):
    """Detects the largest EMG spike using rectified and low-pass filtered signal."""
    if np.all(np.isnan(emg_data)):
        print("  Warning: EMG data contains only NaN values")
        return np.array([])
    
    valid_mask = ~np.isnan(emg_data)
    if not np.any(valid_mask):
        print("  Warning: No valid (non-NaN) EMG samples")
        return np.array([])
    
    emg_data_valid = emg_data[valid_mask]
    
    # Option 1: Rectify then low-pass
    rectified = np.abs(emg_data_valid)
    envelope_rect_lowpass = apply_lowpass_filter(rectified, lowcut, emg_sfreq)
    smoothness_rect_lowpass = np.var(np.diff(envelope_rect_lowpass))
    
    # Option 2: Low-pass then rectify
    lowpassed = apply_lowpass_filter(emg_data_valid, lowcut, emg_sfreq)
    envelope_lowpass_rect = np.abs(lowpassed)
    smoothness_lowpass_rect = np.var(np.diff(envelope_lowpass_rect))
    
    # Choose smoother envelope
    if smoothness_rect_lowpass < smoothness_lowpass_rect:
        envelope = envelope_rect_lowpass
        method = "rectify then low-pass"
    else:
        envelope = envelope_lowpass_rect
        method = "low-pass then rectify"
    
    print(f"  Selected envelope method: {method} (smoothness: {min(smoothness_rect_lowpass, smoothness_lowpass_rect):.4f})")
    
    # Detect largest spike
    threshold = np.std(envelope) * threshold_mult
    distance_samples = max(1, int(emg_sfreq * min_distance))
    peaks, properties = find_peaks(envelope, height=threshold, distance=distance_samples)
    
    if len(peaks) == 0:
        print("  No spikes detected above threshold")
        return np.array([])
    
    # Select largest peak
    max_peak_idx = np.argmax(properties['peak_heights'])
    largest_peak = peaks[max_peak_idx]
    largest_height = properties['peak_heights'][max_peak_idx]
    
    print(f"  Detected 1 largest spike with threshold {threshold:.2f}")
    print(f"  Largest peak height: {largest_height:.4f} at index {largest_peak}")
    
    valid_indices = np.where(valid_mask)[0]
    original_peak = valid_indices[largest_peak]
    
    return np.array([original_peak])

def extract_segments(eeg_data, eeg_time, emg_data, emg_time, eeg_sfreq, emg_sfreq, peaks, samples_per_segment=250):
    """Extracts EEG segments around the largest EMG spike and non-spike regions to its left and right."""
    half_window_emg = int(emg_sfreq * 0.25)  # 0.25s = 25 samples at 100 Hz
    half_window_eeg = int(half_window_emg * (eeg_sfreq / emg_sfreq))  # ~63 samples at 250 Hz
    segment_half_eeg = samples_per_segment // 2  # 125 samples for 250-sample segment
    
    spike_segments = []
    non_spike_segments = []
    spike_times = []
    
    print(f"  Expected EEG segment length: {samples_per_segment} samples, exclusion half_window_eeg: {half_window_eeg}")
    
    # Extract spike segment (single largest peak)
    if len(peaks) > 0:
        peak_idx = peaks[0]
        peak_time = emg_time[peak_idx]
        closest_eeg_idx = np.argmin(np.abs(eeg_time - peak_time))
        eeg_start_idx = closest_eeg_idx - segment_half_eeg
        eeg_end_idx = closest_eeg_idx + segment_half_eeg
        
        print(f"  Largest spike at time {peak_time:.2f}s, EEG idx {closest_eeg_idx}, start {eeg_start_idx}, end {eeg_end_idx}")
        
        if eeg_start_idx < 0:
            print(f"  Adjusting start index from {eeg_start_idx} to 0")
            eeg_start_idx = 0
            eeg_end_idx = samples_per_segment
        elif eeg_end_idx > len(eeg_time):
            print(f"  Adjusting end index from {eeg_end_idx} to {len(eeg_time)}")
            eeg_end_idx = len(eeg_time)
            eeg_start_idx = eeg_end_idx - samples_per_segment
        
        if eeg_end_idx - eeg_start_idx >= samples_per_segment:
            segment = eeg_data[:, eeg_start_idx:eeg_start_idx + samples_per_segment]
            if not np.any(np.isnan(segment)):
                spike_segments.append(segment)
                spike_times.append((
                    emg_time[max(0, peak_idx - half_window_emg)],
                    emg_time[min(len(emg_time)-1, peak_idx + half_window_emg)]
                ))
            else:
                print(f"  Skipped spike segment at time {peak_time:.2f}s due to NaN values")
        else:
            print(f"  Skipped spike segment at time {peak_time:.2f}s due to insufficient length ({eeg_end_idx - eeg_start_idx} samples)")
    
    # Extract non-spike segments (left and right of spike, or default positions if no spike)
    if spike_times:
        spike_time = (spike_times[0][0] + spike_times[0][1]) / 2  # Center of exclusion zone
        # Left non-spike: Center at spike_time - 0.25s - 0.5s
        left_center_time = spike_time - 0.25 - 0.5
        left_center_idx = np.argmin(np.abs(eeg_time - left_center_time))
        left_start_idx = left_center_idx - segment_half_eeg
        left_end_idx = left_center_idx + segment_half_eeg
        
        if left_start_idx >= 0 and left_end_idx <= len(eeg_time):
            left_segment = eeg_data[:, left_start_idx:left_start_idx + samples_per_segment]
            if not np.any(np.isnan(left_segment)):
                non_spike_segments.append(left_segment)
                print(f"  Left non-spike segment centered at {left_center_time:.2f}s, idx {left_center_idx}, start {left_start_idx}, end {left_end_idx}")
            else:
                print(f"  Skipped left non-spike segment at {left_center_time:.2f}s due to NaN values")
        else:
            print(f"  Skipped left non-spike segment at {left_center_time:.2f}s due to out-of-bounds (start {left_start_idx}, end {left_end_idx})")
        
        # Right non-spike: Center at spike_time + 0.25s + 0.5s
        right_center_time = spike_time + 0.25 + 0.5
        right_center_idx = np.argmin(np.abs(eeg_time - right_center_time))
        right_start_idx = right_center_idx - segment_half_eeg
        right_end_idx = right_center_idx + segment_half_eeg
        
        if right_start_idx >= 0 and right_end_idx <= len(eeg_time):
            right_segment = eeg_data[:, right_start_idx:right_start_idx + samples_per_segment]
            if not np.any(np.isnan(right_segment)):
                non_spike_segments.append(right_segment)
                print(f"  Right non-spike segment centered at {right_center_time:.2f}s, idx {right_center_idx}, start {right_start_idx}, end {right_end_idx}")
            else:
                print(f"  Skipped right non-spike segment at {right_center_time:.2f}s due to NaN values")
        else:
            print(f"  Skipped right non-spike segment at {right_center_time:.2f}s due to out-of-bounds (start {right_start_idx}, end {right_end_idx})")
    else:
        # No spike detected: Extract two non-spike segments at default positions (e.g., 2.5s and 7.5s)
        for center_time in [2.5, 7.5]:
            center_idx = np.argmin(np.abs(eeg_time - center_time))
            start_idx = center_idx - segment_half_eeg
            end_idx = center_idx + segment_half_eeg
            
            if start_idx >= 0 and end_idx <= len(eeg_time):
                segment = eeg_data[:, start_idx:start_idx + samples_per_segment]
                if not np.any(np.isnan(segment)):
                    non_spike_segments.append(segment)
                    print(f"  Default non-spike segment centered at {center_time:.2f}s, idx {center_idx}, start {start_idx}, end {end_idx}")
                else:
                    print(f"  Skipped default non-spike segment at {center_time:.2f}s due to NaN values")
            else:
                print(f"  Skipped default non-spike segment at {center_time:.2f}s due to out-of-bounds (start {start_idx}, end {end_idx})")
    
    print(f"  Extracted {len(spike_segments)} spike segments, {len(non_spike_segments)} non-spike segments")
    
    return spike_segments, non_spike_segments

def prepare_csp_dataset(eeg_dfs, emg_dfs, file_ids, eeg_sfreq=250.0, emg_sfreq=100.0):
    """Prepares CSP dataset with spike and non-spike EEG segments, using only EEG channels."""
    # Define the 19 EEG channels from the MCSCap montage
    eeg_channel_names = ['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'T3', 'T4', 'C3', 'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']
    
    X_spike = []
    X_non_spike = []
    file_id_labels = []
    
    for eeg_df, emg_df, file_id in zip(eeg_dfs, emg_dfs, file_ids):
        print(f"\nProcessing file ID {file_id}")
        
        if 'time' not in eeg_df.columns or 'time' not in emg_df.columns:
            raise ValueError(f"Missing 'time' column in file ID {file_id}")
        
        # Get all columns that start with 'eeg'
        all_eeg_columns = [col for col in eeg_df.columns if col.startswith('eeg')]
        print(f"  All EEG columns: {all_eeg_columns}, count: {len(all_eeg_columns)}")
        
        # Map eeg1 to eeg19 to the montage channels (assuming eeg1=Fp1, eeg2=Fp2, ..., eeg19=Pz)
        eeg_columns = []
        if len(all_eeg_columns) >= 19:
            eeg_columns = all_eeg_columns[:19]  # Take first 19 channels
            print(f"  Selected EEG channels: {eeg_columns}")
        else:
            raise ValueError(f"File ID {file_id} has {len(all_eeg_columns)} EEG channels, need at least 19")
        
        # Warn about extra channels (including IMU)
        if len(all_eeg_columns) > 19:
            extra_columns = all_eeg_columns[19:]
            print(f"  Warning: Excluding extra channels (possibly IMU, REF, etc.): {extra_columns}")
        
        emg_channels = [col for col in emg_df.columns if col.startswith('emg')]
        if not emg_channels:
            raise ValueError(f"No EMG channels found in file ID {file_id}")
        
        eeg_time = eeg_df['time'].values
        emg_time = emg_df['time'].values
        
        if len(eeg_time) > 1:
            eeg_dt = np.mean(np.diff(eeg_time))
            if abs(eeg_dt - 0.004) > 1e-6:
                print(f"  Warning: EEG file {file_id} has time interval {eeg_dt:.6f}s (expected 0.004s)")
        
        if len(emg_time) > 1:
            emg_dt = np.mean(np.diff(emg_time))
            if abs(emg_dt - 0.01) > 1e-6:
                print(f"  Warning: EMG file {file_id} has time interval {emg_dt:.6f}s (expected 0.01s)")
        
        if len(eeg_time) < 250:
            print(f"  Warning: EEG file {file_id} too short ({len(eeg_time)} samples, need >=250)")
            continue
        
        if len(emg_time) < 100:
            print(f"  Warning: EMG file {file_id} too short ({len(emg_time)} samples, need >=100)")
            continue
        
        # Use only the 19 EEG channels
        eeg_data = eeg_df[eeg_columns].values.T.astype(np.float64)  # Shape: (19, samples)
        emg_data = emg_df[emg_channels[0]].values.astype(np.float64)
        
        print(f"  EEG data shape: {eeg_data.shape}, dtype: {eeg_data.dtype}")
        print(f"  EMG data shape: {emg_data.shape}, dtype: {emg_data.dtype}")
        
        peaks = detect_spikes(emg_data, emg_time, emg_sfreq)
        spike_segments, non_spike_segments = extract_segments(
            eeg_data, eeg_time, emg_data, emg_time, eeg_sfreq, emg_sfreq, peaks
        )
        
        X_spike.extend(spike_segments)
        X_non_spike.extend(non_spike_segments)
        file_id_labels.extend([file_id] * len(spike_segments))
        file_id_labels.extend([file_id] * len(non_spike_segments))
    
    return np.array(X_spike), np.array(X_non_spike), file_id_labels, eeg_channel_names

def plot_csp_patterns(csp, eeg_channel_names, sfreq=250.0):
    """
    Plots topographical maps of CSP patterns using standard 10-20 EEG layout.
    Uses user's custom EEG channel names (e.g., eeg1, eeg2, ..., eeg19) and maps
    them to standard positions (Fp1, Fp2, ..., Pz) in 10-20 system.
    """
    # Standard 10-20 order
    standard_10_20 = [
        'Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'T3', 'T4', 'C3', 'C4',
        'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz'
    ]

    if len(eeg_channel_names) != len(standard_10_20):
        raise ValueError(f"Expected {len(standard_10_20)} EEG channels, but got {len(eeg_channel_names)}")

    # Accurate theta/phi values from MNE's standard_1020 montage
    theta_phi_map = {
        'Fp1': (94.49, 109.33), 'Fp2': (94.50, 70.61),
        'F7': (97.92, 148.85), 'F8': (97.99, 31.31),
        'F3': (60.01, 133.41), 'F4': (61.47, 46.33),
        'T3': (96.23, -169.22), 'T4': (96.27, -10.01),
        'C3': (45.89, -169.91), 'C4': (46.92, -9.22),
        'T5': (91.38, -134.6), 'T6': (91.41, -45.0),
        'P3': (59.5, -123.93), 'P4': (59.57, -54.68),
        'O1': (85.65, -104.66), 'O2': (85.66, -75.1),
        'Fz': (41.36, 89.69), 'Cz': (5.23, -87.5), 'Pz': (44.48, -89.77)
    }

    # Build custom montage using your channel names mapped to standard positions
    r = 0.09  # head radius in meters
    pos_dict = {}
    for user_ch, standard_ch in zip(eeg_channel_names, standard_10_20):
        theta, phi = theta_phi_map[standard_ch]
        theta_rad = np.radians(theta)
        phi_rad = np.radians(phi)
        x = r * np.sin(theta_rad) * np.cos(phi_rad)
        y = r * np.sin(theta_rad) * np.sin(phi_rad)
        z = r * np.cos(theta_rad)
        pos_dict[user_ch] = [x, y, z]

    # Create montage and MNE Info
    montage = make_dig_montage(ch_pos=pos_dict, coord_frame='head')
    info = create_info(ch_names=eeg_channel_names, sfreq=sfreq, ch_types='eeg')
    info.set_montage(montage)

    # FIXED: Ensure we always plot exactly 4 components
    n_components_to_plot = csp.n_components
    print(f"Plotting {n_components_to_plot} CSP components (out of {csp.n_components} available)")
    
    # Create figure with proper size for components
    fig, axes = plt.subplots(1, n_components_to_plot, figsize=(5 * n_components_to_plot, 5))
    if n_components_to_plot == 1:
        axes = [axes]

    for i in range(n_components_to_plot):
        pattern = csp.patterns_[i, :].reshape(-1, 1)
        evoked = EvokedArray(pattern, info)
        plot_topomap(
            evoked.data[:, 0], evoked.info,
            axes=axes[i],
            names=eeg_channel_names,
            ch_type='eeg',
            res=64,
            sensors='ko',
            show=False,  # FIXED: Don't show immediately
            extrapolate='head',
            sphere=(0.0, 0.0, 0.0, 0.09)
        )
        axes[i].set_title(f'CSP Component {i+1}')

    plt.tight_layout()
    
    # FIXED: Save before closing and add better error handling
    try:
        plt.savefig('csp_patterns_topomap.png', dpi=300, bbox_inches='tight')
        print("CSP patterns topographical map saved to 'csp_patterns_topomap.png'")
    except Exception as e:
        print(f"Error saving image: {e}")
    finally:
        plt.show()  # Show the plot
        plt.close()

def compute_metrics(y_true, y_pred):
    """Computes classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    cm = confusion_matrix(y_true, y_pred)
    return metrics, cm

def main():
    data_dir = "./cleaned_data"
    eeg_sfreq = 250.0
    emg_sfreq = 100.0
    
    try:
        eeg_dfs, emg_dfs, file_ids = load_data(data_dir)
        X_spike, X_non_spike, file_id_labels, eeg_channel_names = prepare_csp_dataset(
            eeg_dfs, emg_dfs, file_ids, eeg_sfreq, emg_sfreq
        )
        
        if len(X_spike) == 0 and len(X_non_spike) == 0:
            raise ValueError("No valid segments extracted for CSP analysis. Check debug output above.")
        
        print(f"\nTotal extracted: {len(X_spike)} spike segments, {len(X_non_spike)} non-spike segments")
            
        X = np.concatenate([X_spike, X_non_spike], axis=0).astype(np.float64)
        y = np.concatenate([np.ones(len(X_spike)), np.zeros(len(X_non_spike))])
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
        
        # FIXED: Force 4 components if we have enough data
        # CSP components should be <= min(n_channels, n_samples_per_class - 1)
        max_possible_components = min(X_train.shape[1], min(len(X_spike), len(X_non_spike)))
        n_components = min(4, max_possible_components)
        # n_components = max_possible_components
        print(f"Using {n_components} CSP components (max possible: {max_possible_components})")
        
        csp = CSP(n_components=n_components, reg='ledoit_wolf', log=True, norm_trace=False)
        rf = RandomForestClassifier(class_weight='balanced', n_estimators=1000)
        
        # Fit and transform
        X_train_csp = csp.fit_transform(X_train, y_train)
        X_test_csp = csp.transform(X_test)
        rf.fit(X_train_csp, y_train)
        y_pred = rf.predict(X_test_csp)
        
        # Plot CSP patterns
        plot_csp_patterns(csp, eeg_channel_names, eeg_sfreq)
        test_eeg_df = eeg_dfs[0]
        def plot_rf_predictions_on_eeg(eeg_df, rf, csp, eeg_channel_names, segment_length=250, step_size=50):
            eeg_data = eeg_df[eeg_channel_names].values.T  # shape: (n_channels, n_samples)
            times = eeg_df['time'].values
            n_samples = eeg_data.shape[1]
            preds = []
            pred_times = []

            for start in range(0, n_samples - segment_length + 1, step_size):
                end = start + segment_length
                segment = eeg_data[:, start:end]
                if np.any(np.isnan(segment)):
                    preds.append(np.nan)
                    pred_times.append(times[start + segment_length // 2])
                    continue
                features = csp.transform(segment[np.newaxis, :, :])  # shape: (1, n_features)
                pred = rf.predict(features)[0]
                preds.append(pred)
                pred_times.append(times[start + segment_length // 2])

            preds = np.array(preds)
            pred_times = np.array(pred_times)

            plt.figure(figsize=(15, 5))
            plt.plot(times, eeg_data.mean(axis=0), label='Mean EEG')
            plt.xlabel('Time (s)')
            plt.ylabel('EEG (uV)')
            plt.title('EEG with RF Model Predictions (Spike/No Spike)')

            for i in range(len(preds)):
                color = 'red' if preds[i] == 1 else 'green'
                plt.axvspan(pred_times[i] - (segment_length/2)/250, pred_times[i] + (segment_length/2)/250, 
                            color=color, alpha=0.2 if preds[i]==1 else 0.08)

            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', alpha=0.2, label='Spike (Predicted)'),
                            Patch(facecolor='green', alpha=0.08, label='No Spike (Predicted)')]
            plt.legend(handles=legend_elements)
            plt.tight_layout()
            plt.show()

        # 10. Call the plotting function


        # Compute metrics   
        metrics, cm = compute_metrics(y_test, y_pred)
        print("\nClassification Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'metric': ['accuracy', 'precision', 'recall', 'f1_score'],
            'value': [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
        })
        metrics_df.to_csv("classification_metrics.csv", index=False)
        print("Metrics saved to 'classification_metrics.csv'")
        
        # NEW
        plot_rf_predictions_on_eeg(test_eeg_df, rf, csp, eeg_channel_names)

        # Save CSP features
        csp_features_df = pd.DataFrame(
            X_train_csp,
            columns=[f'csp_feature_{i+1}' for i in range(n_components)]
        )
        csp_features_df['condition'] = ['spike' if y == 1 else 'non_spike' for y in y_train]
        csp_features_df['file_id'] = file_id_labels
        csp_features_df.to_csv("csp_features.csv", index=False)
        print("CSP features saved to 'csp_features.csv'")
        
        # Save CSP patterns
        patterns_df = pd.DataFrame(
            csp.patterns_[:n_components, :],
            columns=eeg_channel_names,
            index=[f'csp_component_{i+1}' for i in range(n_components)]
        )
        patterns_df.to_csv("csp_patterns.csv")
        print("CSP patterns saved to 'csp_patterns.csv'")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'segment_id': range(1, len(y_pred) + 1),
            'file_id': file_id_labels,
            'predicted_label': ['spike' if y == 1 else 'non_spike' for y in y_pred],
            'true_label': ['spike' if y == 1 else 'non_spike' for y in y_train]
        })
        predictions_df.to_csv("predictions.csv", index=False)
        print("Predictions saved to 'predictions.csv'")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()