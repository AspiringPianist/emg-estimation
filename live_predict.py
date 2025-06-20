import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model.tune_params import Seq2SeqModel, INPUT_DIM, SEQUENCE_LEN, OUTPUT_LEN
from data_preparation import prepare_segments, process_eeg_window, load_and_preprocess_csp_filter
import joblib

# Load scalers and model
eeg_scaler = joblib.load("eeg_scaler.pkl")
emg_scaler = joblib.load("emg_scaler.pkl")
model = Seq2SeqModel(INPUT_DIM, 128, OUTPUT_LEN)
model.load_state_dict(torch.load("seq2seq_model.pt", map_location="cpu"))
model.eval()

# Preload CSP features
csp = load_and_preprocess_csp_filter()

# --- Predict on new EEG/EMG pair ---
def predict_from_files(eeg_path, emg_path):
    eeg_df = pd.read_csv(eeg_path)
    emg_df = pd.read_csv(emg_path)
    
    # Get file ID and corresponding CSP features
    file_id = eeg_path.split('_')[-1].split('.')[0]
    
    eeg_seg, emg_true = prepare_segments(eeg_df, emg_df, csp)

    if eeg_seg is None:
        print("Insufficient data")
        return

    eeg_input = eeg_scaler.transform(eeg_seg).reshape(1, SEQUENCE_LEN, INPUT_DIM)
    eeg_input = torch.tensor(eeg_input, dtype=torch.float32)
    
    with torch.no_grad():
        pred_scaled = model(eeg_input).numpy()

    pred_emg = emg_scaler.inverse_transform(pred_scaled).squeeze()
    true_emg = emg_true[:OUTPUT_LEN]

    plt.figure(figsize=(12, 4))
    plt.plot(true_emg, label="True EMG", linewidth=1.2)
    plt.plot(pred_emg, label="Predicted EMG", linestyle="--")
    plt.legend()
    plt.title("10-second EMG Prediction")
    plt.show()

# Example usage:
predict_from_files("../data/cleaned_data/eeg_clean_11.csv", "../data/cleaned_data/emg_clean_11.csv")