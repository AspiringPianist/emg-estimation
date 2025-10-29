# tune_params.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import optuna
import os

from data_preparation import prepare_all_emg, WINDOW_SAMPLES
from my_model import EMGSeq2SeqNet

# --- Constants for Tuning ---
INPUT_CHANNELS_TUNE = ['emg1', 'emg2']
TARGET_CHANNEL_TUNE = 'emg3'
NUM_INPUT_CHANNELS = len(INPUT_CHANNELS_TUNE)

SEQUENCE_LEN = 3 # CHANGED
OUTPUT_LEN = SEQUENCE_LEN * WINDOW_SAMPLES
BATCH_SIZE = 64
N_TRIALS = 30
TUNING_EPOCHS = 50

# ... (The rest of the file is identical and doesn't need changing) ...
# [The EMGDataset class, criterion, and objective function remain the same]

class EMGDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

criterion = nn.MSELoss()

def objective(trial, X_train, y_train, X_val, y_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'lstm_hidden': trial.suggest_categorical('lstm_hidden', [64, 128, 256]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.7),
    }
    train_ds = EMGDataset(X_train, y_train)
    val_ds = EMGDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE * 2)
    model = EMGSeq2SeqNet(input_channels=NUM_INPUT_CHANNELS, time_len=WINDOW_SAMPLES, output_len=OUTPUT_LEN, lstm_hidden=params['lstm_hidden'], dropout_rate=params['dropout_rate']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    for epoch in range(TUNING_EPOCHS):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in val_dl:
            xb = xb.to(device)
            preds = model(xb)
            all_preds.append(preds.cpu().numpy())
    y_val_pred = np.concatenate(all_preds)
    if np.var(y_val) == 0: r2 = -1.0
    else: r2 = r2_score(y_val, y_val_pred)
    trial.report(r2, epoch)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    return r2

if __name__ == "__main__":
    print(f"Loading data for tuning with inputs={INPUT_CHANNELS_TUNE}, target={TARGET_CHANNEL_TUNE}")
    X_data, Y_data = prepare_all_emg(sequence_len=SEQUENCE_LEN, input_channels=INPUT_CHANNELS_TUNE, target_channel=TARGET_CHANNEL_TUNE)
    if X_data.shape[0] < 10:
        print("Error: Not enough data sequences were created. Cannot run tuning.")
        exit()
    X_train_full, _, y_train_full, _ = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, random_state=42)
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=N_TRIALS)
    print("\n--- Hyperparameter Tuning Complete ---")
    trial = study.best_trial
    print(f"  Value (R²): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")