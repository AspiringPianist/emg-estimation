# train_all_combinations.py

import torch
import torch.optim as optim
import pandas as pd
import os
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from data_preparation import prepare_all_emg, WINDOW_SAMPLES
from my_model import EMGSeq2SeqNet

# --- Hyperparameters Adjusted for Data Length ---
SEQUENCE_LEN = 3 # CHANGED: Reduced from 5 to 3 to match data length
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 100
LSTM_HIDDEN = 128
DROPOUT_RATE = 0.5
NUM_INPUT_CHANNELS = 2

# --- Derived Constants ---
OUTPUT_LEN = SEQUENCE_LEN * WINDOW_SAMPLES
ALL_EMG_CHANNELS = [f'emg{i+1}' for i in range(8)]

class EMGDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def compute_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    if np.var(y_true) < 1e-6: r2 = 0
    else: r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    y_true_flat, y_pred_flat = y_true.flatten(), y_pred.flatten()
    if np.all(y_true_flat == y_true_flat[0]) or np.all(y_pred_flat == y_pred_flat[0]): corr = 0
    else: corr = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
    return mse, r2, corr

def eval_on(dataloader, model, device):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(yb.cpu().numpy())
    return np.concatenate(all_trues), np.concatenate(all_preds)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_combinations = []
    for target_channel in ALL_EMG_CHANNELS:
        input_options = [ch for ch in ALL_EMG_CHANNELS if ch != target_channel]
        for input_combo in itertools.combinations(input_options, NUM_INPUT_CHANNELS):
            all_combinations.append({'inputs': list(input_combo), 'target': target_channel})
    
    print(f"Found {len(all_combinations)} unique combinations to test.")
    
    results = []
    for i, combo in enumerate(all_combinations):
        print("-" * 60)
        print(f"Running Combination {i+1}/{len(all_combinations)}: Inputs={combo['inputs']}, Target={combo['target']}")
        print("-" * 60)

        X_data, Y_data = prepare_all_emg(sequence_len=SEQUENCE_LEN, input_channels=combo['inputs'], target_channel=combo['target'])
        
        # CHANGED: Lowered the threshold for valid data
        if X_data.shape[0] < 10:
            print("  [WARNING] Not enough data sequences created for this combination. Skipping.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
        train_ds = EMGDataset(X_train, y_train)
        test_ds = EMGDataset(X_test, y_test)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE * 2)

        model = EMGSeq2SeqNet(input_channels=NUM_INPUT_CHANNELS, time_len=WINDOW_SAMPLES, output_len=OUTPUT_LEN, lstm_hidden=LSTM_HIDDEN, dropout_rate=DROPOUT_RATE).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} complete.")

        y_test_true, y_test_pred = eval_on(test_dl, model, device)
        mse, r2, corr = compute_metrics(y_test_true, y_test_pred)
        print(f"\n  Results for {combo['inputs']} -> {combo['target']}:\n  Test MSE: {mse:.4f}, R²: {r2:.4f}, Corr: {corr:.4f}\n")

        results.append({'input_channels': f"{combo['inputs'][0]}, {combo['inputs'][1]}", 'target_channel': combo['target'], 'r_squared': r2, 'mse': mse, 'correlation': corr})

    print("\n" + "=" * 80 + "\n" + " " * 25 + "FINAL EXPERIMENT RESULTS" + "\n" + "=" * 80)
    if not results:
        print("No results were generated. Please check your data and configuration.")
    else:
        summary_df = pd.DataFrame(results).sort_values(by='r_squared', ascending=False).reset_index(drop=True)
        print(summary_df.to_string())
    print("\n" + "=" * 80)