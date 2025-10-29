# plot_best_model.py

import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from data_preparation import prepare_all_emg, WINDOW_SAMPLES
from my_model import EMGSeq2SeqNet

# --- STEP 1: DEFINE THE COMBINATION YOU WANT TO PLOT ---
BEST_INPUT_CHANNELS = ['emg1', 'emg7']  # <-- For your #1 result
BEST_TARGET_CHANNEL = 'emg8'             # <-- For your #1 result (used internally only)
# ----------------------------------------------------------------

# --- STEP 2: DEFINE WHERE TO SAVE THE PLOTS ---
MAIN_PLOT_DIRECTORY = 'C:/Users/rajes/Desktop/IIITB/emg-emg/PLOT_RESULTS/'
# ----------------------------------------------------------------

# --- Hyperparameters ---
SEQUENCE_LEN = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 100
LSTM_HIDDEN = 128
DROPOUT_RATE = 0.5
NUM_INPUT_CHANNELS = len(BEST_INPUT_CHANNELS)
OUTPUT_LEN = SEQUENCE_LEN * WINDOW_SAMPLES


class EMGDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def compute_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    if np.var(y_true) < 1e-6:
        r2 = 0
    else:
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    y_true_flat, y_pred_flat = y_true.flatten(), y_pred.flatten()
    if np.all(y_true_flat == y_true_flat[0]) or np.all(y_pred_flat == y_pred_flat[0]):
        corr = 0
    else:
        corr = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
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


def plot_and_save_predictions(y_true, y_pred, save_dir, max_plots=20):
    """Saves plots of actual vs. predicted EMG signals."""
    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams.update({
    'font.size': 14,
    'font.weight': 'bold',          # <-- makes all base text bold
    'axes.labelsize': 18,
    'axes.labelweight': 'bold',     # <-- makes X/Y axis labels bold
    'axes.titlesize': 20,
    'axes.titleweight': 'bold',     # <-- makes title bold
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'legend.title_fontsize': 16
})

    for i in range(min(len(y_true), max_plots)):
        plt.figure(figsize=(15, 5))
        plt.plot(y_true[i], label='Actual EMG', color='blue', linewidth=2)
        plt.plot(y_pred[i], label='Predicted EMG', color='red', linestyle='--', linewidth=2)
        plt.title(f'Test Sample {i+1}', pad=15)
        plt.xlabel('Time Points')
        plt.ylabel('Normalized Amplitude')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_{i+1}.png'), dpi=300)
        plt.close()

    print(f"\nSaved {min(len(y_true), max_plots)} plots to the '{save_dir}' folder.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training and plotting for combination: Inputs={BEST_INPUT_CHANNELS}, Target={BEST_TARGET_CHANNEL}")

    X_data, Y_data = prepare_all_emg(
        sequence_len=SEQUENCE_LEN,
        input_channels=BEST_INPUT_CHANNELS,
        target_channel=BEST_TARGET_CHANNEL
    )

    if X_data.shape[0] < 10:
        print("Error: Not enough data to train. Exiting.")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

    train_ds = EMGDataset(X_train, y_train)
    test_ds = EMGDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE * 2)

    model = EMGSeq2SeqNet(
        input_channels=NUM_INPUT_CHANNELS,
        time_len=WINDOW_SAMPLES,
        output_len=OUTPUT_LEN,
        lstm_hidden=LSTM_HIDDEN,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
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
            print(f"  Epoch {epoch + 1}/{EPOCHS} complete.")

    print("Training finished.")

    y_test_true, y_test_pred = eval_on(test_dl, model, device)

    # --- Create final save directory ---
    combination_subfolder = f"plots_{BEST_INPUT_CHANNELS[0]}_{BEST_INPUT_CHANNELS[1]}_to_{BEST_TARGET_CHANNEL}"
    plot_save_dir = os.path.join(MAIN_PLOT_DIRECTORY, combination_subfolder)

    plot_and_save_predictions(y_test_true, y_test_pred, save_dir=plot_save_dir, max_plots=20)

    mse, r2, corr = compute_metrics(y_test_true, y_test_pred)
    print("\n--- Final Test Metrics for this Combination ---")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Correlation: {corr:.4f}")
