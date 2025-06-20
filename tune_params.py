import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import optuna

from data_preparation import prepare_all

# --- Constants ---
# --- Dynamic Constants ---
SEGMENT_DURATION = 2             # seconds
FS_EEG = 250
FS_EMG = 100
EEG_WINDOW = FS_EEG * SEGMENT_DURATION  # 500 samples per 2s window

EEG_CHANNELS = 19                # Based on CSP channel selection
BANDS = [(0.5, 7), (7, 13), (13, 20)]  # Bandpower bands
N_BANDS = len(BANDS) + 1         # 3 bandpowers + 1 phase angle per channel

CSP_COMPONENTS = 4               # Number of CSP components appended
INPUT_DIM = EEG_CHANNELS * N_BANDS + CSP_COMPONENTS  # 19*(3+1)+4 = 80

SEQUENCE_LEN = 5                 # 5 x 2s windows = 10s
OUTPUT_LEN = FS_EMG * 10         # Predict 10s of EMG = 1000 points
EPOCHS = 100                     # For tuning, increase for final
METRIC_PATH = "training_metrics.csv"

# --- Dataset ---
class EEGEMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Models ---
class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_len):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_len)
        )
    def forward(self, x):
        enc_output, _ = self.encoder(x)
        context = enc_output[:, -1, :]
        return self.decoder(context)

class CNNTransformerModel(nn.Module):
    def __init__(self, input_dim, seq_len, output_len, cnn_channels=64, n_heads=4, n_layers=2, trans_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU()
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=cnn_channels, nhead=n_heads, dim_feedforward=trans_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.Sequential(
            nn.Linear(seq_len * cnn_channels, trans_dim),
            nn.ReLU(),
            nn.Linear(trans_dim, output_len)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.flatten(1)
        x = self.decoder(x)
        return x

class EEGNetRegressor(nn.Module):
    def __init__(self, input_channels=24, seq_len=5, output_len=1000, dropout_rate=0.5):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(8)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(8, 16, (input_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout_rate)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(16, 16, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout_rate)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._get_flattened_size(input_channels, seq_len), output_len)
        )
    def _get_flattened_size(self, input_channels, seq_len):
        dummy = torch.zeros(1, 1, input_channels, seq_len)
        x = self.firstconv(dummy)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        return x.numel()
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.regressor(x)
        return x

# --- Loss Functions ---
class SoftDTWLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, y_pred, y_true):
        batch_size = y_pred.size(0)
        total_loss = 0.0
        for i in range(batch_size):
            D = torch.cdist(y_pred[i].unsqueeze(0), y_true[i].unsqueeze(0), p=2)
            R = torch.zeros_like(D)
            R[0, 0] = D[0, 0]
            for j in range(1, D.size(1)):
                R[0, j] = D[0, j] + R[0, j-1]
            for i2 in range(1, D.size(0)):
                R[i2, 0] = D[i2, 0] + R[i2-1, 0]
            for i2 in range(1, D.size(0)):
                for j in range(1, D.size(1)):
                    rmin = torch.min(torch.stack([R[i2-1, j], R[i2, j-1], R[i2-1, j-1]]))
                    R[i2, j] = D[i2, j] + rmin
            total_loss += R[-1, -1]
        return total_loss / batch_size

def get_loss_function(loss_name, gamma=0.1):
    if loss_name == "softdtw":
        return SoftDTWLoss(gamma=gamma)
    elif loss_name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError("Unknown loss function")

# --- Model selection ---
def get_model(model_name, device, input_dim, seq_len, output_len, hidden_dim=128, dropout_rate=0.5):
    if model_name == "seq2seq":
        return Seq2SeqModel(input_dim, hidden_dim, output_len).to(device)
    elif model_name == "cnn_transformer":
        return CNNTransformerModel(input_dim, seq_len, output_len).to(device)
    elif model_name == "eegnet":
        return EEGNetRegressor(input_channels=24, seq_len=seq_len, output_len=output_len, dropout_rate=dropout_rate).to(device)
    else:
        raise ValueError("Unknown model name")

# --- Data preprocessing for model ---
def prepare_inputs_for_model(X, model_name, n_channels=24, n_bands=4):
    if model_name == "eegnet":
        X_reshaped = X.reshape(X.shape[0], X.shape[1], n_channels, n_bands)
        X_reduced = X_reshaped.mean(axis=-1)
        return X_reduced
    else:
        return X

# --- Metrics ---
def compute_metrics(true, pred):
    mse = mean_squared_error(true, pred)
    r2 = r2_score(true, pred)
    corr = pearsonr(true.flatten(), pred.flatten())[0]
    return mse, r2, corr

def plot_and_save_predictions(y_true, y_pred, save_dir="plots", num_samples=10):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(num_samples, len(y_true))):
        true_signal = y_true[i]
        pred_signal = y_pred[i]
        plt.figure(figsize=(12, 4))
        plt.plot(true_signal, label="True EMG")
        plt.plot(pred_signal, label="Predicted EMG", linestyle='--')
        plt.title(f"Test Sample #{i}")
        plt.xlabel("Time Steps")
        plt.ylabel("EMG Envelope")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{i}.png"))
        plt.close()

def eval_on(loader, model, device):
    all_preds, all_trues = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().detach().numpy()
            all_preds.append(pred)
            all_trues.append(yb.numpy())
    return np.concatenate(all_trues), np.concatenate(all_preds)

# --- Optuna Objective ---
def objective(trial):
    # --- Hyperparameters to tune ---
    model_name = trial.suggest_categorical("model_name", ["seq2seq", "cnn_transformer", "eegnet"])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7)
    gamma = trial.suggest_float("gamma", 0.05, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    loss_name = trial.suggest_categorical("loss_name", ["softdtw", "mse"])

    # --- Data Preparation ---
    eeg, emg = prepare_all()
    X_temp, X_test, y_temp, y_test = train_test_split(eeg, emg, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    eeg_scaler = StandardScaler()
    emg_scaler = StandardScaler()
    X_train_scaled = eeg_scaler.fit_transform(X_train.reshape(-1, INPUT_DIM)).reshape(X_train.shape)
    y_train_scaled = emg_scaler.fit_transform(y_train)
    X_val_scaled = eeg_scaler.transform(X_val.reshape(-1, INPUT_DIM)).reshape(X_val.shape)
    y_val_scaled = emg_scaler.transform(y_val)

    X_train_mod = prepare_inputs_for_model(X_train_scaled, model_name)
    X_val_mod   = prepare_inputs_for_model(X_val_scaled, model_name)

    train_ds = EEGEMGDataset(X_train_mod, y_train_scaled)
    val_ds   = EEGEMGDataset(X_val_mod,   y_val_scaled)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, device, INPUT_DIM, SEQUENCE_LEN, OUTPUT_LEN, hidden_dim, dropout_rate)
    criterion = get_loss_function(loss_name, gamma=gamma)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Training Loop (short for tuning) ---
    for epoch in range(10):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    y_val_true, y_val_pred = eval_on(val_dl, model, device)
    mse, r2, corr = compute_metrics(y_val_true, y_val_pred)
    return -r2  # maximize r2

# --- Main ---
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=60)  # Increase n_trials for more thorough search

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {-trial.value:.4f} (Best Correlation)")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # You can now use trial.params to run a final training with the best hyperparameters