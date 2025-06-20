import torch
import torch.optim as optim
import pandas as pd
import joblib
import os
from data_preparation import prepare_all
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from my_model import RawSeq2SeqNet
import torch.nn as nn
# --- Import your classes and functions from your main code ---
from tune_params_onlycsp import (
    INPUT_DIM, SEQUENCE_LEN, OUTPUT_LEN, EEGEMGDataset,
    Seq2SeqModel, CNNTransformerModel, EEGNetRegressor, SoftDTWLoss,
    compute_metrics, plot_and_save_predictions
)
INPUT_DIM=19

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=0.2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.softdtw = SoftDTWLoss(gamma=gamma)
        self.alpha = alpha

    def forward(self, pred, target):
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * self.softdtw(pred, target)

def get_model(model_name, device, input_dim, seq_len, output_len, hidden_dim=128, dropout_rate=0.5):
    if model_name == "seq2seq":
        return Seq2SeqModel(input_dim, hidden_dim, output_len).to(device)
    elif model_name == "cnn_transformer":
        return CNNTransformerModel(input_dim, seq_len, output_len).to(device)
    elif model_name == "eegnet":
        return EEGNetRegressor(input_channels=24, seq_len=seq_len, output_len=output_len, dropout_rate=dropout_rate).to(device)
    else:
        raise ValueError("Unknown model name")

def get_loss_function(loss_name, gamma=0.1):
    if loss_name == "softdtw":
        return SoftDTWLoss(gamma=gamma)
    elif loss_name == "mse":
        import torch.nn as nn
        return nn.MSELoss()
    else:
        raise ValueError("Unknown loss function")

def prepare_inputs_for_model(X, model_name, n_channels=24, n_bands=4):
    if model_name == "eegnet":
        X_reshaped = X.reshape(X.shape[0], X.shape[1], n_channels, n_bands)
        X_reduced = X_reshaped.mean(axis=-1)
        return X_reduced
    else:
        return X

def eval_on(loader, model, device):
    all_preds, all_trues = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().detach().numpy()
            all_preds.append(pred)
            all_trues.append(yb.numpy())
    import numpy as np
    return np.concatenate(all_trues), np.concatenate(all_preds)

# ------------ FILL THIS WITH YOUR OPTUNA RESULTS -------------
best_params = {
    "model_name": "cnn_transformer",       # e.g. "seq2seq", "cnn_transformer", "eegnet"
    "lr": 1e-3,                 # example value, replace with Optuna's best
    "batch_size": 8,             # example value
    "hidden_dim": 128,            # example value
    "dropout_rate": 0.3,          # example value
    "gamma": 0.2,                 # example value
    "loss_name": "softdtw"        # or "mse"
}
EPOCHS = 1000
METRIC_PATH = "training_metrics.csv"

if __name__ == "__main__":
    eeg, emg = prepare_all()
    X_temp, X_test, y_temp, y_test = train_test_split(eeg, emg, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    print("X_train shape:", X_train.shape)
    global_mean = emg.mean()
    global_std = emg.std()

    # No standardization needed unless doing per-channel scaling
    mean = y_train.mean()
    std = y_train.std()
    y_train = (y_train - global_mean) / global_std
    y_val   = (y_val - global_mean) / global_std
    y_test  = (y_test - global_mean) / global_std
    
    X_train_scaled = X_train
    y_train_scaled = y_train
    X_val_scaled = X_val
    y_val_scaled = y_val
    X_test_scaled = X_test
    y_test_scaled = y_test

    model_name = best_params["model_name"]
    batch_size = best_params["batch_size"]
    hidden_dim = best_params["hidden_dim"]
    dropout_rate = best_params["dropout_rate"]
    gamma = best_params["gamma"]
    loss_name = best_params["loss_name"]
    lr = best_params["lr"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_mod = X_train_scaled
    X_val_mod   = X_val_scaled
    X_test_mod  = X_test_scaled

    train_ds = EEGEMGDataset(X_train_mod, y_train_scaled)
    val_ds   = EEGEMGDataset(X_val_mod,   y_val_scaled)
    test_ds  = EEGEMGDataset(X_test_mod,  y_test_scaled)

    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size)
    test_dl  = DataLoader(test_ds, batch_size=batch_size)

    # model = get_model(model_name, device, INPUT_DIM, SEQUENCE_LEN, OUTPUT_LEN, hidden_dim,dropout_rate)
    model = RawSeq2SeqNet(
        input_channels=INPUT_DIM,
        seq_len=SEQUENCE_LEN,
        time_len=500,
        output_len=OUTPUT_LEN
    ).to(device)
    criterion = CombinedLoss(alpha = 0.9, gamma=gamma)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    metrics_history = []

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 25 == 0 or epoch == EPOCHS - 1:
            y_val_true, y_val_pred = eval_on(val_dl, model, device)
            mse, r2, corr = compute_metrics(y_val_true, y_val_pred)
            print(f"[Epoch {epoch}] Val Loss: {mse:.4f}, R²: {r2:.4f}, Corr: {corr:.4f}")
            metrics_history.append({
                "epoch": epoch,
                "val_mse": mse,
                "val_r2": r2,
                "val_corr": corr
            })

    y_test_true, y_test_pred = eval_on(test_dl, model, device)
    y_test_pred = y_test_pred * global_std + global_mean
    y_test_true = y_test_true * global_std + global_mean
    plot_and_save_predictions(y_test_true, y_test_pred, save_dir="plots/test_predictions")
    test_mse, test_r2, test_corr = compute_metrics(y_test_true, y_test_pred)
    print(f"\nTest Loss: {test_mse:.4f}, R²: {test_r2:.4f}, Corr: {test_corr:.4f}")

    metrics_history.append({
        "epoch": "final_test",
        "val_mse": test_mse,
        "val_r2": test_r2,
        "val_corr": test_corr
    })

    torch.save(model.state_dict(), f"{model_name}_model.pt")
    pd.DataFrame(metrics_history).to_csv(METRIC_PATH, index=False)