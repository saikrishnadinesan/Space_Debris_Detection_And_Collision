"""
Phase 6: Trajectory Prediction - LSTM Model
=============================================
Predicts future debris positions using an LSTM neural network
trained on historical trajectory data from TLE propagation.

Input:  Sequence of past positions & velocities [x, y, z, vx, vy, vz]
Output: Predicted future positions [x, y, z]
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

MODEL_SAVE_DIR = "models/prediction"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# --- Dataset ---

class DebrisTrajectoryDataset(Dataset):
    """
    Sliding window dataset for trajectory prediction.
    Builds sequences SEPARATELY per object (grouped by norad_id) so that
    no sequence ever crosses from one object's trajectory into another's.
    """
    def __init__(self, df: pd.DataFrame, feature_cols: list, scaler,
                 seq_len: int = 20, pred_len: int = 10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.X, self.y = self._create_sequences(df, feature_cols, scaler)

    def _create_sequences(self, df, feature_cols, scaler):
        X, y = [], []
        for norad_id, group in df.groupby("norad_id"):
            if "timestamp" in group.columns:
                group = group.sort_values("timestamp")
            data = scaler.transform(group[feature_cols].values)
            for i in range(len(data) - self.seq_len - self.pred_len):
                X.append(data[i:i + self.seq_len])
                y.append(data[i + self.seq_len:i + self.seq_len + self.pred_len, :3])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# --- Model Architecture ---

class DebrisLSTM(nn.Module):
    """
    LSTM model for debris trajectory prediction.

    Architecture:
    Input -> LSTM layers -> Dropout -> Fully Connected -> Output
    """
    def __init__(self, input_size=6, hidden_size=128, num_layers=2,
                 pred_len=10, dropout=0.2):
        super(DebrisLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, pred_len * 3)  # predict x,y,z for each future step

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])   # take last timestep
        out = self.fc(out)
        out = out.view(-1, self.pred_len, 3)      # reshape to (batch, pred_len, 3)
        return out


class DebrisTransformer(nn.Module):
    """
    Transformer-based trajectory predictor (alternative to LSTM).
    Generally better for longer sequences. Not trained in this phase -
    scaffolded here for future work, per the project's future-scope plan.
    """
    def __init__(self, input_size=6, d_model=128, nhead=8,
                 num_layers=3, pred_len=10, dropout=0.1):
        super(DebrisTransformer, self).__init__()

        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, pred_len * 3)
        self.pred_len = pred_len

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])
        return out.view(-1, self.pred_len, 3)


# --- Training ---

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """Train trajectory prediction model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    print(f"Training on {device}")
    print("-" * 50)

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_lstm.pt"))

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    return train_losses, val_losses


def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Trajectory Prediction - Training Curves")
    plt.legend()
    plt.grid(True)
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/lstm_training_curves.png", dpi=150, bbox_inches="tight")
    print("Training curves saved to outputs/lstm_training_curves.png")


# --- Inference ---

def predict_trajectory(model_path: str, input_sequence: np.ndarray,
                       scaler_path: str = None) -> np.ndarray:
    """
    Predict future trajectory from input sequence.

    Args:
        model_path: Path to saved model weights
        input_sequence: Array of shape (seq_len, 6) with [x,y,z,vx,vy,vz]
        scaler_path: Optional path to StandardScaler for denormalization

    Returns:
        Predicted positions of shape (pred_len, 3)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DebrisLSTM()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        input_sequence = scaler.transform(input_sequence)

    x = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x).squeeze(0).cpu().numpy()

    return pred


# --- Main ---

def run_training_pipeline():
    """Full training pipeline from data to saved model."""
    print("=" * 60)
    print("TRAJECTORY PREDICTION - LSTM TRAINING")
    print("=" * 60)

    # Load trajectory data
    traj_path = "data/processed/trajectories.csv"
    if not os.path.exists(traj_path):
        print("No trajectory data found. Run preprocess.py first (Phase 6).")
        return
    else:
        df = pd.read_csv(traj_path)

    feature_cols = ["x", "y", "z", "vx", "vy", "vz"]

    # Fit scaler on all data (this part is fine to do globally - it's just
    # computing mean/std per feature column, not building sequences)
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].values)
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "scaler.pkl"))

    # Create datasets - built PER OBJECT to avoid cross-object contamination
    SEQ_LEN, PRED_LEN = 20, 10
    dataset = DebrisTrajectoryDataset(df, feature_cols, scaler, SEQ_LEN, PRED_LEN)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    print(f"Dataset: {len(dataset)} sequences")
    print(f"   Train: {train_size} | Val: {val_size} | Test: {test_size}")

    # Train LSTM
    model = DebrisLSTM(input_size=6, hidden_size=128, num_layers=2,
                       pred_len=PRED_LEN, dropout=0.2)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50)
    plot_training_curves(train_losses, val_losses)


if __name__ == "__main__":
    run_training_pipeline()