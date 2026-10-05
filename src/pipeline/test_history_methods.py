"""
Phase 16 diagnostic: compare pipeline's linear-history approximation
against real stored history, for an object present in both datasets.
"""

import sys
sys.path.insert(0, "src")
sys.path.append("src/models")

import numpy as np
import pandas as pd
import joblib
from train_prediction import predict_trajectory

TARGET_NORAD_ID = 655
MODEL_PATH = "models/prediction/best_lstm.pt"
SCALER_PATH = "models/prediction/scaler.pkl"

# --- Method A: pipeline's fake linear-backward history ---
df_pos = pd.read_csv("outputs/real_positions.csv")
row = df_pos[df_pos["norad_id"] == TARGET_NORAD_ID].iloc[0]
pos = np.array([row["x"], row["y"], row["z"]])
vel = np.array([row["vx"], row["vy"], row["vz"]])
dt = 900  # seconds per step (15 min) - must match trajectories.csv's actual spacing
fake_history = np.array([
    np.concatenate([pos - vel * dt * (20 - t), vel]) for t in range(20)
], dtype=np.float32)

pred_fake = predict_trajectory(MODEL_PATH, fake_history, SCALER_PATH)

# --- Method B: real stored SGP4 history ---
df_traj = pd.read_csv("data/processed/trajectories.csv")
real_obj = df_traj[df_traj["norad_id"] == TARGET_NORAD_ID].sort_values("timestamp")
real_history = real_obj.iloc[:20][["x","y","z","vx","vy","vz"]].values

pred_real = predict_trajectory(MODEL_PATH, real_history, SCALER_PATH)

# --- Unscale both to real km ---
scaler = joblib.load(SCALER_PATH)

def unscale(pred):
    dummy = np.zeros((10, 6))
    dummy[:, :3] = pred
    return scaler.inverse_transform(dummy)[:, :3]

pred_fake_km = unscale(pred_fake)
pred_real_km = unscale(pred_real)

print(f"Object: NORAD {TARGET_NORAD_ID}\n")
print(f"{'Step':<6}{'Pred (fake history) km':<45}{'Pred (real history) km':<45}{'Difference (km)'}")
print("-" * 130)
for i in range(10):
    diff = np.linalg.norm(pred_fake_km[i] - pred_real_km[i])
    print(f"{i+1:<6}{str(np.round(pred_fake_km[i],1)):<45}{str(np.round(pred_real_km[i],1)):<45}{diff:.1f}")

mean_diff = np.mean([np.linalg.norm(pred_fake_km[i] - pred_real_km[i]) for i in range(10)])
print(f"\nMean difference between the two methods: {mean_diff:.1f} km")
