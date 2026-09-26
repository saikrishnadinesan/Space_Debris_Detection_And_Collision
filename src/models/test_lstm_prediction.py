"""
Phase 13: Trajectory Prediction Test
=======================================
Takes a real object's actual first 20 known positions, predicts the
next 10 using the trained LSTM, and compares against the REAL next 10
positions (which we already have, since this is historical data).
"""

import sys
sys.path.append("src/models")

import numpy as np
import pandas as pd
from train_prediction import predict_trajectory

TRAJ_PATH = "data/processed/trajectories.csv"
MODEL_PATH = "models/prediction/best_lstm.pt"
SCALER_PATH = "models/prediction/scaler.pkl"
TARGET_NORAD_ID = 51
FEATURE_COLS = ["x", "y", "z", "vx", "vy", "vz"]

if __name__ == "__main__":
    df = pd.read_csv(TRAJ_PATH)
    obj_data = df[df["norad_id"] == TARGET_NORAD_ID].sort_values("timestamp")

    input_seq = obj_data.iloc[:20][FEATURE_COLS].values
    actual_future = obj_data.iloc[20:30][["x", "y", "z"]].values

    predicted_future = predict_trajectory(MODEL_PATH, input_seq, SCALER_PATH)

    # predicted_future comes back normalized (scaled) - un-scale it back to real km
    import joblib
    scaler = joblib.load(SCALER_PATH)
    dummy = np.zeros((10, 6))
    dummy[:, :3] = predicted_future
    predicted_future_km = scaler.inverse_transform(dummy)[:, :3]

    print(f"Object: NORAD ID {TARGET_NORAD_ID}\n")
    print(f"{'Step':<6}{'Actual (x,y,z) km':<45}{'Predicted (x,y,z) km':<45}{'Error (km)'}")
    print("-" * 120)
    for i in range(10):
        actual = actual_future[i]
        pred = predicted_future_km[i]
        error = np.linalg.norm(actual - pred)
        print(f"{i+1:<6}{str(np.round(actual,1)):<45}{str(np.round(pred,1)):<45}{error:.1f}")

    mean_error = np.mean([np.linalg.norm(actual_future[i] - predicted_future_km[i]) for i in range(10)])
    print(f"\nMean position error across 10-step prediction: {mean_error:.1f} km")
