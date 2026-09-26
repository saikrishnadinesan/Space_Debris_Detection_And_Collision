"""
Phase 14: Collision Detection Test (real data)
=================================================
Uses the trained LSTM to predict future paths for two REAL debris
objects, then feeds those predicted paths into the collision risk
math from collision_risk.py.
"""

import sys
sys.path.append("src/models")

import numpy as np
import pandas as pd
import joblib
from train_prediction import predict_trajectory
from collision_risk import (
    DebrisObject, compute_miss_distance,
    monte_carlo_collision_probability, classify_risk
)

TRAJ_PATH = "data/processed/trajectories.csv"
MODEL_PATH = "models/prediction/best_lstm.pt"
SCALER_PATH = "models/prediction/scaler.pkl"
FEATURE_COLS = ["x", "y", "z", "vx", "vy", "vz"]


def get_predicted_path(df, norad_id, scaler):
    obj_data = df[df["norad_id"] == norad_id].sort_values("timestamp")
    input_seq = obj_data.iloc[:20][FEATURE_COLS].values
    last_velocity = obj_data.iloc[19][["vx", "vy", "vz"]].values

    pred_normalized = predict_trajectory(MODEL_PATH, input_seq, SCALER_PATH)

    dummy = np.zeros((10, 6))
    dummy[:, :3] = pred_normalized
    pred_km = scaler.inverse_transform(dummy)[:, :3]

    name = obj_data.iloc[0]["name"]
    return pred_km, last_velocity, name


if __name__ == "__main__":
    df = pd.read_csv(TRAJ_PATH)
    scaler = joblib.load(SCALER_PATH)

    ID_A, ID_B = 51, 53

    path_a, vel_a, name_a = get_predicted_path(df, ID_A, scaler)
    path_b, vel_b, name_b = get_predicted_path(df, ID_B, scaler)

    obj_a = DebrisObject(id=ID_A, name=name_a, position=path_a[0],
                          velocity=vel_a, size_m=1.0, predicted_path=path_a)
    obj_b = DebrisObject(id=ID_B, name=name_b, position=path_b[0],
                          velocity=vel_b, size_m=1.0, predicted_path=path_b)

    min_dist, tca = compute_miss_distance(obj_a.predicted_path, obj_b.predicted_path)
    print(f"Object A: {name_a} (NORAD {ID_A})")
    print(f"Object B: {name_b} (NORAD {ID_B})")
    print(f"\nMinimum predicted distance: {min_dist:.1f} km (at future step {tca})")

    prob = monte_carlo_collision_probability(
        obj_a.predicted_path[tca], obj_b.predicted_path[tca],
        obj_a.velocity, obj_b.velocity,
        obj_a.size_m, obj_b.size_m,
        samples=500
    )
    risk = classify_risk(prob, min_dist)

    print(f"Monte Carlo collision probability: {prob:.6f}")
    print(f"Risk level: {risk}")
    print(f"\nNote: real debris objects are typically thousands of km apart,")
    print(f"so LOW risk here is an EXPECTED, correct result - not a failure.")
