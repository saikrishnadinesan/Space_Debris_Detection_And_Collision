"""
Phase 15: Monte Carlo Stress Test (forced close approach)
=============================================================
Takes one real predicted trajectory and creates an artificial
"twin" object offset by a small, deliberate distance, to verify
the MEDIUM/HIGH/CRITICAL risk tiers actually trigger correctly.
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
    return pred_km, last_velocity


if __name__ == "__main__":
    df = pd.read_csv(TRAJ_PATH)
    scaler = joblib.load(SCALER_PATH)

    real_path, real_vel = get_predicted_path(df, 51, scaler)

    print("=" * 70)
    print("Testing THREE scenarios: FAR / CLOSE / VERY CLOSE offset twin")
    print("=" * 70)

    offsets_km = {
        "FAR (50 km offset)": 50.0,
        "CLOSE (0.3 km offset)": 0.3,
        "VERY CLOSE (0.02 km offset)": 0.02,
    }

    for label, offset in offsets_km.items():
        twin_path = real_path + np.array([offset, 0, 0])

        obj_a = DebrisObject(id=51, name="OBJ-51 (real)", position=real_path[0],
                              velocity=real_vel, size_m=2.0, predicted_path=real_path)
        obj_b = DebrisObject(id=9999, name="TWIN (artificial)", position=twin_path[0],
                              velocity=real_vel, size_m=2.0, predicted_path=twin_path)

        min_dist, tca = compute_miss_distance(obj_a.predicted_path, obj_b.predicted_path)
        prob = monte_carlo_collision_probability(
            obj_a.predicted_path[tca], obj_b.predicted_path[tca],
            obj_a.velocity, obj_b.velocity,
            obj_a.size_m, obj_b.size_m,
            samples=500
        )
        risk = classify_risk(prob, min_dist)

        print(f"\n{label}")
        print(f"   Min distance: {min_dist:.4f} km")
        print(f"   Collision probability: {prob:.4f}")
        print(f"   Risk level: {risk}")
