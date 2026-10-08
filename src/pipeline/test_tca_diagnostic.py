import sys
sys.path.insert(0, "src")
sys.path.append("src/models")

import numpy as np
import pandas as pd
import joblib
from train_prediction import predict_trajectory
from sgp4.api import Satrec, jday
from datetime import datetime, timezone

MODEL_PATH = "models/prediction/best_lstm.pt"
SCALER_PATH = "models/prediction/scaler.pkl"

def get_real_history_and_predict(norad_id, catalog):
    row = catalog[catalog["norad_id"] == norad_id].iloc[0]
    sat = Satrec.twoline2rv(row["line1"], row["line2"])
    now = datetime.now(timezone.utc)
    dt_seconds = 900
    history_rows = []
    for t in range(20, 0, -1):
        past_time = now - pd.Timedelta(seconds=dt_seconds * t)
        jd, fr = jday(past_time.year, past_time.month, past_time.day,
                      past_time.hour, past_time.minute,
                      past_time.second + past_time.microsecond / 1e6)
        err, p, v = sat.sgp4(jd, fr)
        history_rows.append([p[0], p[1], p[2], v[0], v[1], v[2]])
    history = np.array(history_rows, dtype=np.float32)
    pred = predict_trajectory(MODEL_PATH, history, SCALER_PATH)
    scaler = joblib.load(SCALER_PATH)
    dummy = np.zeros((10, 6))
    dummy[:, :3] = pred
    return scaler.inverse_transform(dummy)[:, :3]

catalog = pd.read_csv("data/processed/debris_catalog.csv")
path_a = get_real_history_and_predict(2775, catalog)
path_b = get_real_history_and_predict(7060, catalog)

print("Step | Object A position (km)        | Object B position (km)        | Distance (km)")
print("-" * 100)
for i in range(10):
    dist = np.linalg.norm(path_a[i] - path_b[i])
    print(f"{i:<5}{str(np.round(path_a[i],1)):<32}{str(np.round(path_b[i],1)):<32}{dist:.2f}")
