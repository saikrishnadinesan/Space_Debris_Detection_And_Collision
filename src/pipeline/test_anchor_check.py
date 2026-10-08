import sys
sys.path.insert(0, "src")
from pipeline.real_pipeline import load_catalog, propagate_positions, predict_trajectories
import numpy as np

df_catalog = load_catalog()
df_pos = propagate_positions(df_catalog)
predictions = predict_trajectories(df_pos, steps=10)

for nid in [2775, 7060]:
    row = df_pos[df_pos["norad_id"] == nid].iloc[0]
    real_now = np.array([row["x"], row["y"], row["z"]])
    pred_step0 = np.array(predictions[nid][0])
    drift = np.linalg.norm(real_now - pred_step0)
    print(f"\nNORAD {nid}")
    print(f"  Real current position (SGP4, trusted): {np.round(real_now,1)}")
    print(f"  LSTM predicted step-0 position:        {np.round(pred_step0,1)}")
    print(f"  Drift between the two:                 {drift:.1f} km")
