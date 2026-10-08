"""
Phase 16 diagnostic (corrected): reproduce the EXACT same pipeline run
in one single execution, so "now" is consistent, then inspect the full
10-step distance curve for the flagged CRITICAL pairs.
"""

import sys
sys.path.insert(0, "src")
from pipeline.real_pipeline import load_catalog, propagate_positions, predict_trajectories
import numpy as np

df_catalog = load_catalog(300)
df_pos = propagate_positions(df_catalog)
predictions = predict_trajectories(df_pos, steps=10)

pairs_to_check = [(2775, 7060), (30867, 30124), (30737, 61839)]

for id_a, id_b in pairs_to_check:
    if id_a not in predictions or id_b not in predictions:
        print(f"\n{id_a} vs {id_b}: one or both objects not in this run's sample - skipping")
        continue
    path_a = predictions[id_a]
    path_b = predictions[id_b]
    print(f"\n{id_a} vs {id_b}")
    print("Step | Distance (km)")
    for i in range(10):
        dist = np.linalg.norm(np.array(path_a[i]) - np.array(path_b[i]))
        print(f"{i:<5}{dist:.3f}")
