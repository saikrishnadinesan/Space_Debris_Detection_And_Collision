"""
test_risk_injection.py

Validates the collision risk classification pipeline (Monte Carlo probability
+ distance-based classification) by feeding it a synthetic, deliberately
close pair of trajectories. This proves the risk logic is correct,
independent of whether any real TLE pair in a given pipeline run happens to
come close enough on its own to trigger a CRITICAL/HIGH flag.
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np
from models.collision_risk import (
    compute_miss_distance,
    monte_carlo_collision_probability,
    classify_risk,
)

LSTM_PREDICTION_ERROR_KM = 184.0
N_STEPS = 10

np.random.seed(42)

# Build two synthetic predicted trajectories (km), wandering near a realistic
# LEO-ish altitude, diverging at most steps except at step 5 where they are
# deliberately placed ~50 meters apart (well under the 0.1km CRITICAL
# threshold).
base = np.cumsum(np.random.normal(0, 50, size=(N_STEPS, 3)), axis=0) + np.array([7000.0, 0.0, 0.0])
traj_a = base.copy()
traj_b = base.copy()

offset = np.random.normal(0, 300, size=(N_STEPS, 3))
offset[5] = np.array([0.03, 0.02, -0.01])  # ~0.037 km apart at step 5
traj_b = traj_b + offset

min_dist, tca_step = compute_miss_distance(traj_a, traj_b)
print(f"Min distance: {min_dist:.4f} km at step {tca_step}")

vel_a = np.array([7.5, 0.1, 0.0])
vel_b = np.array([-7.4, 0.2, 0.05])
size_a = 0.5  # m
size_b = 0.5  # m

prob = monte_carlo_collision_probability(
    traj_a[tca_step], traj_b[tca_step], vel_a, vel_b,
    size_a, size_b,
    uncertainty_km=LSTM_PREDICTION_ERROR_KM, samples=1000,
)
print(f"Collision probability: {prob:.6f}")

risk = classify_risk(prob, min_dist)
print(f"Risk level: {risk}")

assert "CRITICAL" in risk or "HIGH" in risk, f"Expected CRITICAL/HIGH, got {risk}"
print("\nRisk classification logic verified: correctly flags a close-approach pair.")