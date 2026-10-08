"""
Phase 16: Full Pipeline Integration
=====================================
Connects all modules together:
  SGP4 propagation -> LSTM trajectory prediction -> Monte Carlo collision risk -> dashboard data

This is the "real" pipeline (uses real catalog + real trained models),
as opposed to the individual test_*.py scripts which test one module at a time.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from datetime import datetime, timezone
from sgp4.api import Satrec, jday

# Make sure we can import sibling modules (src/models, src/data) regardless
# of which directory this script is run from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.train_prediction import DebrisLSTM
from models.collision_risk import (
    compute_miss_distance,
    monte_carlo_collision_probability,
    classify_risk,
)

# Rough "hard-body" size (meters) per object class, used by
# monte_carlo_collision_probability() to compute a combined collision
# radius. These line up with the detection classes trained in Phase 8.
OBJECT_SIZE_M = {
    "small_debris": 0.1,
    "medium_debris": 0.5,
    "large_debris": 2.0,
    "rocket_body": 5.0,
    "defunct_satellite": 3.0,
    "debris": 0.5,      # generic fallback used by classify_object_type()
    "unknown": 0.5,
}

# ─── Paths ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(ROOT)  # project root (one above src/)
CATALOG_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "debris_catalog.csv")
LSTM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "prediction", "best_lstm.pt")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "prediction", "scaler.pkl")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Grounded in Phase 13's measured real-world LSTM error (NORAD 51,
# 10-step horizon, mean error 184.1 km). We use this as a FIXED
# uncertainty (sigma) for Monte Carlo collision probability, instead of
# deriving sigma from the very distance we are testing (that was the old,
# self-referential bug).
LSTM_PREDICTION_ERROR_KM = 184.0

# Must match Phase 6's step_minutes=15 (trajectories.csv sampling), since
# the LSTM was trained on that spacing.
STEP_SECONDS = 900
SEQ_LEN = 20
PRED_LEN = 10
FEATURE_COLS = ["x", "y", "z", "vx", "vy", "vz"]


# ─── Step 1: Load catalog ───────────────────────────────────────────────────

def load_catalog(max_objects: int = None) -> pd.DataFrame:
    """Load the real debris catalog (TLE lines + metadata)."""
    df = pd.read_csv(CATALOG_PATH)
    if max_objects is not None:
        df = df.head(max_objects).copy()
    return df


# ─── Step 2: Propagate current positions with SGP4 ──────────────────────────

def propagate_positions(df_catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Use SGP4 to compute each object's CURRENT position/velocity from its TLE.
    Keeps line1/line2 in the output so later steps (history propagation)
    can re-propagate the same object without re-reading the catalog.
    """
    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day,
                  now.hour, now.minute, now.second + now.microsecond / 1e6)

    records = []
    for _, row in df_catalog.iterrows():
        try:
            sat = Satrec.twoline2rv(row["line1"], row["line2"])
            err, p, v = sat.sgp4(jd, fr)
            if err != 0:
                continue
            records.append({
                "norad_id": row["norad_id"],
                "x": p[0], "y": p[1], "z": p[2],
                "vx": v[0], "vy": v[1], "vz": v[2],
                "line1": str(row["line1"]),
                "line2": str(row["line2"]),
                "mean_motion": row.get("mean_motion", np.nan),
                "object_type": row.get("object_type", "unknown"),
            })
        except Exception:
            continue

    return pd.DataFrame(records)


# ─── Step 3: Classify object type (size proxy for collision radius) ────────

def classify_object_type(row: pd.Series) -> str:
    """Fallback classifier if object_type is missing from the catalog."""
    obj_type = row.get("object_type", None)
    if isinstance(obj_type, str) and obj_type.lower() != "unknown":
        return obj_type
    return "debris"


# ─── Step 4: LSTM trajectory prediction (REAL SGP4 history, not fake) ──────

def build_real_history(line1: str, line2: str, steps: int = SEQ_LEN,
                        dt_seconds: int = STEP_SECONDS) -> np.ndarray:
    """
    Build the past `steps` [x,y,z,vx,vy,vz] points for this object by
    propagating the SAME TLE backward in time with SGP4 - this is the
    object's real orbital history, not a straight-line guess.

    Returns array of shape (steps, 6), oldest point first, most recent
    point last (so index [-1] is "now").
    """
    sat = Satrec.twoline2rv(line1, line2)
    now = datetime.now(timezone.utc)

    history_rows = []
    for t in range(steps, 0, -1):
        past_time = now - pd.Timedelta(seconds=dt_seconds * t)
        jd, fr = jday(past_time.year, past_time.month, past_time.day,
                      past_time.hour, past_time.minute,
                      past_time.second + past_time.microsecond / 1e6)
        err, p, v = sat.sgp4(jd, fr)
        if err != 0:
            raise ValueError(f"SGP4 error {err} in history propagation")
        history_rows.append([p[0], p[1], p[2], v[0], v[1], v[2]])

    return np.array(history_rows, dtype=np.float32)


def predict_trajectories(df_pos: pd.DataFrame, steps: int = PRED_LEN) -> dict:
    """
    For every object in df_pos, build its real SGP4 history (last 20
    points) and feed it to the trained LSTM to predict the next `steps`
    positions.

    Returns: dict mapping norad_id -> predicted positions, shape (steps, 3),
    in REAL KILOMETERS (already inverse-transformed back from the
    scaler's normalized space).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DebrisLSTM(input_size=6, hidden_size=128, num_layers=2,
                        pred_len=PRED_LEN, dropout=0.2)
    model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

    predictions = {}
    for _, row in df_pos.iterrows():
        try:
            history = build_real_history(row["line1"], row["line2"],
                                          steps=SEQ_LEN, dt_seconds=STEP_SECONDS)
        except Exception:
            continue

        # Scale the input history the same way the training data was scaled.
        if scaler:
            history_scaled = scaler.transform(history)
        else:
            history_scaled = history

        x = torch.tensor(history_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).squeeze(0).cpu().numpy()  # shape (steps, 3), SCALED units

        # CRITICAL: the model outputs SCALED (normalized) values, because
        # it was trained on scaler.transform()-ed data (Phase 11/12).
        # We must inverse-transform back to real km before using these
        # values in real-world distance calculations - this step was
        # previously MISSING, which caused every object's predicted path
        # to collapse near [0,0,0] in scaled-space (since StandardScaler
        # centers the data at mean=0), regardless of the object's true
        # physical position. That bug produced the false "CRITICAL"
        # collision alerts between unrelated, distant objects.
        if scaler:
            dummy = np.zeros((pred.shape[0], 6))
            dummy[:, :3] = pred
            pred = scaler.inverse_transform(dummy)[:, :3]

        predictions[row["norad_id"]] = pred

    return predictions


# ─── Step 4b: Simple linear fallback (used only if LSTM/model unavailable) ──

def predict_linear_single(row: pd.Series, steps: int = PRED_LEN,
                           dt: int = STEP_SECONDS) -> np.ndarray:
    """Straight-line extrapolation for ONE object, as a lightweight
    fallback / sanity-check baseline (not used for the real risk numbers
    unless the LSTM model/scaler files are missing)."""
    pos = np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
    vel = np.array([row["vx"], row["vy"], row["vz"]], dtype=np.float64)
    out = np.zeros((steps, 3))
    for t in range(1, steps + 1):
        out[t - 1] = pos + vel * dt * t
    return out


def predict_linear(df_pos: pd.DataFrame, steps: int = PRED_LEN) -> dict:
    """Linear fallback for every object (dict norad_id -> (steps,3))."""
    return {row["norad_id"]: predict_linear_single(row, steps=steps)
            for _, row in df_pos.iterrows()}


# ─── Step 5: Monte Carlo collision risk assessment ──────────────────────────

def assess_collision_risks(df_pos: pd.DataFrame, predictions: dict,
                            n_samples: int = 1000) -> pd.DataFrame:
    """
    For every pair of objects with predicted trajectories, find the
    closest approach across the predicted horizon and run a Monte Carlo
    simulation (using a FIXED uncertainty grounded in the Phase 13
    measured LSTM error) to get a collision probability + risk
    classification.

    Matches the REAL collision_risk.py signatures:
      compute_miss_distance(path1, path2) -> (min_dist, tca_step)
      monte_carlo_collision_probability(pos1, pos2, vel1, vel2,
                                         size1_m, size2_m,
                                         uncertainty_km, samples) -> float
      classify_risk(probability, min_distance_km) -> str  (prob FIRST)
    """
    norad_ids = list(predictions.keys())
    pos_lookup = df_pos.set_index("norad_id")
    results = []

    for i in range(len(norad_ids)):
        for j in range(i + 1, len(norad_ids)):
            id_a, id_b = norad_ids[i], norad_ids[j]
            traj_a, traj_b = predictions[id_a], predictions[id_b]

            n_steps = min(len(traj_a), len(traj_b))
            min_dist, tca_step = compute_miss_distance(traj_a[:n_steps], traj_b[:n_steps])

            # Only bother running Monte Carlo on pairs that are even
            # remotely close - this keeps the full-catalog run fast.
            if min_dist > 1000:  # km
                continue

            row_a = pos_lookup.loc[id_a]
            row_b = pos_lookup.loc[id_b]
            vel_a = np.array([row_a["vx"], row_a["vy"], row_a["vz"]])
            vel_b = np.array([row_b["vx"], row_b["vy"], row_b["vz"]])
            size_a = OBJECT_SIZE_M.get(row_a.get("object_type", "unknown"), 0.5)
            size_b = OBJECT_SIZE_M.get(row_b.get("object_type", "unknown"), 0.5)

            pos_a_tca = traj_a[tca_step]
            pos_b_tca = traj_b[tca_step]

            # uncertainty_km is grounded in Phase 13's independently
            # measured LSTM prediction error (184 km), NOT derived from
            # min_dist itself. The old formula (sigma = max(0.1,
            # min_dist * 0.1)) was self-referential/circular: it made the
            # uncertainty shrink to near-zero exactly when two predicted
            # paths happened to sit close together, which made Monte
            # Carlo over-confident and produced false CRITICAL alerts.
            prob = monte_carlo_collision_probability(
                pos_a_tca, pos_b_tca, vel_a, vel_b, size_a, size_b,
                uncertainty_km=LSTM_PREDICTION_ERROR_KM,
                samples=n_samples,
            )
            risk = classify_risk(prob, min_dist)

            results.append({
                "norad_id_a": id_a,
                "norad_id_b": id_b,
                "tca_step": int(tca_step),
                "min_distance_km": min_dist,
                "collision_probability": prob,
                "risk_level": risk,
            })

    return pd.DataFrame(results)


def assign_object_risks(df_pos: pd.DataFrame, risk_df: pd.DataFrame) -> pd.DataFrame:
    """Attach each object's highest observed risk level (for dashboard coloring).

    classify_risk() returns strings with an emoji baked in (e.g.
    "CRITICAL 🔴"), so we rank by checking which level NAME is contained
    in the string rather than an exact match.
    """
    risk_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    highest = {}

    def rank(risk_level: str) -> int:
        for idx, name in enumerate(risk_order):
            if name in risk_level:
                return idx
        return 0

    for _, row in risk_df.iterrows():
        for oid in (row["norad_id_a"], row["norad_id_b"]):
            current = highest.get(oid, "LOW")
            if rank(row["risk_level"]) > rank(current):
                highest[oid] = row["risk_level"]

    df_pos = df_pos.copy()
    df_pos["risk_level"] = df_pos["norad_id"].map(lambda oid: highest.get(oid, "LOW"))
    return df_pos


# ─── Step 6: Prepare dashboard-ready output ─────────────────────────────────

def prepare_dashboard_data(df_pos: pd.DataFrame, risk_df: pd.DataFrame) -> dict:
    """Bundle everything the Streamlit dashboard needs into one dict."""
    n_critical = int(risk_df["risk_level"].str.contains("CRITICAL").sum()) if len(risk_df) else 0
    return {
        "positions": df_pos,
        "conjunctions": risk_df,
        "n_objects": len(df_pos),
        "n_conjunctions": len(risk_df),
        "n_critical": n_critical,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── Full pipeline ───────────────────────────────────────────────────────────

def run_full_pipeline(max_objects: int = 300):
    print("=" * 60)
    print("REAL PIPELINE: SGP4 -> LSTM -> Monte Carlo -> Dashboard")
    print("=" * 60)

    print(f"\n[1/5] Loading catalog (max {max_objects} objects)...")
    df_catalog = load_catalog(max_objects=max_objects)
    print(f"      Loaded {len(df_catalog)} objects")

    print("\n[2/5] Propagating current positions with SGP4...")
    df_pos = propagate_positions(df_catalog)
    print(f"      Propagated {len(df_pos)} objects successfully")

    print("\n[3/5] Predicting future trajectories with LSTM...")
    predictions = predict_trajectories(df_pos, steps=PRED_LEN)
    print(f"      Predicted trajectories for {len(predictions)} objects")

    print("\n[4/5] Assessing collision risks (Monte Carlo)...")
    risk_df = assess_collision_risks(df_pos, predictions)
    print(f"      Found {len(risk_df)} conjunction(s) within 1000km")
    if len(risk_df):
        print(risk_df.sort_values("min_distance_km").to_string(index=False))

    print("\n[5/5] Preparing dashboard data...")
    df_pos = assign_object_risks(df_pos, risk_df)
    dashboard_data = prepare_dashboard_data(df_pos, risk_df)

    out_path = os.path.join(OUTPUTS_DIR, "real_collision_risks.csv")
    risk_df.to_csv(out_path, index=False)
    print(f"\nSaved conjunction report to {out_path}")

    pos_out_path = os.path.join(OUTPUTS_DIR, "real_positions.csv")
    df_pos.to_csv(pos_out_path, index=False)
    print(f"Saved object positions + risk levels to {pos_out_path}")

    return dashboard_data


if __name__ == "__main__":
    run_full_pipeline(max_objects=300)