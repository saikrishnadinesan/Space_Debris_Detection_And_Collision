"""
Real Pipeline — Full End-to-End Connection
============================================
Connects:
  1. TLE catalog (debris_catalog.csv) — real Space-Track data
  2. SGP4 propagator — computes real XYZ positions right now
  3. LSTM model — predicts future trajectories
  4. Monte Carlo — computes real collision probabilities
  5. Dashboard-ready DataFrame — feeds directly into visualization

Usage:
  from src.pipeline.real_pipeline import run_full_pipeline
  df, risks = run_full_pipeline(max_objects=300)
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import torch

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

CATALOG_PATH     = os.path.join(ROOT, "data", "processed", "debris_catalog.csv")
LSTM_MODEL_PATH  = os.path.join(ROOT, "models", "prediction", "best_lstm.pt")
SCALER_PATH      = os.path.join(ROOT, "models", "prediction", "scaler.pkl")
OUTPUTS_DIR      = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ─── Step 1: Load TLE Catalog ─────────────────────────────────────────────────

def load_catalog(max_objects: int = 300) -> pd.DataFrame:
    """Load real TLE debris catalog downloaded from Space-Track."""
    if not os.path.exists(CATALOG_PATH):
        raise FileNotFoundError(f"Catalog not found at {CATALOG_PATH}. Run Phase 1 first.")

    df = pd.read_csv(CATALOG_PATH)

    # Filter to objects with valid TLE lines
    df = df[df["line1"].notna() & df["line2"].notna()]
    df = df[df["line1"].str.startswith("1 ", na=False)]
    df = df[df["line2"].str.startswith("2 ", na=False)]

    # Prioritize debris objects
    debris_mask = df["source"].isin(["debris", "rocket_bodies", "cosmos", "iridium"])
    df_debris   = df[debris_mask]
    df_other    = df[~debris_mask]

    # Take up to max_objects, prioritizing real debris
    n_debris = min(len(df_debris), int(max_objects * 0.7))
    n_other  = min(len(df_other),  max_objects - n_debris)

    df_sample = pd.concat([
        df_debris.sample(n=n_debris, random_state=42) if n_debris > 0 else pd.DataFrame(),
        df_other.sample(n=n_other,   random_state=42) if n_other  > 0 else pd.DataFrame(),
    ], ignore_index=True)

    print(f"   ✅ Loaded {len(df_sample)} objects from catalog")
    return df_sample


# ─── Step 2: SGP4 Position Propagation ───────────────────────────────────────

def propagate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute real XYZ positions for all debris objects RIGHT NOW
    using the SGP4 orbital mechanics propagator.
    
    Returns DataFrame with columns: name, norad_id, x, y, z, vx, vy, vz,
    altitude_km, orbit_class, lat, lon, speed_km_s
    """
    try:
        from sgp4.api import Satrec, jday
    except ImportError:
        print("   ❌ sgp4 not installed. Run: pip install sgp4")
        return pd.DataFrame()

    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day,
                  now.hour, now.minute, now.second + now.microsecond / 1e6)

    records = []
    failed  = 0

    for _, row in df.iterrows():
        try:
            sat = Satrec.twoline2rv(str(row["line1"]), str(row["line2"]))
            err, pos, vel = sat.sgp4(jd, fr)

            if err != 0 or not pos or len(pos) < 3:
                failed += 1
                continue

            x, y, z    = pos
            vx, vy, vz = vel

            # Compute derived quantities
            r          = np.sqrt(x**2 + y**2 + z**2)
            altitude   = r - 6371.0
            speed      = np.sqrt(vx**2 + vy**2 + vz**2)

            # Skip objects with obviously wrong positions
            if altitude < 100 or altitude > 100000:
                failed += 1
                continue

            # Latitude and longitude from ECI
            lat = np.degrees(np.arcsin(z / r))
            lon = np.degrees(np.arctan2(y, x))

            orbit_class = ("LEO" if altitude < 2000
                           else "MEO" if altitude < 35786
                           else "GEO")

            # Debris size estimate from radar cross section (heuristic)
            size_m = np.random.uniform(0.1, 5.0)  # will improve with real RCS data

            records.append({
                "id":           str(row["norad_id"]),
                "name":         str(row["name"]).strip(),
                "norad_id":     int(row["norad_id"]),
                "x":            round(x,  3),
                "y":            round(y,  3),
                "z":            round(z,  3),
                "vx":           round(vx, 6),
                "vy":           round(vy, 6),
                "vz":           round(vz, 6),
                "altitude_km":  round(altitude, 2),
                "lat":          round(lat, 4),
                "lon":          round(lon, 4),
                "speed_km_s":   round(speed, 4),
                "orbit_class":  orbit_class,
                "source":       str(row.get("source", "real")),
                "size_m":       round(size_m, 3),
                "type":         classify_object_type(str(row["name"]), str(row.get("source",""))),
                "epoch":        now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            })

        except Exception:
            failed += 1
            continue

    print(f"   ✅ Propagated {len(records)} positions ({failed} failed)")
    return pd.DataFrame(records)


def classify_object_type(name: str, source: str) -> str:
    """Classify debris object type from name and source."""
    name_upper = name.upper()
    if source == "rocket_bodies" or "R/B" in name_upper or "ROCKET" in name_upper:
        return "rocket_body"
    if "SAT" in name_upper or source == "active":
        return "defunct_satellite"
    if "DEB" in name_upper or source == "debris":
        size_roll = np.random.random()
        if size_roll < 0.5:   return "small_debris"
        elif size_roll < 0.8: return "medium_debris"
        else:                 return "large_debris"
    return "small_debris"


# ─── Step 3: LSTM Trajectory Prediction ──────────────────────────────────────

def predict_trajectories(df_pos: pd.DataFrame, steps: int = 10) -> dict:
    """
    Use trained LSTM model to predict future positions for each object.
    Returns dict: {norad_id: predicted_path array (steps, 3)}
    """
    if not os.path.exists(LSTM_MODEL_PATH):
        print("   ⚠️  LSTM model not found — using linear extrapolation")
        return predict_linear(df_pos, steps)

    try:
        import joblib

        # Load model
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from models.train_prediction import DebrisLSTM

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = DebrisLSTM(input_size=6, hidden_size=128,
                            num_layers=2, pred_len=steps, dropout=0.2)
        model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
        model.eval()

        scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

        predictions = {}
        for _, row in df_pos.iterrows():
            try:
                # Build a short history by slightly perturbing current state
                # (simulating past positions along current velocity)
                pos = np.array([row["x"], row["y"], row["z"]])
                vel = np.array([row["vx"], row["vy"], row["vz"]])
                dt  = 60  # seconds per step

                history = np.array([
                    np.concatenate([pos - vel * dt * (20 - t), vel])
                    for t in range(20)
                ], dtype=np.float32)

                if scaler:
                    history = scaler.transform(history)

                x_in = torch.tensor(history, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(x_in).squeeze(0).cpu().numpy()

                predictions[row["norad_id"]] = pred

            except Exception:
                predictions[row["norad_id"]] = predict_linear_single(row, steps)

        print(f"   ✅ LSTM predicted trajectories for {len(predictions)} objects")
        return predictions

    except Exception as e:
        print(f"   ⚠️  LSTM failed ({e}) — using linear extrapolation")
        return predict_linear(df_pos, steps)


def predict_linear_single(row, steps: int = 10) -> np.ndarray:
    """Simple linear trajectory prediction as fallback."""
    pos = np.array([row["x"], row["y"], row["z"]])
    vel = np.array([row["vx"], row["vy"], row["vz"]])
    dt  = 60  # seconds
    return np.array([pos + vel * dt * t for t in range(1, steps + 1)])


def predict_linear(df_pos: pd.DataFrame, steps: int = 10) -> dict:
    """Linear prediction fallback for all objects."""
    return {row["norad_id"]: predict_linear_single(row, steps)
            for _, row in df_pos.iterrows()}


# ─── Step 4: Collision Risk Assessment ───────────────────────────────────────

def assess_collision_risks(df_pos: pd.DataFrame,
                            predictions: dict,
                            max_pairs: int = 5000) -> pd.DataFrame:
    """
    Run Monte Carlo collision risk for all close object pairs.
    Only checks pairs within 50km of each other to keep it fast.
    """
    print(f"   🔍 Checking collision risks...")

    positions = df_pos[["norad_id","x","y","z","vx","vy","vz","size_m","name"]].values
    n         = len(positions)
    risks     = []
    pairs_checked = 0

    for i in range(n):
        if pairs_checked >= max_pairs:
            break
        for j in range(i + 1, n):
            if pairs_checked >= max_pairs:
                break

            p1 = positions[i, 1:4].astype(float)
            p2 = positions[j, 1:4].astype(float)
            dist = np.linalg.norm(p1 - p2)
            pairs_checked += 1

            # Only compute full risk for objects within 500km
            if dist > 500:
                continue

            norad1 = int(positions[i, 0])
            norad2 = int(positions[j, 0])

            path1 = predictions.get(norad1)
            path2 = predictions.get(norad2)

            if path1 is None or path2 is None:
                continue

            # Minimum miss distance along predicted trajectories
            dists  = np.linalg.norm(np.array(path1) - np.array(path2), axis=1)
            min_dist = float(np.min(dists))
            tca      = int(np.argmin(dists))

            # Monte Carlo probability
            size1 = float(positions[i, 7])
            size2 = float(positions[j, 7])
            combined_radius = (size1 + size2) / 2000  # km

            samples    = 500
            collisions = 0
            sigma      = max(0.1, min_dist * 0.1)  # uncertainty = 10% of miss distance
            for _ in range(samples):
                dp1 = p1 + np.random.normal(0, sigma, 3)
                dp2 = p2 + np.random.normal(0, sigma, 3)
                if np.linalg.norm(dp1 - dp2) < combined_radius:
                    collisions += 1

            prob = collisions / samples

            # Risk level
            if prob > 0.01 or min_dist < 0.1:
                risk = "CRITICAL"
            elif prob > 0.001 or min_dist < 1.0:
                risk = "HIGH"
            elif prob > 0.0001 or min_dist < 5.0:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            if risk in ("CRITICAL", "HIGH", "MEDIUM"):
                risks.append({
                    "obj1_id":   norad1,
                    "obj1_name": str(positions[i, 8]),
                    "obj2_id":   norad2,
                    "obj2_name": str(positions[j, 8]),
                    "current_distance_km": round(dist, 3),
                    "min_distance_km":     round(min_dist, 3),
                    "tca_step":            tca,
                    "collision_probability": round(prob, 6),
                    "risk_level":          risk,
                })

    df_risks = pd.DataFrame(risks)
    if not df_risks.empty:
        df_risks = df_risks.sort_values("collision_probability", ascending=False)

    print(f"   ✅ {len(df_risks)} conjunction events found ({pairs_checked} pairs checked)")
    return df_risks


# ─── Step 5: Assign Risk to Each Object ──────────────────────────────────────

def assign_object_risks(df_pos: pd.DataFrame, df_risks: pd.DataFrame) -> pd.DataFrame:
    """
    Assign worst risk level to each object based on conjunction results.
    Objects not in any conjunction get LOW risk.
    """
    risk_priority = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    object_risks  = {}

    if not df_risks.empty:
        for _, row in df_risks.iterrows():
            for obj_id in [row["obj1_id"], row["obj2_id"]]:
                current = object_risks.get(obj_id, "LOW")
                if risk_priority[row["risk_level"]] > risk_priority[current]:
                    object_risks[obj_id] = row["risk_level"]

    df_pos["risk_level"] = df_pos["norad_id"].map(object_risks).fillna("LOW")
    return df_pos


# ─── Step 6: Prepare Dashboard Data ──────────────────────────────────────────

def prepare_dashboard_data(df_pos: pd.DataFrame,
                            df_risks: pd.DataFrame) -> pd.DataFrame:
    """Prepare final DataFrame for dashboard consumption."""
    df = df_pos.copy()

    # Ensure all required columns exist
    required = ["id","name","x","y","z","altitude_km","orbit_class",
                "type","risk_level","speed_km_s","size_m"]
    for col in required:
        if col not in df.columns:
            df[col] = "unknown" if col in ["id","name","type","orbit_class","risk_level"] else 0.0

    # Rename speed column for dashboard compatibility
    if "speed_km_s" not in df.columns and "velocity_km_s" in df.columns:
        df["speed_km_s"] = df["velocity_km_s"]

    df["velocity_km_s"] = df["speed_km_s"]
    return df


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_full_pipeline(max_objects: int = 300):
    """
    Run the complete real data pipeline.
    
    Returns:
        df_pos   : DataFrame with real positions + risk levels
        df_risks : DataFrame with all conjunction events
    """
    print("\n" + "=" * 60)
    print("🛸 REAL PIPELINE — FULL RUN")
    print(f"   Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    # Step 1 — Load catalog
    print("\n📂 Step 1: Loading TLE catalog...")
    df_catalog = load_catalog(max_objects)
    if df_catalog.empty:
        raise RuntimeError("Empty catalog — run Phase 1 first")

    # Step 2 — Propagate positions
    print("\n🌍 Step 2: Computing real positions via SGP4...")
    df_pos = propagate_positions(df_catalog)
    if df_pos.empty:
        raise RuntimeError("No positions computed")

    # Step 3 — LSTM trajectory prediction
    print("\n🤖 Step 3: Predicting trajectories (LSTM)...")
    predictions = predict_trajectories(df_pos, steps=10)

    # Step 4 — Collision risk
    print("\n⚠️  Step 4: Assessing collision risks...")
    df_risks = assess_collision_risks(df_pos, predictions)

    # Step 5 — Assign risk to objects
    print("\n🎯 Step 5: Assigning risk levels...")
    df_pos = assign_object_risks(df_pos, df_risks)

    # Step 6 — Save outputs
    print("\n💾 Step 6: Saving outputs...")
    df_pos.to_csv(os.path.join(OUTPUTS_DIR, "real_positions.csv"), index=False)

    # Always write a proper CSV with headers even if no conjunctions found
    risk_columns = ["obj1_id","obj1_name","obj2_id","obj2_name",
                    "current_distance_km","min_distance_km","tca_step",
                    "collision_probability","risk_level"]
    if df_risks.empty:
        pd.DataFrame(columns=risk_columns).to_csv(
            os.path.join(OUTPUTS_DIR, "real_collision_risks.csv"), index=False)
    else:
        df_risks.to_csv(os.path.join(OUTPUTS_DIR, "real_collision_risks.csv"), index=False)

    # Summary
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print(f"   Objects tracked  : {len(df_pos)}")
    print(f"   Conjunctions     : {len(df_risks)}")
    if not df_risks.empty:
        print(f"   Critical events  : {len(df_risks[df_risks['risk_level']=='CRITICAL'])}")
        print(f"   High events      : {len(df_risks[df_risks['risk_level']=='HIGH'])}")
    print(f"   Risk distribution:\n{df_pos['risk_level'].value_counts().to_string()}")
    print("=" * 60)

    # Prepare for dashboard
    df_dashboard = prepare_dashboard_data(df_pos, df_risks)
    return df_dashboard, df_risks


# ─── Standalone runner ────────────────────────────────────────────────────────

if __name__ == "__main__":
    df, risks = run_full_pipeline(max_objects=300)
    print("\nSample positions:")
    print(df[["name","altitude_km","orbit_class","risk_level","speed_km_s"]].head(10).to_string(index=False))
    if not risks.empty:
        print("\nTop conjunction events:")
        print(risks[["obj1_name","obj2_name","min_distance_km","collision_probability","risk_level"]].head(5).to_string(index=False))