"""
Phase 2: Data Collection — spacetrack Python library (Fixed)
=============================================================
Uses correct spacetrack library argument format.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.getenv("SPACETRACK_USER", "")
PASSWORD = os.getenv("SPACETRACK_PASS", "")

RAW_DATA_DIR  = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


EARTH_RADIUS_KM = 6378.137
EARTH_MU = 398600.4418  # Earth's gravitational parameter, km^3/s^2

def compute_orbit_class(mean_motion: float) -> str:
    try:
        period_min = 1440 / mean_motion          # revs/day -> minutes per orbit
        period_sec = period_min * 60
        semi_major_axis = (EARTH_MU * (period_sec / (2 * 3.14159)) ** 2) ** (1/3)
        alt = semi_major_axis - EARTH_RADIUS_KM
        return "LEO" if alt < 2000 else ("MEO" if alt < 35786 else "GEO")
    except:
        return "LEO"


def parse_tle_text(text: str, source: str = "") -> pd.DataFrame:
    """Parse raw TLE text into DataFrame."""
    records = []
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    i = 0
    while i < len(lines) - 1:
        try:
            # 3-line format
            if i + 2 < len(lines) and not lines[i].startswith("1 ") and not lines[i].startswith("2 ") \
               and lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
                name, line1, line2 = lines[i], lines[i+1], lines[i+2]
                i += 3
            # 2-line format
            elif lines[i].startswith("1 ") and i+1 < len(lines) and lines[i+1].startswith("2 "):
                name  = f"OBJ-{lines[i][2:7]}"
                line1, line2 = lines[i], lines[i+1]
                i += 2
            else:
                i += 1
                continue

            mm = float(line2[52:63])
            records.append({
                "name":         name,
                "norad_id":     int(line1[2:7]),
                "epoch":        line1[18:32].strip(),
                "inclination":  float(line2[8:16]),
                "eccentricity": float("0." + line2[26:33]),
                "mean_motion":  round(mm, 8),
                "orbit_class":  compute_orbit_class(mm),
                "source":       source,
                "line1":        line1,
                "line2":        line2,
            })
        except:
            i += 1
    return pd.DataFrame(records)


def download_with_spacetrack() -> pd.DataFrame:
    """Download using official spacetrack library with correct API syntax."""
    try:
        from spacetrack import SpaceTrackClient
        import spacetrack.operators as op
    except ImportError:
        print("❌ Run: pip install spacetrack")
        return pd.DataFrame()

    print(f"🔐 Logging in as {USERNAME}...")
    client = SpaceTrackClient(USERNAME, PASSWORD)
    all_dfs = []

    # ── Query 1: All debris objects ──────────────────────────────
    try:
        print("\n📡 Downloading: debris objects...")
        data = client.gp(
            object_type="DEBRIS",
            epoch=op.greater_than(">now-30"),
            format="tle",
            limit=5000
        )
        if data and len(data) > 10:
            df = parse_tle_text(data, "debris")
            if not df.empty:
                all_dfs.append(df)
                print(f"   ✅ {len(df)} debris objects")
    except Exception as e:
        print(f"   ❌ {e}")

    # ── Query 2: Rocket bodies ───────────────────────────────────
    try:
        print("\n📡 Downloading: rocket bodies...")
        data = client.gp(
            object_type="ROCKET BODY",
            format="tle",
            limit=2000
        )
        if data and len(data) > 10:
            df = parse_tle_text(data, "rocket_bodies")
            if not df.empty:
                all_dfs.append(df)
                print(f"   ✅ {len(df)} rocket bodies")
    except Exception as e:
        print(f"   ❌ {e}")

    # ── Query 3: Recent objects (last 30 days) ───────────────────
    try:
        print("\n📡 Downloading: recent launches (last 30 days)...")
        data = client.gp(
            epoch=op.greater_than(">now-30"),
            format="tle",
            limit=3000
        )
        if data and len(data) > 10:
            df = parse_tle_text(data, "recent")
            if not df.empty:
                all_dfs.append(df)
                print(f"   ✅ {len(df)} recent objects")
    except Exception as e:
        print(f"   ❌ {e}")

    # ── Query 4: LEO objects (period < 128 min) ──────────────────
    try:
        print("\n📡 Downloading: LEO objects...")
        data = client.gp(
            period=op.less_than(128),
            format="tle",
            limit=3000
        )
        if data and len(data) > 10:
            df = parse_tle_text(data, "leo")
            if not df.empty:
                all_dfs.append(df)
                print(f"   ✅ {len(df)} LEO objects")
    except Exception as e:
        print(f"   ❌ {e}")

    # ── Query 5: All active satellites ──────────────────────────
    try:
        print("\n📡 Downloading: all catalogued objects...")
        data = client.gp(
            format="tle",
            limit=10000
        )
        if data and len(data) > 10:
            df = parse_tle_text(data, "catalog")
            if not df.empty:
                all_dfs.append(df)
                print(f"   ✅ {len(df)} catalogued objects")
    except Exception as e:
        print(f"   ❌ {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["norad_id"])
        print(f"\n✅ Total real objects downloaded: {len(combined)}")
        return combined

    return pd.DataFrame()


def generate_synthetic(n: int = 500) -> pd.DataFrame:
    """Generate synthetic orbital records as supplement."""
    print(f"\n🔧 Generating {n} synthetic records...")
    np.random.seed(42)
    records = []
    for i in range(n):
        orbit = np.random.choice(["LEO","MEO","GEO"], p=[0.7,0.2,0.1])
        alt   = {"LEO": np.random.uniform(300,2000),
                 "MEO": np.random.uniform(2000,20000),
                 "GEO": np.random.uniform(35500,36000)}[orbit]
        r  = 6378.137 + alt
        mm = 1440 / (2 * 3.14159 * (r**1.5 / 398600.4418**0.5) / 60)
        records.append({
            "name":         f"SYNTHETIC-{i+1:05d}",
            "norad_id":     99000 + i,
            "epoch":        datetime.utcnow().strftime("%y%j.%f")[:14],
            "inclination":  round(np.random.uniform(0, 98), 4),
            "eccentricity": round(np.random.uniform(0.0001, 0.01), 7),
            "mean_motion":  round(mm, 8),
            "orbit_class":  orbit,
            "source":       "synthetic",
            "line1": "", "line2": "",
        })
    df = pd.DataFrame(records)
    print(f"   ✅ {len(df)} synthetic records")
    return df


def collect_all_data():
    print("=" * 60)
    print("🛸 SPACE DEBRIS — DATA COLLECTION")
    print("=" * 60)

    # Try real download
    real_df = download_with_spacetrack()

    # Supplement with synthetic
    synthetic_df = generate_synthetic(300)

    if not real_df.empty:
        combined = pd.concat([real_df, synthetic_df], ignore_index=True)
    else:
        print("\n⚠️  No real data — using synthetic only")
        combined = generate_synthetic(800)

    combined = combined.drop_duplicates(subset=["norad_id"])

    out = os.path.join(PROCESSED_DIR, "debris_catalog.csv")
    combined.to_csv(out, index=False)

    print("\n" + "=" * 60)
    print(f"✅ COLLECTION COMPLETE")
    print(f"   Total objects : {len(combined)}")
    print(f"   Orbit split   :\n{combined['orbit_class'].value_counts().to_string()}")
    print(f"   Saved → {out}")
    print("=" * 60)
    return combined


if __name__ == "__main__":
    df = collect_all_data()
    print("\nSample:")
    print(df[["name","norad_id","orbit_class"]].head(10).to_string(index=False))