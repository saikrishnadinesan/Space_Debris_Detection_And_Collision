"""
Phase 5: SGP4 Verification
============================
Computes a real position for one known object (NORAD ID 51) and
sanity-checks it against the Kepler-formula altitude we already
verified in Phase 3's orbit classification fix.
"""

import pandas as pd
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import tle_to_position

CATALOG_PATH = "data/processed/debris_catalog.csv"
EARTH_RADIUS_KM = 6378.137
EARTH_MU = 398600.4418


def kepler_altitude_estimate(mean_motion: float) -> float:
    """Same corrected formula from collect_data.py, used here only
    as an independent sanity check."""
    period_min = 1440 / mean_motion
    period_sec = period_min * 60
    a = (EARTH_MU * (period_sec / (2 * 3.14159)) ** 2) ** (1 / 3)
    return a - EARTH_RADIUS_KM


if __name__ == "__main__":
    df = pd.read_csv(CATALOG_PATH)
    row = df[df["norad_id"] == 51].iloc[0]

    now = datetime.utcnow()
    pos, vel = tle_to_position(row["line1"], row["line2"], now)

    if pos is None:
        print("SGP4 returned an error - check the TLE lines.")
    else:
        import numpy as np
        distance_from_center = np.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
        sgp4_altitude = distance_from_center - EARTH_RADIUS_KM
        speed = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)

        print(f"Object: {row['name']} (NORAD {row['norad_id']}, {row['orbit_class']})")
        print(f"Time queried (UTC): {now}")
        print()
        print(f"Position (km): X={pos[0]:.2f}  Y={pos[1]:.2f}  Z={pos[2]:.2f}")
        print(f"Velocity (km/s): VX={vel[0]:.4f}  VY={vel[1]:.4f}  VZ={vel[2]:.4f}")
        print(f"Speed: {speed:.4f} km/s")
        print(f"SGP4-derived altitude: {sgp4_altitude:.2f} km")
        print()

        kepler_est = kepler_altitude_estimate(row["mean_motion"])
        print(f"Kepler-formula estimate (avg/circular altitude): {kepler_est:.2f} km")
        print(f"Difference: {abs(sgp4_altitude - kepler_est):.2f} km")
        print()
        print("Note: some difference is EXPECTED - SGP4 gives the exact")
        print("altitude at THIS instant, while the Kepler estimate is an")
        print("average based on orbital period alone. Since eccentricity")
        print(f"here is only {row['eccentricity']:.4f} (nearly circular),")
        print("the two numbers should be reasonably close (within a few")
        print("hundred km, not thousands).")
