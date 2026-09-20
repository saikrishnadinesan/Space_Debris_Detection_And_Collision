"""
Phase 4: TLE Decoder — Beginner Reference Tool
================================================
Reads one row from the real catalog and prints a human-readable
breakdown of its TLE, so you can always check what a raw TLE means.
"""

import pandas as pd

CATALOG_PATH = "data/processed/debris_catalog.csv"


def decode_tle(line1: str, line2: str) -> None:
    print("=" * 60)
    print("LINE 1 - Identity, Epoch, Drag")
    print("=" * 60)
    print(f"Satellite number   : {line1[2:7].strip()}")
    print(f"Classification     : {line1[7]}")
    print(f"Intl designator    : {line1[9:17].strip()}  (launch year/number/piece)")
    print(f"Epoch year         : 20{line1[18:20]}")
    print(f"Epoch day-of-year  : {line1[20:32].strip()}")
    print(f"Mean motion deriv1 : {line1[33:43].strip()}")
    print(f"BSTAR drag term    : {line1[53:61].strip()}")

    print()
    print("=" * 60)
    print("LINE 2 - Orbit Shape and Motion")
    print("=" * 60)
    print(f"Inclination (deg)  : {line2[8:16].strip()}")
    print(f"RAAN (deg)         : {line2[17:25].strip()}")
    eccentricity = "0." + line2[26:33].strip()
    print(f"Eccentricity       : {eccentricity}")
    print(f"Arg of perigee     : {line2[34:42].strip()}")
    print(f"Mean anomaly       : {line2[43:51].strip()}")
    print(f"Mean motion (rev/day): {line2[52:63].strip()}")
    print(f"Revolution number  : {line2[63:68].strip()}")


if __name__ == "__main__":
    df = pd.read_csv(CATALOG_PATH)
    row = df.iloc[0]
    print(f"Decoding object: {row['name']}  (NORAD ID {row['norad_id']}, {row['orbit_class']})")
    print()
    decode_tle(row["line1"], row["line2"])
