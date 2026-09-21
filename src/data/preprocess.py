"""
Phase 3: Data Preprocessing
============================
Converts raw TLE data into orbital positions,
generates synthetic imagery, and prepares datasets
for model training.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm


PROCESSED_DIR = "data/processed"
SYNTHETIC_DIR = "data/synthetic"
ANNOTATIONS_DIR = "data/annotations"

for d in [PROCESSED_DIR, SYNTHETIC_DIR, ANNOTATIONS_DIR]:
    os.makedirs(d, exist_ok=True)


# --- Step 1: Compute Positions from TLE ---

def tle_to_position(line1: str, line2: str, dt: datetime) -> tuple:
    """
    Compute satellite/debris position in ECI (Earth-Centered Inertial) coords.
    Returns (x, y, z) in km and (vx, vy, vz) in km/s.
    """
    satellite = Satrec.twoline2rv(line1, line2)
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    error, position, velocity = satellite.sgp4(jd, fr)

    if error == 0:
        return position, velocity
    return None, None


def generate_trajectory(line1: str, line2: str, hours: int = 48, step_minutes: int = 15) -> pd.DataFrame:
    """Generate a time series of positions for a debris object."""
    records = []
    start_time = datetime.utcnow()

    for i in range(0, hours * 60, step_minutes):
        dt = start_time + timedelta(minutes=i)
        pos, vel = tle_to_position(line1, line2, dt)

        if pos:
            records.append({
                "timestamp": dt,
                "x": pos[0], "y": pos[1], "z": pos[2],
                "vx": vel[0], "vy": vel[1], "vz": vel[2],
                "speed_km_s": np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2),
                "altitude_km": np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2) - 6371
            })

    return pd.DataFrame(records)


def process_catalog(catalog_path: str, max_objects: int = 500) -> pd.DataFrame:
    """Process TLE catalog into position trajectories."""
    print("Processing TLE catalog into trajectories...")
    df = pd.read_csv(catalog_path)
    df = df.head(max_objects)

    all_trajectories = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing positions"):
        try:
            traj = generate_trajectory(row["line1"], row["line2"], hours=48, step_minutes=15)
            if not traj.empty:
                traj["norad_id"] = row["norad_id"]
                traj["name"] = row["name"]
                traj["orbit_class"] = row["orbit_class"]
                all_trajectories.append(traj)
        except Exception:
            continue

    if all_trajectories:
        result = pd.concat(all_trajectories, ignore_index=True)
        output = os.path.join(PROCESSED_DIR, "trajectories.csv")
        result.to_csv(output, index=False)
        print(f"Saved {len(result)} trajectory points to {output}")
        return result

    return pd.DataFrame()


# --- Step 2: Generate Synthetic Detection Images ---

def generate_synthetic_image(num_debris: int = 5, image_size: int = 640) -> tuple:
    """
    Generate a synthetic space image with debris objects.
    Returns (image array, list of bounding boxes).

    Classes:
    0 = small_debris, 1 = medium_debris, 2 = large_debris,
    3 = rocket_body, 4 = defunct_satellite
    """
    img = np.random.normal(5, 3, (image_size, image_size, 3)).clip(0, 30).astype(np.uint8)

    num_stars = np.random.randint(50, 200)
    for _ in range(num_stars):
        sx, sy = np.random.randint(0, image_size, 2)
        brightness = np.random.randint(150, 255)
        img[sy, sx] = [brightness, brightness, brightness]

    bboxes = []

    for _ in range(num_debris):
        cls = np.random.randint(0, 5)

        size_map = {0: (3, 8), 1: (8, 20), 2: (20, 40), 3: (25, 50), 4: (30, 60)}
        min_s, max_s = size_map[cls]
        w = np.random.randint(min_s, max_s)
        h = np.random.randint(min_s, max_s)

        x = np.random.randint(w, image_size - w)
        y = np.random.randint(h, image_size - h)

        color_map = {
            0: [180, 180, 200],
            1: [200, 200, 150],
            2: [220, 180, 180],
            3: [180, 220, 180],
            4: [180, 180, 220],
        }
        color = color_map[cls]

        for dy in range(-h//2, h//2):
            for dx in range(-w//2, w//2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < image_size and 0 <= ny < image_size:
                    dist = np.sqrt(dx**2 + dy**2) / max(w, h) * 2
                    intensity = max(0, 1 - dist)
                    img[ny, nx] = np.clip(
                        img[ny, nx] + np.array(color) * intensity, 0, 255
                    ).astype(np.uint8)

        cx = x / image_size
        cy = y / image_size
        nw = w / image_size
        nh = h / image_size
        bboxes.append((cls, cx, cy, nw, nh))

    return img, bboxes


def generate_synthetic_dataset(num_images: int = 1000, image_size: int = 640):
    """Generate a full synthetic dataset for YOLOv8 training."""
    print(f"\nGenerating {num_images} synthetic training images...")

    splits = {"train": int(0.7 * num_images),
              "val": int(0.15 * num_images),
              "test": int(0.15 * num_images)}

    for split, count in splits.items():
        img_dir = os.path.join(SYNTHETIC_DIR, "images", split)
        lbl_dir = os.path.join(SYNTHETIC_DIR, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i in tqdm(range(count), desc=f"Generating {split}"):
            num_obj = np.random.randint(1, 8)
            img, bboxes = generate_synthetic_image(num_obj, image_size)

            img_path = os.path.join(img_dir, f"debris_{split}_{i:05d}.png")
            plt.imsave(img_path, img)

            lbl_path = os.path.join(lbl_dir, f"debris_{split}_{i:05d}.txt")
            with open(lbl_path, "w") as f:
                for bbox in bboxes:
                    f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")

    print(f"Synthetic dataset saved to {SYNTHETIC_DIR}")
    create_yolo_yaml()


def create_yolo_yaml():
    """Create the dataset YAML config for YOLOv8 training."""
    yaml_content = f"""# Space Debris Detection Dataset
path: {os.path.abspath(SYNTHETIC_DIR)}
train: images/train
val: images/val
test: images/test

nc: 5  # number of classes
names:
  0: small_debris
  1: medium_debris
  2: large_debris
  3: rocket_body
  4: defunct_satellite
"""
    yaml_path = os.path.join(SYNTHETIC_DIR, "debris.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"YOLO config saved to {yaml_path}")


# --- Main ---

if __name__ == "__main__":
    print("=" * 60)
    print("SPACE DEBRIS - DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Process TLE catalog (if available)
    catalog_path = os.path.join(PROCESSED_DIR, "debris_catalog.csv")
    if os.path.exists(catalog_path):
        trajectories = process_catalog(catalog_path)
        print(f"\nGenerated {len(trajectories)} trajectory points")
    else:
        print("No catalog found. Run collect_data.py first.")
        print("Skipping trajectory generation...")

    # Step 2: Generate synthetic training data - DISABLED FOR PHASE 6
    # We'll re-enable this in Phase 7, once the trajectory dataset
    # (Phase 6) is generated and verified on its own.
    # print("\nGenerating synthetic training dataset...")
    # generate_synthetic_dataset(num_images=500, image_size=640)

    print("\nPreprocessing complete (Phase 6: trajectories only).")