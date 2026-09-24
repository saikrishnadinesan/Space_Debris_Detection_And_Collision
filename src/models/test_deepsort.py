"""
Phase 10: DeepSORT Tracking Demo
==================================
Generates a short sequence of synthetic frames with debris objects
drifting slightly between frames, runs YOLO detection on each frame,
and feeds those detections into DeepSORT to demonstrate persistent
object IDs across frames.

NOTE: this is a standalone proof-of-concept. The real dashboard uses
SGP4-computed positions, not a live video feed, so DeepSORT is not
connected to the live pipeline - this demonstrates the technique works.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

OUTPUT_DIR = "outputs/tracking_demo"
MODEL_PATH = "models/detection/debris_detector3/weights/best.pt"
CLASS_NAMES = ["small_debris", "medium_debris", "large_debris",
               "rocket_body", "defunct_satellite"]
NUM_FRAMES = 8
IMAGE_SIZE = 416

os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_frame(objects, image_size=IMAGE_SIZE):
    """Draw one synthetic frame with the given objects at their current positions."""
    img = np.random.normal(5, 3, (image_size, image_size, 3)).clip(0, 30).astype(np.uint8)

    for _ in range(100):
        sx, sy = np.random.randint(0, image_size, 2)
        b = np.random.randint(150, 255)
        img[sy, sx] = [b, b, b]

    color_map = {0: [180, 180, 200], 1: [200, 200, 150], 2: [220, 180, 180],
                 3: [180, 220, 180], 4: [180, 180, 220]}

    for obj in objects:
        x, y, w, h, cls = obj["x"], obj["y"], obj["w"], obj["h"], obj["cls"]
        color = color_map[cls]
        for dy in range(-h // 2, h // 2):
            for dx in range(-w // 2, w // 2):
                nx, ny = int(x + dx), int(y + dy)
                if 0 <= nx < image_size and 0 <= ny < image_size:
                    dist = np.sqrt(dx ** 2 + dy ** 2) / max(w, h) * 2
                    intensity = max(0, 1 - dist)
                    img[ny, nx] = np.clip(
                        img[ny, nx] + np.array(color) * intensity, 0, 255
                    ).astype(np.uint8)
    return img


def main():
    np.random.seed(42)

    # Two fixed debris objects with constant velocity (plus small random drift)
    objects = [
        {"x": 100.0, "y": 100.0, "vx": 6.0, "vy": 4.0, "w": 24, "h": 24, "cls": 1},
        {"x": 300.0, "y": 300.0, "vx": -5.0, "vy": 5.0, "w": 32, "h": 32, "cls": 2},
    ]

    print(f"Loading trained model from {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    tracker = DeepSort()

    print(f"\nRunning {NUM_FRAMES}-frame tracking demo...\n")
    print(f"{'Frame':<7} Confirmed Tracks")
    print("-" * 60)

    for frame_idx in range(NUM_FRAMES):
        frame = make_frame(objects)
        cv2.imwrite(f"{OUTPUT_DIR}/frame_{frame_idx:02d}.png", frame)

        results = model.predict(frame, conf=0.3, verbose=False)[0]
        raw_detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            raw_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, CLASS_NAMES[cls_id]))

        tracks = tracker.update_tracks(raw_detections, frame=frame)

        line = f"Frame {frame_idx}: "
        confirmed_count = 0
        for t in tracks:
            if not t.is_confirmed():
                continue
            confirmed_count += 1
            l, tp, r, b = t.to_ltrb()
            cx, cy = (l + r) / 2, (tp + b) / 2
            line += f"[ID {t.track_id} | {t.det_class} @ ({cx:.0f},{cy:.0f})]  "
        if confirmed_count == 0:
            line += "(none confirmed yet)"
        print(line)

        # Move objects for the next frame (constant velocity + small noise)
        for obj in objects:
            obj["x"] += obj["vx"] + np.random.uniform(-1, 1)
            obj["y"] += obj["vy"] + np.random.uniform(-1, 1)

    print("\nDone. Frames saved to:", OUTPUT_DIR)
    print("If the SAME Track ID number appears across multiple frames for")
    print("the same object, DeepSORT is correctly maintaining identity.")


if __name__ == "__main__":
    main()
