"""
Phase 4: Detection Model — YOLOv8
===================================
Trains a YOLOv8 model to detect and classify space debris
from optical/radar imagery.

Classes:
  0 - small_debris     (< 10cm)
  1 - medium_debris    (10cm - 1m)
  2 - large_debris     (> 1m)
  3 - rocket_body
  4 - defunct_satellite
"""

import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import yaml

# ─── Config ──────────────────────────────────────────────────────────────────
DATASET_YAML = "data/synthetic/debris.yaml"
MODEL_SAVE_DIR = os.path.abspath("models/detection")   # absolute path - fixes the runs/detect/ bug
EPOCHS = 30                 # reduced for CPU-only training (was 100, tuned for GPU)
BATCH_SIZE = 8
IMAGE_SIZE = 416            # reduced from 640 for CPU speed
BASE_MODEL = "yolov8s.pt"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ─── Device Setup ────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)} ({vram}GB VRAM)")
        
        # RTX 3060 6GB memory optimizations
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True   # faster convolutions
        torch.backends.cuda.matmul.allow_tf32 = True  # faster matmul
        print(f"✅ Memory optimizations enabled for 6GB VRAM")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✅ Using Apple Silicon MPS")
    else:
        device = "cpu"
        print("⚠️  No GPU found. Using CPU (training will be slow)")
    return device


# ─── Train Model ─────────────────────────────────────────────────────────────
def train_detection_model():
    """Train YOLOv8 on the space debris dataset."""
    print("=" * 60)
    print("🤖 TRAINING YOLOV8 DETECTION MODEL")
    print("=" * 60)

    # Check dataset exists
    if not os.path.exists(DATASET_YAML):
        print("❌ Dataset not found! Run preprocess.py first.")
        return None

    # Load pretrained YOLOv8 model
    print(f"\n📦 Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    device = get_device()

    # Train
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        project=MODEL_SAVE_DIR,
        name="debris_detector",
        save=True,
        plots=True,
        verbose=True,
        # RTX 3060 6GB optimizations
        amp=True,             # automatic mixed precision (saves ~40% VRAM)
        cache=False,          # don't cache images (saves RAM/VRAM)
        workers=4,            # data loading workers
        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        flipud=0.5,
        fliplr=0.5,
        # Optimizer
        optimizer="Adam",
        lr0=0.001,
        lrf=0.01,
        # Early stopping
        patience=20,
    )

    print("\n✅ Training complete!")
    best_model_path = os.path.join(MODEL_SAVE_DIR, "debris_detector", "weights", "best.pt")
    print(f"📁 Best model saved to: {best_model_path}")
    return model, results


# ─── Evaluate Model ──────────────────────────────────────────────────────────
def evaluate_model(model_path: str):
    """Evaluate trained model on test set."""
    print("\n📊 Evaluating model...")
    model = YOLO(model_path)

    results = model.val(
        data=DATASET_YAML,
        split="test",
        imgsz=IMAGE_SIZE,
        conf=0.5,
        iou=0.45,
    )

    metrics = {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "Precision": results.box.mp,
        "Recall": results.box.mr,
    }

    print("\n📈 Evaluation Results:")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")
    print("-" * 40)

    return metrics


# ─── Run Inference ───────────────────────────────────────────────────────────
def run_inference(model_path: str, image_path: str, conf: float = 0.5):
    """Run detection on a single image."""
    model = YOLO(model_path)
    results = model(image_path, conf=conf)

    for result in results:
        boxes = result.boxes
        print(f"\n🔍 Detected {len(boxes)} objects:")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            class_names = ["small_debris", "medium_debris", "large_debris",
                          "rocket_body", "defunct_satellite"]
            print(f"   [{class_names[cls]}] conf={conf:.2f} bbox={[round(x,1) for x in xyxy]}")

        # Save annotated image
        annotated = result.plot()
        output_path = "outputs/detection_result.jpg"
        os.makedirs("outputs", exist_ok=True)
        plt.imsave(output_path, annotated)
        print(f"\n💾 Saved annotated image to {output_path}")

    return results


# ─── Export Model ────────────────────────────────────────────────────────────
def export_model(model_path: str, format: str = "onnx"):
    """Export model to ONNX or TensorRT for deployment."""
    print(f"\n📦 Exporting model to {format.upper()}...")
    model = YOLO(model_path)
    model.export(format=format, imgsz=IMAGE_SIZE)
    print(f"✅ Model exported to {format}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Train
    model, results = train_detection_model()

    if model:
        best_model = os.path.join(MODEL_SAVE_DIR, "debris_detector", "weights", "best.pt")

        # Evaluate
        if os.path.exists(best_model):
            metrics = evaluate_model(best_model)

            # Export
            export_model(best_model, format="onnx")
