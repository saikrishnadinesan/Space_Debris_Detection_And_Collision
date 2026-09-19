"""
main.py — Space Debris Detection & Tracking Pipeline
=====================================================
Master script that runs the full pipeline from data collection
to collision risk assessment.

Usage:
    python main.py --phase all          # Run full pipeline
    python main.py --phase collect      # Data collection only
    python main.py --phase preprocess   # Preprocessing only
    python main.py --phase train        # Model training only
    python main.py --phase risk         # Risk assessment only
    python main.py --phase dashboard    # Launch dashboard
"""

import argparse
import subprocess
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║          🛸 SPACE DEBRIS DETECTION SYSTEM 🛸            ║
║     Detection | Tracking | Prediction | Risk Assessment  ║
╚══════════════════════════════════════════════════════════╝
    """)


def run_phase(phase_name: str, script_path: str):
    """Run a pipeline phase."""
    print(f"\n{'='*60}")
    print(f"▶ Running Phase: {phase_name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"❌ Phase '{phase_name}' failed with exit code {result.returncode}")
        return False
    print(f"✅ Phase '{phase_name}' completed successfully!")
    return True


def phase_collect():
    return run_phase("Data Collection", "src/data/collect_data.py")


def phase_preprocess():
    return run_phase("Data Preprocessing", "src/data/preprocess.py")


def phase_train_detection():
    return run_phase("Detection Model Training", "src/models/train_detection.py")


def phase_train_prediction():
    return run_phase("Trajectory Prediction Training", "src/models/train_prediction.py")


def phase_risk():
    return run_phase("Collision Risk Assessment", "src/models/collision_risk.py")


def phase_dashboard():
    print("\n🚀 Launching Dashboard...")
    print("   Open browser at: http://localhost:8501")
    subprocess.run(["streamlit", "run", "src/visualization/dashboard.py"])


def run_full_pipeline():
    """Run all phases in sequence."""
    phases = [
        ("Data Collection", phase_collect),
        ("Preprocessing", phase_preprocess),
        ("Detection Training", phase_train_detection),
        ("Trajectory Training", phase_train_prediction),
        ("Risk Assessment", phase_risk),
    ]

    print("\n🚀 Running full pipeline...")
    print(f"   Total phases: {len(phases)}\n")

    for name, fn in phases:
        success = fn()
        if not success:
            print(f"\n⚠️  Pipeline stopped at: {name}")
            print("   Fix the error above and re-run from that phase.")
            return

    print("\n" + "="*60)
    print("🎉 FULL PIPELINE COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  • Launch dashboard: python main.py --phase dashboard")
    print("  • View outputs in: outputs/")
    print("  • Model weights in: models/")


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="Space Debris Detection Pipeline")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "collect", "preprocess", "train",
                                 "predict", "risk", "dashboard"],
                        help="Pipeline phase to run")
    args = parser.parse_args()

    phase_map = {
        "all":        run_full_pipeline,
        "collect":    phase_collect,
        "preprocess": phase_preprocess,
        "train":      phase_train_detection,
        "predict":    phase_train_prediction,
        "risk":       phase_risk,
        "dashboard":  phase_dashboard,
    }

    phase_map[args.phase]()


if __name__ == "__main__":
    main()
