"""
gpu_check.py — Verify your GPU is ready for training
Run this before starting: python gpu_check.py
"""

import subprocess
import sys

def check_python():
    print(f"✅ Python: {sys.version.split()[0]}")

def check_torch():
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            cuda_ver = torch.version.cuda
            print(f"✅ GPU: {gpu}")
            print(f"✅ VRAM: {vram} GB")
            print(f"✅ CUDA: {cuda_ver}")
            
            # Quick tensor test
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print(f"✅ GPU tensor ops: Working")
            
            # Recommend batch size based on VRAM
            if vram >= 16:
                print(f"\n💡 Recommended batch size: 32 (high VRAM)")
            elif vram >= 8:
                print(f"\n💡 Recommended batch size: 16 (standard)")
            elif vram >= 6:
                print(f"\n💡 RTX 3060 6GB detected!")
                print(f"   Recommended settings:")
                print(f"   • Model:      YOLOv8s (small)")
                print(f"   • Batch size: 8")
                print(f"   • AMP:        True (saves ~40% VRAM)")
                print(f"   • Est. training time: ~45 min / 100 epochs")
            else:
                print(f"\n💡 Recommended batch size: 8 (low VRAM - adjust in config.yaml)")
        else:
            print("⚠️  CUDA not available — will train on CPU (much slower)")
            print("   Fix: Install CUDA toolkit from https://developer.nvidia.com/cuda-downloads")
            print("   Then reinstall PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    except ImportError:
        print("❌ PyTorch not installed. Run: bash setup.sh")

def check_ultralytics():
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics (YOLOv8): Installed")
    except ImportError:
        print("❌ Ultralytics missing. Run: pip install ultralytics")

def check_sgp4():
    try:
        from sgp4.api import Satrec
        print("✅ sgp4 (orbital mechanics): Installed")
    except ImportError:
        print("❌ sgp4 missing. Run: pip install sgp4")

def check_streamlit():
    try:
        import streamlit
        print(f"✅ Streamlit: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit missing. Run: pip install streamlit")

def check_deepsort():
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
        print("✅ DeepSORT: Installed")
    except ImportError:
        print("⚠️  DeepSORT not installed. Run: pip install deep-sort-realtime")

if __name__ == "__main__":
    print("=" * 50)
    print("🛸 SPACE DEBRIS PROJECT — GPU & ENV CHECK")
    print("=" * 50)
    check_python()
    check_torch()
    check_ultralytics()
    check_sgp4()
    check_streamlit()
    check_deepsort()
    print("=" * 50)
    print("✅ Run check complete! Fix any ❌ items above.")
    print("   Then start with: python main.py --phase collect")
    print("=" * 50)
