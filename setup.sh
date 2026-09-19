#!/bin/bash
# ============================================================
# Space Debris Project — Local GPU Setup Script
# Run this ONCE to set up your full environment
# Usage: bash setup.sh
# ============================================================

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║        🛸 SPACE DEBRIS PROJECT — ENVIRONMENT SETUP      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Check Python ──────────────────────────────────────
echo "▶ Checking Python version..."
python3 --version
echo ""

# ── Step 2: Create Virtual Environment ───────────────────────
echo "▶ Creating virtual environment (space_debris_env)..."
python3 -m venv space_debris_env
source space_debris_env/bin/activate
echo "✅ Virtual environment created and activated"
echo ""

# ── Step 3: Upgrade pip ───────────────────────────────────────
echo "▶ Upgrading pip..."
pip install --upgrade pip
echo ""

# ── Step 4: Install PyTorch with CUDA ────────────────────────
echo "▶ Installing PyTorch with CUDA support..."
echo "   (Installs CUDA 11.8 compatible version)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo ""

# ── Step 5: Install Project Dependencies ─────────────────────
echo "▶ Installing project dependencies..."
pip install \
    ultralytics \
    deep-sort-realtime \
    sgp4 \
    astropy \
    numpy pandas scipy \
    opencv-python \
    Pillow \
    albumentations \
    plotly matplotlib \
    streamlit \
    requests tqdm \
    PyYAML python-dotenv \
    scikit-learn \
    joblib \
    jupyter ipykernel
echo ""

# ── Step 6: Verify GPU ────────────────────────────────────────
echo "▶ Verifying GPU setup..."
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
else:
    print('⚠️  No GPU detected. Check your CUDA/driver installation.')
"
echo ""

# ── Step 7: Create folder structure ──────────────────────────
echo "▶ Setting up project directories..."
mkdir -p data/{raw,processed,synthetic,annotations}
mkdir -p models/{detection,tracking,prediction}
mkdir -p outputs logs
echo "✅ Directories created"
echo ""

# ── Done ──────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                  ✅ SETUP COMPLETE!                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Activate environment:  source space_debris_env/bin/activate"
echo "  2. Collect data:          python main.py --phase collect"
echo "  3. Preprocess data:       python main.py --phase preprocess"
echo "  4. Train detector:        python main.py --phase train"
echo "  5. Launch dashboard:      python main.py --phase dashboard"
echo ""
echo "💡 Tip: Always activate the venv before running:"
echo "   source space_debris_env/bin/activate"
echo ""
