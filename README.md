# 🛸 Space Debris Detection & Tracking System

A full end-to-end ML pipeline for detecting, tracking, and predicting space debris trajectories using YOLOv8, DeepSORT, and LSTM.

## Project Structure
```
space_debris/
├── data/
│   ├── raw/              # Raw downloaded data (TLE, radar)
│   ├── processed/        # Cleaned and parsed data
│   ├── synthetic/        # Simulated debris data
│   └── annotations/      # Labeled bounding boxes
├── models/
│   ├── detection/        # YOLOv8 weights
│   ├── tracking/         # DeepSORT config
│   └── prediction/       # LSTM trajectory model
├── src/
│   ├── data/             # Data collection & preprocessing scripts
│   ├── models/           # Model definitions
│   ├── utils/            # Helper functions
│   └── visualization/    # Dashboard & plotting
├── notebooks/            # Jupyter notebooks for each phase
├── outputs/              # Results, plots, exports
├── configs/              # Config files
└── logs/                 # Training logs
```

## Phases
- [x] Phase 1: Project Setup
- [ ] Phase 2: Data Collection
- [ ] Phase 3: Data Preprocessing
- [ ] Phase 4: Detection Model (YOLOv8)
- [ ] Phase 5: Tracking (DeepSORT)
- [ ] Phase 6: Trajectory Prediction (LSTM)
- [ ] Phase 7: Collision Risk Estimation
- [ ] Phase 8: Visualization Dashboard
- [ ] Phase 9: Optimization & Deployment

## Tech Stack
- Python 3.10+
- YOLOv8 (Ultralytics)
- DeepSORT
- PyTorch
- sgp4
- Streamlit
- Plotly
```
