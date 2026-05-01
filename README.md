# Neural Intent Decoder for Lane Change Prediction

## Overview

This project builds an artificial neural network (ANN) to predict whether a vehicle will perform a lane change within the next 3 seconds using trajectory data.
The goal is to model **driver behavior**, not just vehicle physics.

---

## Problem Statement
Given recent motion data of a vehicle (position, velocity, acceleration, lane), predict:

**P(lane change within next 3 seconds)**

---

## Dataset

This project uses the **NGSIM I-80 vehicle trajectory dataset**, which contains real-world highway traffic data collected at 10 Hz.

### Features Used:

- Local X (lateral position)
- Local Y (longitudinal position)
- Velocity (v_Vel)
- Acceleration (v_Acc)
- Lane ID

---

## Approach

### Data Pipeline

- Filtered and cleaned trajectory data  
- Removed noise in lane-change signals using temporal consistency  
- Downsampled data (10 Hz → 5 Hz)  
- Constructed sliding windows (15 timesteps = 3 seconds)  
- Flattened each window into a 75-dimensional input vector  
- Generated labels based on future lane changes (within 3 seconds)  

### Model

- Feedforward neural network (PyTorch)  
- Architecture: **75 → 64 → 32 → 1**  
- ReLU activation for hidden layers  
- Uses logits (no sigmoid in model; handled by loss function)  

### Training Improvements

- Addressed class imbalance using weighted loss (`BCEWithLogitsLoss` with `pos_weight`)  
- Tuned decision threshold to balance precision and recall  
- Evaluated using precision, recall, and F1 score instead of accuracy  

---

## Results

### Best Performance

- **Precision:** ~0.31  
- **Recall:** ~0.76  
- **F1 Score:** ~0.44  

### Key Insights

- Accuracy alone was misleading due to class imbalance  
- Weighted loss significantly improved detection of lane changes  
- Threshold tuning increased recall while managing false positives  

---

## Confusion Matrix

![Confusion Matrix](figures/confusion_matrix.png)

- **True Positives:** Correct lane change detections  
- **False Positives:** False alarms  
- **False Negatives:** Missed lane changes  

---

## Precision-Recall Curve

![Precision-Recall Curve](figures/precision_recall_curve.png)

This curve shows the tradeoff between precision and recall across different thresholds.

---

## Project Structure

```
data/
  raw/        # original dataset (not tracked)
  processed/  # processed dataset (not tracked)
notebooks/
  01_data_exploration.ipynb
  02_preprocessing.ipynb
  03_model_training.ipynb
src/          # reusable code (future work)
figures/      # plots and visuals
```

---

## Tech Stack

- Python  
- PyTorch  
- Pandas / NumPy  
- Matplotlib / Seaborn  
- Scikit-learn  

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run notebooks in order:

- `01_data_exploration.ipynb`  
- `02_preprocessing.ipynb`  
- `03_model_training.ipynb`  

> Note: Raw dataset is not included in this repository.

---

## Status

✅ **Completed (v1)**  

- Full pipeline implemented  
- Model trained and evaluated  
- Class imbalance handled  
- Metrics and visualizations added  

---

## Future Work

- Add baseline model comparison  
- Engineer additional features (e.g., relative motion)  
- Try alternative architectures  
- Refactor into modular `src/` pipeline