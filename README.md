# Neural Intent Decoder for Lane Change Prediction

## Overview
This project builds an artificial neural network (ANN) to predict whether a vehicle will perform a lane change within the next 3 seconds using trajectory data.

The goal is to model **driver behavior**, not just vehicle physics.

---

## Problem Statement
Given recent motion data of a vehicle (position, velocity, acceleration, lane), predict:

P(lane change within next 3 seconds)

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
- Architecture: 75 → 64 → 32 → 1
- ReLU activation for hidden layers
- Sigmoid output for probability

### Training Improvements
- Addressed class imbalance using weighted loss (BCEWithLogitsLoss)
- Improved recall and F1 score significantly
- Tuned decision threshold for better precision-recall balance

---

## Results

### Best Performance (after tuning)
- Precision: ~0.31
- Recall: ~0.76
- F1 Score: ~0.44

### Key Insight
- Accuracy alone was misleading due to class imbalance
- Weighted loss improved detection of lane changes
- Model now balances detection vs false positives effectively

---

## Project Structure

data/
  raw/        # original dataset (not tracked)
  processed/  # processed dataset (not tracked)

notebooks/
  01_data_exploration.ipynb
  02_preprocessing.ipynb
  03_model_training.ipynb

src/          # reusable code (future work)
figures/      # plots and visuals

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
pip install -r requirements.txt

2. Run notebooks in order:
- 01_data_exploration.ipynb
- 02_preprocessing.ipynb
- 03_model_training.ipynb

Note: Raw dataset is not included in this repository.

---

## Status
🚧 In Progress  
- Full pipeline implemented  
- Model trained and evaluated  
- Evaluation metrics and visualization added  

---

## Future Work
- Add Precision-Recall curve visualization
- Experiment with additional features
- Try alternative architectures
- Refactor code into reusable modules
