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

---

## Current Progress

- Completed full data preprocessing pipeline
- Built labeled dataset using sliding window approach
- Cleaned data to remove noise and formatting issues
- Trained initial ANN model on subset of data
- Achieved ~89% accuracy (above baseline)

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
- Data pipeline complete  
- Initial model trained and evaluated  
- Ongoing improvements and scaling  

---

## Future Work
- Improve model evaluation (precision, recall, F1, ROC-AUC)
- Train on larger dataset
- Experiment with different architectures
- Move pipeline into reusable src/ modules
- Add visualization of predictions
