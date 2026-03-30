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
- Velocity
- Acceleration
- Lane ID

---

## Approach
- Construct sliding windows of 3 seconds (15 timesteps)
- Flatten into a 75-dimensional input vector
- Train a feedforward neural network
- Output probability of lane change

---

## Project Structure

data/
  raw/        # original dataset (not tracked)
  processed/  # cleaned dataset

notebooks/    # experiments and exploration
src/          # reusable code
figures/      # plots and visuals

---

## Tech Stack
- Python
- PyTorch
- Pandas / NumPy
- Matplotlib / Seaborn
- Scikit-learn

---

## Status
🚧 In Progress  
- Project setup complete  
- Beginning data exploration  

---

## Future Work
- Data preprocessing pipeline
- Model training and evaluation
- Baseline comparison
- Visualization of predictions