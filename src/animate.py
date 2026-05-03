#!/usr/bin/env python3
"""
animate.py

Generates figures/ann_demo.gif: a top-down animation of a real NGSIM vehicle
trajectory with the ANN predicting lane changes in real time.

Prerequisites: run  python src/train.py  first to generate models/ann.pt
Run:           python src/animate.py
Output:        figures/ann_demo.gif
"""

import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model   import LaneChangeModel
from predict import load_model, load_norm_params

DATA_CSV = ROOT / "data" / "raw" / "i80_trajectories_NGSIM.csv"
OUT_GIF  = ROOT / "figures" / "ann_demo.gif"

WINDOW   = 15
FEATURES = ["Local_X", "Local_Y", "v_Vel", "v_Acc", "Lane_ID"]
FPS      = 10
THRESHOLD = 0.50   # visual threshold for the animation

LANE_WIDTH = 12.0  # approximate feet per lane in NGSIM I-80
N_LANES    = 6

# ── colours ───────────────────────────────────────────────────────────────────
BG      = "#0F172A"
ROAD_BG = "#1E293B"
ROAD_MK = "#334155"
WHITE   = "#FFFFFF"
SLATE   = "#94A3B8"
ORANGE  = "#F97316"
RED     = "#EF4444"
GREEN   = "#22C55E"
SKY     = "#38BDF8"
YELLOW  = "#FBBF24"


def prob_color(p):
    if p >= THRESHOLD:
        return RED
    elif p >= 0.35:
        return ORANGE
    return GREEN


# ════════════════════════════════════════════════════════════════════════════
# Step 1 — Load and clean raw trajectory data
# ════════════════════════════════════════════════════════════════════════════
print("Loading raw data...")
df = pd.read_csv(DATA_CSV, low_memory=False)

for col in FEATURES:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", ""), errors="coerce")

df = df.dropna(subset=FEATURES + ["Vehicle_ID", "Frame_ID"])
df["Frame_ID"]   = df["Frame_ID"].astype(int)
df["Vehicle_ID"] = df["Vehicle_ID"].astype(int)
df["Lane_ID"]    = df["Lane_ID"].astype(int)

# downsample 10 Hz → 5 Hz
df = df[df["Frame_ID"] % 2 == 0].copy()


# ════════════════════════════════════════════════════════════════════════════
# Step 2 — Load saved model
# ════════════════════════════════════════════════════════════════════════════
print("Loading model...")
model     = load_model()
mean, std = load_norm_params()


# ════════════════════════════════════════════════════════════════════════════
# Step 3 — Auto-select the best vehicle for animation
# ════════════════════════════════════════════════════════════════════════════
print("Searching for best vehicle...")

def evaluate_vehicle(vdf):
    """
    Returns (peak_prob, change_idx, trajectory_df) for a vehicle that makes
    a clear stable lane change and the model predicts with high confidence.
    Returns None if the vehicle is not suitable.
    """
    vdf = vdf.sort_values("Frame_ID").reset_index(drop=True)
    if len(vdf) < WINDOW + 25:
        return None

    lanes = vdf["Lane_ID"].values

    # find first stable lane change (lane stays different for 3+ frames)
    change_idx = None
    for i in range(WINDOW, len(lanes) - 5):
        if lanes[i] != lanes[WINDOW - 1] and np.all(lanes[i:i+5] == lanes[i]):
            change_idx = i
            break
    if change_idx is None:
        return None

    # need enough frames after the change for the animation to complete
    if change_idx + 20 > len(vdf):
        return None

    # run inference on the 10 frames leading up to the change
    feats = vdf[FEATURES].values.astype(np.float32)
    peak_p = 0.0
    window_start = max(WINDOW, change_idx - 10)
    for i in range(window_start, change_idx + 1):
        w = feats[i - WINDOW:i].flatten()
        w_norm = (w - mean) / std
        x = torch.tensor(w_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p = torch.sigmoid(model(x)).item()
        peak_p = max(peak_p, p)

    return (peak_p, change_idx, vdf)


best_score = 0.0
best_result = None
checked = 0

for vid, vdf in df.groupby("Vehicle_ID"):
    result = evaluate_vehicle(vdf)
    checked += 1
    if result and result[0] > best_score:
        best_score  = result[0]
        best_result = result
        print(f"  Candidate: vehicle {vid:5d}  "
              f"peak_p={best_score:.3f}  change@idx={result[1]}")
    if best_score >= 0.92 or checked >= 400:
        break

if best_result is None:
    raise RuntimeError(
        "No suitable vehicle found. Make sure models/ann.pt exists "
        "and data/raw/i80_trajectories_NGSIM.csv is present.")

peak_p, change_idx, traj = best_result
print(f"\nSelected vehicle: peak_p={peak_p:.3f}, change at frame index {change_idx}")


# ════════════════════════════════════════════════════════════════════════════
# Step 4 — Pre-compute probability for every frame
# ════════════════════════════════════════════════════════════════════════════
feats   = traj[FEATURES].values.astype(np.float32)
lanes   = traj["Lane_ID"].values
local_x = traj["Local_X"].values.astype(float)  # lateral (changes at lane change)
local_y = traj["Local_Y"].values.astype(float)  # longitudinal (car moves forward)
vels    = traj["v_Vel"].values.astype(float)
accs    = traj["v_Acc"].values.astype(float)

n_total = len(traj)
probs   = np.zeros(n_total)

for i in range(WINDOW, n_total):
    w      = feats[i - WINDOW:i].flatten()
    w_norm = (w - mean) / std
    x      = torch.tensor(w_norm, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs[i] = torch.sigmoid(model(x)).item()

# animation window: 10 frames before first full window up to 25 frames past change
anim_start  = max(0, WINDOW - 10)
anim_end    = min(n_total, change_idx + 25)
anim_frames = list(range(anim_start, anim_end))
n_anim      = len(anim_frames)

print(f"Animating {n_anim} frames  ({n_anim / FPS:.1f} seconds at {FPS} fps)")

# position of the actual lane change for the vertical marker
change_ly = local_y[change_idx]


# ════════════════════════════════════════════════════════════════════════════
# Step 5 — Build the figure
# ════════════════════════════════════════════════════════════════════════════
ROAD_Y_MIN = 0.0
ROAD_Y_MAX = N_LANES * LANE_WIDTH
VIEW_W     = 280.0   # feet of road visible at once
CAR_LEN    = 18.0    # feet (longitudinal)
CAR_WID    =  7.0    # feet (lateral)

fig = plt.figure(figsize=(14, 7), facecolor=BG)
fig.patch.set_facecolor(BG)

gs = fig.add_gridspec(
    2, 2,
    width_ratios=[3.2, 1],
    height_ratios=[5.5, 1],
    hspace=0.06, wspace=0.04,
    left=0.03, right=0.97,
    top=0.91, bottom=0.06)

ax_hw  = fig.add_subplot(gs[0, 0])   # highway view
ax_bar = fig.add_subplot(gs[0, 1])   # probability bar
ax_ft  = fig.add_subplot(gs[1, :])   # feature strip

for ax in [ax_hw, ax_bar, ax_ft]:
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(ROAD_MK)
        sp.set_linewidth(0.8)

fig.suptitle("NGSIM I-80  —  ANN Lane Change Prediction",
             color=WHITE, fontsize=13, fontweight="bold", y=0.97)

# ── Highway panel ─────────────────────────────────────────────────────────
ax_hw.set_ylim(ROAD_Y_MIN - 3, ROAD_Y_MAX + 3)
ax_hw.set_ylabel("Lateral Position (ft)", color=SLATE, fontsize=9)
ax_hw.set_xlabel("Longitudinal Position (ft)", color=SLATE, fontsize=9)
ax_hw.tick_params(colors=SLATE, labelsize=8)

# Road surface
road_bg = plt.Rectangle((0, ROAD_Y_MIN), VIEW_W, ROAD_Y_MAX,
                          facecolor=ROAD_BG, edgecolor="none", zorder=1)
ax_hw.add_patch(road_bg)

# Lane divider lines (stored for update)
lane_line_objs = []
for ln in range(1, N_LANES):
    lx_pos = ln * LANE_WIDTH
    line_obj, = ax_hw.plot([], [], color=WHITE, linewidth=0.9,
                            linestyle="--", alpha=0.35, zorder=2)
    lane_line_objs.append((lx_pos, line_obj))

# Shoulder lines (solid)
shoulder_lo, = ax_hw.plot([], [], color=WHITE, linewidth=1.5, alpha=0.6, zorder=2)
shoulder_hi, = ax_hw.plot([], [], color=WHITE, linewidth=1.5, alpha=0.6, zorder=2)

# Lane labels (static text, updated by xlim)
lane_label_texts = []
for ln in range(N_LANES):
    t = ax_hw.text(0, (ln + 0.5) * LANE_WIDTH, f"Lane {ln+1}",
                   color=SLATE, fontsize=7, alpha=0.5,
                   va="center", ha="left", zorder=3)
    lane_label_texts.append(t)

# Trailing path
trail_line, = ax_hw.plot([], [], color=SKY, linewidth=1.5,
                          alpha=0.35, zorder=3, solid_capstyle="round")

# Actual lane-change marker
lc_line = ax_hw.axvline(x=change_ly, color=ORANGE, linewidth=1.2,
                         linestyle=":", alpha=0.0, zorder=4)
lc_label = ax_hw.text(change_ly, ROAD_Y_MAX + 1.5, "Actual\nChange",
                       color=ORANGE, fontsize=7, ha="center", alpha=0.0, zorder=4)

# Car rectangle
car_patch = plt.Rectangle(
    (0, 0), CAR_LEN, CAR_WID,
    facecolor=YELLOW, edgecolor=GREEN,
    linewidth=2.5, zorder=5)
ax_hw.add_patch(car_patch)

# Prediction banner (hidden until threshold crossed)
pred_box = ax_hw.text(
    0.5, 0.88, "⚠  LANE CHANGE PREDICTED",
    transform=ax_hw.transAxes,
    fontsize=13, fontweight="bold", color=RED,
    ha="center", va="top",
    bbox=dict(boxstyle="round,pad=0.35",
              facecolor="#450A0A", edgecolor=RED,
              linewidth=1.8, alpha=0.92),
    visible=False, zorder=10)

# Time stamp
time_text = ax_hw.text(0.02, 0.04, "", transform=ax_hw.transAxes,
                        fontsize=8, color=SLATE, zorder=6)

# ── Probability bar panel ─────────────────────────────────────────────────
ax_bar.set_xlim(0, 1)
ax_bar.set_ylim(0, 1)
ax_bar.set_xticks([])
ax_bar.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_bar.tick_params(right=True, left=False, labelright=True, labelleft=False,
                   colors=SLATE, labelsize=8)
ax_bar.set_ylabel("P(Lane Change)", color=SLATE, fontsize=9, labelpad=6)
ax_bar.yaxis.set_label_position("right")
ax_bar.yaxis.label.set_color(SLATE)

# Background trough
ax_bar.bar([0.5], [1.0], width=0.7, color=ROAD_BG,
           edgecolor=ROAD_MK, linewidth=0.8)

# Threshold line
ax_bar.axhline(y=THRESHOLD, color=ORANGE, linewidth=1.5,
               linestyle="--", alpha=0.85, zorder=3)
ax_bar.text(0.5, THRESHOLD + 0.025, "threshold",
            transform=ax_bar.get_yaxis_transform(),
            ha="center", va="bottom", color=ORANGE, fontsize=7)

# Live bar
prob_bar_patch = ax_bar.bar([0.5], [0.0], width=0.7,
                             color=GREEN, zorder=4)[0]

# Numeric label
prob_num = ax_bar.text(0.5, 0.03, "p = 0.00",
                        ha="center", va="bottom",
                        fontsize=12, fontweight="bold",
                        color=WHITE, zorder=5)

ax_bar.set_title("Confidence", color=SLATE, fontsize=9, pad=4)

# ── Feature strip ─────────────────────────────────────────────────────────
ax_ft.set_xlim(0, 1); ax_ft.set_ylim(0, 1)
ax_ft.set_xticks([]); ax_ft.set_yticks([])
ax_ft.set_facecolor(ROAD_BG)
feat_text = ax_ft.text(0.5, 0.5, "",
                        ha="center", va="center",
                        fontsize=11, color=WHITE,
                        fontfamily="monospace")


# ════════════════════════════════════════════════════════════════════════════
# Step 6 — Animation update function
# ════════════════════════════════════════════════════════════════════════════
def update(fi):
    idx = anim_frames[fi]

    ly  = local_y[idx]   # longitudinal (horizontal axis)
    lx  = local_x[idx]   # lateral (vertical axis)
    p   = probs[idx]
    v   = vels[idx]
    a   = accs[idx]
    ln  = lanes[idx]
    col = prob_color(p)

    # ── scrolling highway window ──────────────────────────────────────────
    win_lo = ly - VIEW_W * 0.35
    win_hi = ly + VIEW_W * 0.65
    ax_hw.set_xlim(win_lo, win_hi)

    road_bg.set_xy((win_lo, ROAD_Y_MIN))
    road_bg.set_width(VIEW_W)

    # lane dividers
    for lx_pos, lo in lane_line_objs:
        lo.set_data([win_lo, win_hi], [lx_pos, lx_pos])

    shoulder_lo.set_data([win_lo, win_hi], [ROAD_Y_MIN, ROAD_Y_MIN])
    shoulder_hi.set_data([win_lo, win_hi], [ROAD_Y_MAX, ROAD_Y_MAX])

    for i, lt in enumerate(lane_label_texts):
        lt.set_x(win_lo + 8)

    # trailing path (last 25 frames)
    ts = max(0, idx - 25)
    trail_line.set_data(local_y[ts:idx+1], local_x[ts:idx+1])

    # actual change marker (fade in as we approach)
    dist = abs(ly - change_ly)
    marker_alpha = min(0.7, max(0.0, (200 - dist) / 200))
    lc_line.set_alpha(marker_alpha)
    lc_label.set_alpha(marker_alpha)
    lc_label.set_x(change_ly)

    # car
    car_patch.set_xy((ly - CAR_LEN * 0.5, lx - CAR_WID * 0.5))
    car_patch.set_edgecolor(col)
    car_patch.set_linewidth(3.2 if p >= THRESHOLD else 2.0)

    # prediction banner
    pred_box.set_visible(p >= THRESHOLD)

    time_text.set_text(f"t = {idx * 0.2:.1f} s")

    # ── probability bar ───────────────────────────────────────────────────
    prob_bar_patch.set_height(p)
    prob_bar_patch.set_color(col)
    prob_num.set_text(f"p = {p:.2f}")
    prob_num.set_color(col)
    prob_num.set_y(min(p + 0.02, 0.92))

    # ── feature strip ─────────────────────────────────────────────────────
    feat_text.set_text(
        f"Velocity: {v:6.1f} ft/s   |   "
        f"Acceleration: {a:+.2f} ft/s²   |   "
        f"Lane: {ln}")


# ════════════════════════════════════════════════════════════════════════════
# Step 7 — Render and save
# ════════════════════════════════════════════════════════════════════════════
print(f"Rendering {n_anim} frames...")

anim = FuncAnimation(fig, update, frames=n_anim,
                     interval=1000 // FPS, blit=False)

anim.save(str(OUT_GIF), writer="pillow", fps=FPS, dpi=100)
plt.close(fig)

size_kb = OUT_GIF.stat().st_size / 1024
print(f"Saved  -> {OUT_GIF}")
print(f"Size   -> {size_kb:.0f} KB")
print(f"Length -> {n_anim / FPS:.1f} seconds")
