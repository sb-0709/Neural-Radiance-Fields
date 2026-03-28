# HW4 — Neural Radiance Fields
### COMS 4732: Computer Vision II

> Fitting a 2D neural field to a single image, then scaling up to a full 3D Neural Radiance Field (NeRF) trained from multi-view photographs.

---

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Part 1 — 2D Neural Field](#part-1--2d-neural-field)
- [Part 2 — 3D NeRF on Lego Scene](#part-2--3d-nerf-on-lego-scene)
- [Part 3 — Real-World Data (Extra Credit)](#part-3--real-world-data-extra-credit)
- [Results Summary](#results-summary)

---

## Overview

This homework builds a Neural Radiance Field (NeRF) from scratch in PyTorch.

- **Part 1** trains a small MLP to memorize a single 2D image by mapping pixel coordinates `(u, v) → (r, g, b)` using sinusoidal positional encoding.
- **Part 2** extends this to 3D: an 8-layer MLP maps `(x, y, z, dx, dy, dz) → (r, g, b, σ)` and volume rendering composites predictions into pixel colors, trained on 100 multi-view images of the Lego bulldozer scene.
- **Part 3** captures a real object with a smartphone, runs COLMAP to estimate camera poses, and trains the same NeRF pipeline on the resulting dataset.

---

## Setup

```bash
# Clone and enter project
git clone <your-repo-url>
cd cv2_hw4

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows

# Install dependencies
pip install torch torchvision numpy matplotlib Pillow requests imageio viser tyro
```

> **Apple Silicon (M-series Mac):** MPS backend is auto-detected. No extra setup needed.  
> **Google Colab:** Change `device = get_device()` to `device = "cuda"` at the top of `part2_nerf3d.py`.

### Download the Lego dataset

```bash
wget https://coms4732.github.io/hws/hw4/assets/lego_200x200.npz
```

---

## Project Structure

```
cv2_hw4/
│
├── part1_nerf2d.py          # Part 1: 2D Neural Field — training script
├── part2_nerf3d.py          # Part 2 & 3: 3D NeRF — training script
├── dataset_3d.py            # Ray generation, RaysData dataset class
├── rendering.py             # sample_along_rays(), volume_render()
├── visualise_viser.py       # 3D visualization of cameras + rays + samples
├── vis_orbit.py             # Render orbit GIF for custom dataset (Part 3)
├── render_gif.py            # Render novel-view GIF from saved checkpoint
│
├── lego_200x200.npz         # Lego dataset (download separately)
├── my_object.npz            # custom COLMAP dataset (Part 3)
│
├── index.html               # Report webpage
│
├── nerf_outputs/            # Part 2 outputs (auto-created)
│   ├── nerf_model.pth
│   ├── nerf_progression.png
│   ├── val_psnr_curve.png
│   ├── train_loss_curve.png
│   └── lego_novel_view.gif
│
└── my_object_outputs/       # Part 3 outputs (auto-created)
    ├── nerf_model.pth
    ├── nerf_progression.png
    ├── val_psnr_curve.png
    ├── train_loss_curve.png
    └── orbit.gif
```

---

## Part 1 — 2D Neural Field

### Architecture

| Parameter | Value |
|---|---|
| Positional Encoding levels L | 10 |
| Input dim after PE | 42 &nbsp;`(2 + 2×2×10)` |
| Hidden layers | 4 |
| Hidden width | 256 |
| Output activation | Sigmoid → RGB ∈ [0, 1] |
| Loss | MSE |
| Optimizer | Adam, lr = 1e-2 |
| Batch size | 10,000 pixels / iter |
| Iterations | 2,000 |

### Run

```bash
# Place your images in the project folder:
#   fox.jpg      — the provided test image
#   my_image.jpg — any image of your choice

python part1_nerf2d.py
```

Outputs saved to current directory:
- `test_image_progression.png` — training snapshots at iter 1, 100, 250, 500, 1000, 2000
- `test_image_psnr.png` — PSNR curve over training
- `my_image_progression.png` — same for your own image
- `hyperparameter_grid.png` — 2×2 grid over L ∈ {4,10} × width ∈ {64,256}

### Key findings

- **L (PE frequency) matters more than width.** L=4 caps at ~25 dB regardless of width because the network lacks high-frequency basis functions. L=10 reaches ~28 dB.
- **Width adds capacity but not expressiveness.** Going from width=64 → 256 at L=10 gives ~2 dB gain. At L=4 the width barely matters.
- **Best config:** L=10, width=256 → **~28 dB PSNR** after 2000 iterations.

---

## Part 2 — 3D NeRF on Lego Scene

### Architecture

| Component | Details |
|---|---|
| Point encoding (L_pts) | L=10 → 63-dim |
| Direction encoding (L_dir) | L=4 → 27-dim |
| MLP depth | 8 hidden layers |
| Hidden width | 256 |
| Skip connection | Input re-injected at layer 5 |
| Density output | Linear(256→1) + ReLU ≥ 0 |
| Color output | Linear(256+27→128) + Linear(128→3) + Sigmoid |
| Loss | MSE on rendered pixel colors |
| Optimizer | Adam, lr = 5e-4 |
| Batch size | 10,000 rays / iter |
| Near / Far | 2.0 / 6.0 |
| Samples per ray | 64 |

### Run

```bash
# Train NeRF on Lego scene (5000 iterations, ~40-60 min on MPS)
python part2_nerf3d.py

# Visualize cameras + rays + samples in 3D
python visualise_viser.py
# → open http://localhost:8080 in your browser

# Render novel-view GIF from saved checkpoint (if needed separately)
python render_gif.py
```

### Implementation notes

**Ray generation** (`dataset_3d.py`):
- Pixel centers computed as `(u + 0.5, v + 0.5)` to avoid off-by-half error
- Ray direction: `r_d = normalize(R @ K⁻¹ @ [u, v, 1])`
- All rays for all 100 training images precomputed in `RaysData.__init__()`
- `float64` replaced with `float32` throughout for MPS compatibility

**Sampling** (`rendering.py`):
- Uniform sampling between `near` and `far` with `n_samples` bins
- During training: random jitter `t += rand() * bin_width` prevents overfitting to fixed grid
- During inference: no jitter (`perturb=False`)

**Volume rendering** (`rendering.py`):
- `alpha_i = 1 - exp(-sigma_i * delta_i)`
- `T_i = cumprod(1 - alpha_{j<i})`
- `C(r) = sum_i T_i * alpha_i * c_i`

### Results

| Metric | Value |
|---|---|
| 23 dB target crossed at | ~800 iterations |
| Final validation PSNR | **~26.5 dB** @ 5000 iters |

---

## Part 3 — Real-World Data (Extra Credit)

### Data capture pipeline

1. **Record video** — ~35 sec, walking a full circle around a textured object (plush toy), portrait 480×848 at 30fps
2. **Extract frames & run COLMAP** — using the [provided Colab notebook](https://colab.research.google.com/drive/14jl4Qu_nXLnBOGyaVoc3RL7pNEt1dUTt)
   - Extracts ~60 frames from the video
   - Runs COLMAP Structure-from-Motion to estimate camera poses
   - Undistorts images with `cv2.undistort()` (NeRF assumes pinhole camera)
   - Saves as `.npz` with `images_train`, `c2ws_train`, `focal`
3. **Auto-split** — `load_data()` detects custom format and automatically:
   - Uses last 5 images as validation
   - Generates 60 circular test cameras at average training camera radius

### Run

```bash
# Place your custom npz in the project folder as my_object.npz
# Then edit the data_path in part2_nerf3d.py __main__ block:
#   data_path = "my_object.npz"
#   save_dir  = "my_object_outputs"
#   near = 2.0, far = 6.0   (adjust for your scene)

python part2_nerf3d.py

# Render orbit GIF
python vis_orbit.py
```

### Tuning near/far for real scenes

Run a quick 100-iteration debug to find the right range before committing to 5000 iters:

```python
# in part2_nerf3d.py, temporarily set:
num_iters=100, val_interval=100, save_dir="debug_outputs"
```

| Output at iter 100 | Meaning | Fix |
|---|---|---|
| All black | Rays miss object entirely | Decrease `near` and `far` |
| Blurry colored blob | ✅ Correct range | Keep these values |
| All white / blown out | Camera inside the object | Increase `near` |

### Results

| Metric | Value |
|---|---|
| Dataset | Real smartphone capture (plush toy) |
| Near / Far used | 2.0 / 6.0 |
| Training iterations | 5,000 |
| Final validation PSNR | ~20 dB |

> Real-world PSNR is lower than synthetic due to lighting inconsistencies, motion blur, and imperfect COLMAP pose estimation.

---

## Results Summary

### Part 1 — 2D Neural Field

| Config | PSNR |
|---|---|
| L=4, width=64 | ~25 dB |
| L=4, width=256 | ~25 dB |
| L=10, width=64 | ~26 dB |
| **L=10, width=256** | **~28 dB** |

### Part 2 — Lego NeRF

| Checkpoint | Val PSNR |
|---|---|
| 500 iters | ~21.7 dB |
| 1000 iters | ~23.9 dB ✦ target met |
| 5000 iters | ~26.5 dB |

### Part 3 — Real Data

| Metric | Value |
|---|---|
| Final PSNR | ~20 dB |
| Novel view | Orbit GIF |

