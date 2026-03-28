"""
rendering.py — Part 2 implementation
======================================
This file implements:
  - sample_along_rays()   (Part 2.2)
  - volume_render()       (PROVIDED — already in starter code)

The volume rendering equation is given by:
    C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i
    T_i  = exp(-sum_{j<i} sigma_j * delta_j)
"""

import torch
import numpy as np


def get_device() -> str:
    """Return the best available device: cuda > mps (Apple Silicon) > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─────────────────────────────────────────────────────────────
# IMPL: sample_along_rays  (Part 2.2)
# ─────────────────────────────────────────────────────────────
def sample_along_rays(
    r_os: torch.Tensor,
    r_ds: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    perturb: bool = True,
):
    """
    Sample 3D points along each ray between near and far planes.

    For each ray defined by origin r_o and direction r_d, we create
    n_samples evenly-spaced sample depths t in [near, far], then
    compute the 3D positions: x = r_o + t * r_d

    During training (perturb=True) we add random jitter within each
    interval so that we don't always sample the exact same set of t
    values — this prevents overfitting to a fixed discrete set of
    3D locations.

    Args:
        r_os:     (N, 3) ray origins
        r_ds:     (N, 3) ray directions (unit vectors)
        near:     float, minimum depth along ray (e.g. 2.0 for lego)
        far:      float, maximum depth along ray (e.g. 6.0 for lego)
        n_samples: int, number of samples per ray (e.g. 32 or 64)
        perturb:  bool, if True add random jitter (use during training only)

    Returns:
        pts:      (N, n_samples, 3) 3D world coordinates of all samples
        t_vals:   (N, n_samples)    depth values for each sample

    Example for a single ray with n_samples=4, near=2, far=6:
        t_vals = [2.0, 3.0, 4.0, 5.0]  (before perturbation)
        pts    = r_o + t * r_d for each t
    """
    N = r_os.shape[0]
    device = r_os.device

    # ── 1. Create evenly-spaced t values in [near, far) ─────
    # We use n_samples bins; the bin boundaries are at:
    #   near, near + step, near + 2*step, ..., far
    # Each sample is placed at the *start* of its bin.
    # Using torch.linspace gives us n_samples values from near to far.
    t_vals = torch.linspace(near, far, n_samples, device=device)  # (n_samples,)

    # ── 2. Optional perturbation (stochastic sampling) ───────
    # Each sample t[i] is perturbed within its bin [t[i], t[i+1]).
    # The bin width (step size) is:
    #   t_width = (far - near) / n_samples
    # We add a uniform random offset in [0, t_width) to each t value.
    if perturb:
        t_width = (far - near) / n_samples  # scalar bin width
        # Random offsets: (N, n_samples) in [0, t_width)
        noise = torch.rand(N, n_samples, device=device) * t_width
        # Expand t_vals to (N, n_samples) and add noise
        t_vals = t_vals.unsqueeze(0).expand(N, -1) + noise  # (N, n_samples)
    else:
        # No perturbation: just expand to (N, n_samples)
        t_vals = t_vals.unsqueeze(0).expand(N, -1)  # (N, n_samples)

    # ── 3. Compute 3D sample positions ───────────────────────
    # x = r_o + t * r_d
    # r_os: (N, 3)      → expand to (N, 1, 3)
    # r_ds: (N, 3)      → expand to (N, 1, 3)
    # t_vals: (N, n_samples) → expand to (N, n_samples, 1)
    pts = (
        r_os.unsqueeze(1)                      # (N, 1, 3)
        + t_vals.unsqueeze(-1)                 # (N, n_samples, 1)
        * r_ds.unsqueeze(1)                    # (N, 1, 3)
    )  # → (N, n_samples, 3)

    return pts, t_vals


# ─────────────────────────────────────────────────────────────
# PROVIDED: volume_render  (already implemented in starter code)
# ─────────────────────────────────────────────────────────────
def volume_render(
    sigmas: torch.Tensor,
    colors: torch.Tensor,
    t_vals: torch.Tensor,
):
    """
    Discretized volume rendering equation.

    Implements:
        C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i
        T_i  = exp(-sum_{j=1}^{i-1} sigma_j * delta_j)

    Args:
        sigmas: (N, n_samples)    volume density at each sample
        colors: (N, n_samples, 3) RGB color at each sample
        t_vals: (N, n_samples)    depth values for each sample

    Returns:
        rgb_map: (N, 3)  final rendered color for each ray
        weights: (N, n_samples)  per-sample contribution weights (useful for visualization)
    """
    # ── Compute delta_i (distance between consecutive samples) ──
    # delta_i = t_{i+1} - t_i for i < n_samples-1
    # For the last sample we use a large value (effectively infinity)
    deltas = t_vals[:, 1:] - t_vals[:, :-1]                    # (N, n_samples-1)
    delta_last = torch.full(
        (t_vals.shape[0], 1), 1e10, device=t_vals.device
    )
    deltas = torch.cat([deltas, delta_last], dim=-1)            # (N, n_samples)

    # ── Compute alpha_i = 1 - exp(-sigma_i * delta_i) ──────────
    alpha = 1.0 - torch.exp(-sigmas * deltas)                  # (N, n_samples)

    # ── Compute transmittance T_i ────────────────────────────────
    # T_i = exp(-sum_{j<i} sigma_j * delta_j) = prod_{j<i} (1 - alpha_j)
    # We prepend a 1 and use cumprod, then remove the last element
    ones = torch.ones(sigmas.shape[0], 1, device=sigmas.device)
    T = torch.cumprod(
        torch.cat([ones, 1.0 - alpha + 1e-10], dim=-1), dim=-1
    )[:, :-1]  # (N, n_samples)

    # ── Per-sample weights: T_i * alpha_i ───────────────────────
    weights = T * alpha                                         # (N, n_samples)

    # ── Composite color ──────────────────────────────────────────
    rgb_map = (weights.unsqueeze(-1) * colors).sum(dim=1)       # (N, 3)

    return rgb_map, weights