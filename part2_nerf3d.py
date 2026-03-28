"""
part2_nerf3d.py — Part 2: Fit a Neural Radiance Field from Multi-view Images
=============================================================================
This file implements:
  1. PositionalEncoding (same as Part 1, reused)
  2. NeRF MLP (3D version with view-direction conditioning)
  3. Training loop for NeRF
  4. Novel-view rendering and GIF generation

Run this file AFTER downloading the lego_200x200.npz dataset:
    wget https://coms4732.github.io/hws/hw4/assets/lego_200x200.npz

Usage:
    python part2_nerf3d.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

from dataset_3d_2 import load_data, RaysData, get_device
from rendering import sample_along_rays, volume_render


# ─────────────────────────────────────────────────────────────
# 1. POSITIONAL ENCODING (reused from Part 1)
# ─────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    PE(x) = [x, sin(2^0 pi x), cos(2^0 pi x), ..., sin(2^(L-1) pi x), cos(2^(L-1) pi x)]
    Input dim D → output dim D*(1 + 2L)
    """
    def __init__(self, L: int = 10):
        super().__init__()
        self.L = L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = [x]
        for k in range(self.L):
            freq = (2 ** k) * torch.pi
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)


# ─────────────────────────────────────────────────────────────
# 2. NERF MLP (Part 2.4)
# ─────────────────────────────────────────────────────────────
class NeRF(nn.Module):
    """
    Neural Radiance Field MLP.

    Architecture (following the homework spec diagram):
    ─────────────────────────────────────────────────────
    Input: (x, y, z)  with PE at L=10  → dim = 3*(1+20) = 63
    Input: (dx, dy, dz) with PE at L=4 → dim = 3*(1+8)  = 27

    Network:
        pts_enc  → Linear(63, W) → ReLU
                 → Linear(W, W)  → ReLU
                 → Linear(W, W)  → ReLU
                 → Linear(W, W)  → ReLU
                 → CONCAT with pts_enc again  (skip connection)
                 → Linear(W+63, W) → ReLU
                 → Linear(W, W)    → ReLU
                 → Linear(W, W)    → ReLU
                 → Linear(W, 1)    → ReLU  → density sigma (must be ≥ 0)
                 → Linear(W, W)             → feature vector
        feature_vec CONCAT dir_enc
                 → Linear(W+27, W//2)  → ReLU
                 → Linear(W//2, 3)     → Sigmoid  → RGB color [0,1]

    The skip connection (re-injecting the input halfway through)
    helps the deep network retain information about the input.
    """
    def __init__(
        self,
        L_pts: int = 10,   # PE levels for 3D coordinates
        L_dir: int = 4,    # PE levels for view directions
        width: int = 256,  # hidden layer width
    ):
        super().__init__()

        self.pe_pts = PositionalEncoding(L=L_pts)
        self.pe_dir = PositionalEncoding(L=L_dir)

        # Encoded input dimensions
        pts_enc_dim = 3 * (1 + 2 * L_pts)   # = 63 for L=10
        dir_enc_dim = 3 * (1 + 2 * L_dir)   # = 27 for L=4

        # First half of MLP (before skip connection): 4 layers
        self.layers_pre_skip = nn.Sequential(
            nn.Linear(pts_enc_dim, width), nn.ReLU(),
            nn.Linear(width, width),       nn.ReLU(),
            nn.Linear(width, width),       nn.ReLU(),
            nn.Linear(width, width),       nn.ReLU(),
        )

        # Second half of MLP (after skip connection): 3 layers
        # Input size = width + pts_enc_dim  (feature + re-injected input)
        self.layers_post_skip = nn.Sequential(
            nn.Linear(width + pts_enc_dim, width), nn.ReLU(),
            nn.Linear(width, width),               nn.ReLU(),
            nn.Linear(width, width),               nn.ReLU(),
        )

        # Density head: outputs 1 scalar, ReLU to enforce non-negativity
        self.density_head = nn.Sequential(
            nn.Linear(width, 1),
            nn.ReLU(),
        )

        # Feature head (bottleneck before color prediction)
        self.feature_head = nn.Linear(width, width)

        # Color head: conditioned on view direction
        self.color_head = nn.Sequential(
            nn.Linear(width + dir_enc_dim, width // 2), nn.ReLU(),
            nn.Linear(width // 2, 3),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(
        self,
        pts: torch.Tensor,
        dirs: torch.Tensor,
    ):
        """
        Args:
            pts:  (N, 3) 3D world coordinates
            dirs: (N, 3) unit ray directions (view direction)

        Returns:
            colors: (N, 3) RGB in [0, 1]
            sigmas: (N,)   density (≥ 0)
        """
        # Positional encoding
        pts_enc = self.pe_pts(pts)    # (N, 63)
        dir_enc = self.pe_dir(dirs)   # (N, 27)

        # First MLP block
        h = self.layers_pre_skip(pts_enc)   # (N, W)

        # Skip connection: concatenate the encoded input
        h = torch.cat([h, pts_enc], dim=-1)  # (N, W + 63)

        # Second MLP block
        h = self.layers_post_skip(h)         # (N, W)

        # Density: (N, 1) → squeeze to (N,)
        sigma = self.density_head(h).squeeze(-1)  # (N,)

        # Feature for color prediction
        feat = self.feature_head(h)               # (N, W)

        # Color: conditioned on view direction
        color_input = torch.cat([feat, dir_enc], dim=-1)  # (N, W+27)
        color = self.color_head(color_input)               # (N, 3)

        return color, sigma


# ─────────────────────────────────────────────────────────────
# 3. UTILITY: Compute PSNR
# ─────────────────────────────────────────────────────────────
def compute_psnr(mse: float) -> float:
    return 10.0 * np.log10(1.0 / (mse + 1e-10))


# ─────────────────────────────────────────────────────────────
# 4. UTILITY: Render a full image from the NeRF
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def render_image(
    model: NeRF,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    chunk: int = 4096,
    device: str = "cuda",
):
    """
    Render a full image by splitting rays into chunks to avoid OOM.

    Args:
        model:    trained NeRF
        rays_o:   (H*W, 3) ray origins
        rays_d:   (H*W, 3) ray directions
        near, far: depth bounds
        n_samples: number of samples per ray
        chunk:    number of rays to process at once
        device:   device string

    Returns:
        rgb_map: (H*W, 3) rendered colors
    """
    model.eval()
    all_rgb = []

    for i in range(0, rays_o.shape[0], chunk):
        r_os_chunk = rays_o[i:i+chunk].to(device)
        r_ds_chunk = rays_d[i:i+chunk].to(device)

        # Sample points along rays
        pts, t_vals = sample_along_rays(r_os_chunk, r_ds_chunk, near, far, n_samples, perturb=False)
        # pts: (chunk, n_samples, 3), t_vals: (chunk, n_samples)

        N_chunk = r_os_chunk.shape[0]

        # Expand ray directions to match pts shape: (chunk, n_samples, 3)
        dirs_expanded = r_ds_chunk.unsqueeze(1).expand(-1, n_samples, -1)  # (chunk, n_samples, 3)

        # Flatten for model input
        pts_flat  = pts.reshape(-1, 3)           # (chunk*n_samples, 3)
        dirs_flat = dirs_expanded.reshape(-1, 3)  # (chunk*n_samples, 3)

        # Query NeRF
        colors_flat, sigmas_flat = model(pts_flat, dirs_flat)
        # colors_flat: (chunk*n_samples, 3), sigmas_flat: (chunk*n_samples,)

        # Reshape for volume rendering
        colors = colors_flat.reshape(N_chunk, n_samples, 3)
        sigmas = sigmas_flat.reshape(N_chunk, n_samples)

        # Volume render
        rgb_chunk, _ = volume_render(sigmas, colors, t_vals)  # (chunk, 3)
        all_rgb.append(rgb_chunk.cpu())

    return torch.cat(all_rgb, dim=0)  # (H*W, 3)


# ─────────────────────────────────────────────────────────────
# 5. TRAINING LOOP (Part 2.4 + 2.5)
# ─────────────────────────────────────────────────────────────
def train_nerf(
    data_path: str = "toy_frames_10pct_200x200.npz",
    num_iters: int = 5000,
    batch_size: int = 10_000,
    lr: float = 5e-4,
    near: float = 2.0,
    far: float = 6.0,
    n_samples: int = 64,
    device: str = None,
    val_interval: int = 500,
    save_dir: str = "nerf_outputs",
):
    """
    Full NeRF training pipeline.

    Args:
        data_path:    path to lego_200x200.npz
        num_iters:    number of gradient steps
        batch_size:   number of rays per gradient step
        lr:           Adam learning rate
        near, far:    depth bounds for ray sampling
        n_samples:    number of samples per ray
        device:       torch device string
        val_interval: compute validation PSNR every N iters
        save_dir:     directory to save outputs
    """
    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = get_device()
    print(f"Training NeRF on device: {device}")

    # ── Load data ────────────────────────────────────────────
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K = load_data(data_path)
    H, W = images_train.shape[1], images_train.shape[2]
    print(f"Train: {images_train.shape}, Val: {images_val.shape}")

    # Convert to tensors
    images_train_t = torch.from_numpy(images_train).float()
    c2ws_train_t   = torch.from_numpy(c2ws_train).float()
    images_val_t   = torch.from_numpy(images_val).float()
    c2ws_val_t     = torch.from_numpy(c2ws_val).float()
    c2ws_test_t    = torch.from_numpy(c2ws_test).float()

    # ── Build training dataset ───────────────────────────────
    print("Precomputing training rays...")
    train_dataset = RaysData(images_train_t, K, c2ws_train_t, split="train", device=device)
    print(f"Total training rays: {len(train_dataset)}")

    # ── Build model and optimizer ────────────────────────────
    model = NeRF(L_pts=10, L_dir=4, width=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"NeRF parameters: {num_params:,}")

    psnr_history = []
    snapshot_frames = []

    # ── Training loop ────────────────────────────────────────
    for iteration in range(1, num_iters + 1):
        model.train()

        # Sample random rays from training set
        r_os, r_ds, gt_rgbs = train_dataset.sample_rays(batch_size)
        # r_os, r_ds: (batch_size, 3), gt_rgbs: (batch_size, 3)

        # Sample 3D points along each ray
        pts, t_vals = sample_along_rays(r_os, r_ds, near, far, n_samples, perturb=True)
        # pts: (batch_size, n_samples, 3), t_vals: (batch_size, n_samples)

        # Expand ray directions to match sampled points
        # Each point along a ray shares the same view direction
        dirs_expanded = r_ds.unsqueeze(1).expand(-1, n_samples, -1)  # (batch_size, n_samples, 3)

        # Flatten for model input
        pts_flat  = pts.reshape(-1, 3)            # (batch_size*n_samples, 3)
        dirs_flat = dirs_expanded.reshape(-1, 3)   # (batch_size*n_samples, 3)

        # Query NeRF: predict color and density at each point
        colors_flat, sigmas_flat = model(pts_flat, dirs_flat)

        # Reshape outputs
        colors = colors_flat.reshape(batch_size, n_samples, 3)
        sigmas = sigmas_flat.reshape(batch_size, n_samples)

        # Volume render: composite points along each ray → pixel color
        pred_rgbs, _ = volume_render(sigmas, colors, t_vals)  # (batch_size, 3)

        # MSE loss between predicted and ground truth pixel colors
        loss = loss_fn(pred_rgbs, gt_rgbs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Logging ──────────────────────────────────────────
        if iteration % 100 == 0 or iteration == 1:
            train_psnr = compute_psnr(loss.item())
            print(f"Iter {iteration:5d} | Loss: {loss.item():.6f} | Train PSNR: {train_psnr:.2f} dB")

        # ── Validation ───────────────────────────────────────
        if iteration % val_interval == 0 or iteration == num_iters:
            print(f"  → Computing validation PSNR at iter {iteration}...")
            val_psnrs = []

            for v_idx in range(min(6, len(images_val_t))):
                # Get all rays for this validation image
                val_rays_o_all = []
                val_rays_d_all = []
                from dataset_3d_2 import image_to_rays
                rays_v = image_to_rays(
                    images_val_t[v_idx], c2ws_val_t[v_idx], K, device=device
                )  # (H, W, 6)
                val_rays_o = rays_v[:, :, :3].reshape(-1, 3)  # (H*W, 3)
                val_rays_d = rays_v[:, :, 3:].reshape(-1, 3)  # (H*W, 3)

                # Render full image
                pred_rgb_flat = render_image(
                    model, val_rays_o, val_rays_d,
                    near, far, n_samples, device=device
                )  # (H*W, 3)
                pred_img = pred_rgb_flat.reshape(H, W, 3).numpy()
                gt_img   = images_val_t[v_idx].numpy()

                mse = float(np.mean((pred_img - gt_img) ** 2))
                val_psnrs.append(compute_psnr(mse))

            avg_val_psnr = float(np.mean(val_psnrs))
            psnr_history.append((iteration, avg_val_psnr))
            print(f"  → Val PSNR (avg over 6 images): {avg_val_psnr:.2f} dB")

            # ── Save checkpoint every val_interval ───────────
            ckpt_path = os.path.join(save_dir, "nerf_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Saved checkpoint to {ckpt_path}")

            # Save a snapshot of the first validation image
            rays_v0 = image_to_rays(images_val_t[0], c2ws_val_t[0], K, device=device)
            pred_v0 = render_image(
                model,
                rays_v0[:, :, :3].reshape(-1, 3),
                rays_v0[:, :, 3:].reshape(-1, 3),
                near, far, n_samples, device=device
            ).reshape(H, W, 3).numpy()
            snapshot_frames.append((iteration, np.clip(pred_v0, 0, 1)))

    # ── Plot PSNR curve ──────────────────────────────────────
    iters, psnrs = zip(*psnr_history)
    plt.figure(figsize=(8, 4))
    plt.plot(iters, psnrs, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR (dB)")
    plt.title("Validation PSNR during NeRF training")
    plt.axhline(y=23, color='r', linestyle='--', label='Target (23 dB)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_psnr_curve.png"), dpi=100)
    plt.close()
    print(f"Saved PSNR curve")

    # ── Plot training progression ─────────────────────────────
    n = len(snapshot_frames)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    gt_display = images_val_t[0].numpy()
    for ax, (it, img) in zip(axes, snapshot_frames):
        ax.imshow(img)
        ax.set_title(f"Iter {it}")
        ax.axis("off")
    plt.suptitle("NeRF Training Progression (Validation View 0)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "nerf_progression.png"), dpi=100, bbox_inches="tight")
    plt.close()

    # ── Render novel-view GIF using test cameras ──────────────
    print("Rendering novel-view GIF from test cameras...")
    gif_frames = []
    from dataset_3d_2 import image_to_rays

    # Use a dummy image (same shape) just to get rays
    dummy_img = torch.zeros(H, W, 3)
    for c2w_test in c2ws_test_t:
        rays_t = image_to_rays(dummy_img, c2w_test, K, device=device)
        pred_frame = render_image(
            model,
            rays_t[:, :, :3].reshape(-1, 3),
            rays_t[:, :, 3:].reshape(-1, 3),
            near, far, n_samples, device=device
        ).reshape(H, W, 3).numpy()
        pred_frame = np.clip(pred_frame * 255, 0, 255).astype(np.uint8)
        gif_frames.append(pred_frame)

    gif_path = os.path.join(save_dir, "toy_novel_view.gif")
    imageio.mimsave(gif_path, gif_frames, fps=15)
    print(f"Saved novel-view GIF to {gif_path}")

    return model


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    SAVE_DIR   = "toy_nerf_outputs"
    CKPT_PATH  = os.path.join(SAVE_DIR, "nerf_model.pth")
    DATA_PATH  = "toy_frames_10pct_200x200.npz"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── If a checkpoint already exists, just render the GIF ──
    if os.path.exists(CKPT_PATH):
        print(f"Found existing checkpoint at {CKPT_PATH}")
        print("Skipping training — loading model and rendering GIF...")

        from dataset_3d_2 import load_data, image_to_rays
        images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K = load_data(DATA_PATH)
        H, W   = images_train.shape[1], images_train.shape[2]
        c2ws_test_t = torch.from_numpy(c2ws_test).float()

        model = NeRF(L_pts=10, L_dir=4, width=256).to(device)
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        model.eval()
        print("Model loaded!")

        dummy_img = torch.zeros(H, W, 3)
        gif_frames = []
        for idx, c2w in enumerate(c2ws_test_t):
            print(f"  Rendering frame {idx+1}/{len(c2ws_test_t)}...")
            rays_t = image_to_rays(dummy_img, c2w, K, device=device)
            pred_frame = render_image(
                model,
                rays_t[:, :, :3].reshape(-1, 3),
                rays_t[:, :, 3:].reshape(-1, 3),
                near=2.0, far=6.0, n_samples=64, device=device,
            ).reshape(H, W, 3).numpy()
            gif_frames.append(np.clip(pred_frame * 255, 0, 255).astype(np.uint8))

        gif_path = os.path.join(SAVE_DIR, "toy_novel_view.gif")
        imageio.mimsave(gif_path, gif_frames, fps=15)
        print(f"Saved GIF to {gif_path}")

    else:
        # ── No checkpoint found — train from scratch ──────────
        model = train_nerf(
            data_path=DATA_PATH,
            num_iters=5000,
            batch_size=10_000,
            lr=5e-4,
            near=2.0,
            far=6.0,
            n_samples=64,
            device=device,
            val_interval=500,
            save_dir=SAVE_DIR,
        )
        # Final save (also saved every val_interval during training)
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"Saved final model to {CKPT_PATH}")