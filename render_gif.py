"""
render_gif.py — Render the novel-view GIF from a saved NeRF model
==================================================================
Run this after training:
    python render_gif.py

Make sure nerf_outputs/nerf_model.pth and lego_200x200.npz are present.
"""

import torch
import torch.nn as nn
import numpy as np
import imageio
import os

from dataset_3d import load_data, image_to_rays, get_device
from rendering import sample_along_rays, volume_render


# ── Copy of PositionalEncoding and NeRF (must match training) ──
class PositionalEncoding(nn.Module):
    def __init__(self, L: int = 10):
        super().__init__()
        self.L = L

    def forward(self, x):
        encoded = [x]
        for k in range(self.L):
            freq = (2 ** k) * torch.pi
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)


class NeRF(nn.Module):
    def __init__(self, L_pts=10, L_dir=4, width=256):
        super().__init__()
        self.pe_pts = PositionalEncoding(L=L_pts)
        self.pe_dir = PositionalEncoding(L=L_dir)

        pts_enc_dim = 3 * (1 + 2 * L_pts)
        dir_enc_dim = 3 * (1 + 2 * L_dir)

        self.layers_pre_skip = nn.Sequential(
            nn.Linear(pts_enc_dim, width), nn.ReLU(),
            nn.Linear(width, width),       nn.ReLU(),
            nn.Linear(width, width),       nn.ReLU(),
            nn.Linear(width, width),       nn.ReLU(),
        )
        self.layers_post_skip = nn.Sequential(
            nn.Linear(width + pts_enc_dim, width), nn.ReLU(),
            nn.Linear(width, width),               nn.ReLU(),
            nn.Linear(width, width),               nn.ReLU(),
        )
        self.density_head = nn.Sequential(nn.Linear(width, 1), nn.ReLU())
        self.feature_head = nn.Linear(width, width)
        self.color_head = nn.Sequential(
            nn.Linear(width + dir_enc_dim, width // 2), nn.ReLU(),
            nn.Linear(width // 2, 3),
            nn.Sigmoid(),
        )

    def forward(self, pts, dirs):
        pts_enc = self.pe_pts(pts)
        dir_enc = self.pe_dir(dirs)
        h = self.layers_pre_skip(pts_enc)
        h = torch.cat([h, pts_enc], dim=-1)
        h = self.layers_post_skip(h)
        sigma = self.density_head(h).squeeze(-1)
        feat = self.feature_head(h)
        color = self.color_head(torch.cat([feat, dir_enc], dim=-1))
        return color, sigma


@torch.no_grad()
def render_image(model, rays_o, rays_d, near, far, n_samples, chunk=4096, device="cpu"):
    model.eval()
    all_rgb = []
    for i in range(0, rays_o.shape[0], chunk):
        r_os_c = rays_o[i:i+chunk].to(device)
        r_ds_c = rays_d[i:i+chunk].to(device)
        pts, t_vals = sample_along_rays(r_os_c, r_ds_c, near, far, n_samples, perturb=False)
        N_c = r_os_c.shape[0]
        dirs_exp = r_ds_c.unsqueeze(1).expand(-1, n_samples, -1)
        colors_f, sigmas_f = model(pts.reshape(-1, 3), dirs_exp.reshape(-1, 3))
        colors = colors_f.reshape(N_c, n_samples, 3)
        sigmas = sigmas_f.reshape(N_c, n_samples)
        rgb, _ = volume_render(sigmas, colors, t_vals)
        all_rgb.append(rgb.cpu())
    return torch.cat(all_rgb, dim=0)


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    # ── Config — must match your training settings ──────────
    DATA_PATH  = "lego_200x200.npz"
    MODEL_PATH = "nerf_outputs/nerf_model.pth"
    OUT_DIR    = "nerf_outputs"
    NEAR       = 2.0
    FAR        = 6.0
    N_SAMPLES  = 32   # use 64 if you trained with 64
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load data ────────────────────────────────────────────
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K = load_data(DATA_PATH)
    H, W = images_train.shape[1], images_train.shape[2]
    c2ws_test_t = torch.from_numpy(c2ws_test).float()

    # ── Load model ───────────────────────────────────────────
    model = NeRF(L_pts=10, L_dir=4, width=256).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    # ── Render each test camera ──────────────────────────────
    dummy_img = torch.zeros(H, W, 3)
    gif_frames = []

    for idx, c2w in enumerate(c2ws_test_t):
        print(f"Rendering test frame {idx+1}/{len(c2ws_test_t)}...")
        rays = image_to_rays(dummy_img, c2w, K, device=device)
        pred = render_image(
            model,
            rays[:, :, :3].reshape(-1, 3),
            rays[:, :, 3:].reshape(-1, 3),
            NEAR, FAR, N_SAMPLES,
            device=device,
        ).reshape(H, W, 3).numpy()
        frame = np.clip(pred * 255, 0, 255).astype(np.uint8)
        gif_frames.append(frame)

    gif_path = os.path.join(OUT_DIR, "lego_novel_view.gif")
    imageio.mimsave(gif_path, gif_frames, fps=15)
    print(f"Saved GIF to {gif_path}")