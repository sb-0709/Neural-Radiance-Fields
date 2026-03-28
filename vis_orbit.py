"""
vis_orbit.py — Render an orbit GIF around your custom NeRF scene
=================================================================
Usage:
    python vis_orbit.py

Adjust START_POS, near, far, and data_path for your scene.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import imageio
import os

from dataset_3d import load_data, image_to_rays, get_device
from rendering import sample_along_rays, volume_render


# ── Copy of model (must match training architecture) ─────────
class PositionalEncoding(nn.Module):
    def __init__(self, L=10):
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
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
        )
        self.layers_post_skip = nn.Sequential(
            nn.Linear(width + pts_enc_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
        )
        self.density_head = nn.Sequential(nn.Linear(width, 1), nn.ReLU())
        self.feature_head = nn.Linear(width, width)
        self.color_head = nn.Sequential(
            nn.Linear(width + dir_enc_dim, width // 2), nn.ReLU(),
            nn.Linear(width // 2, 3), nn.Sigmoid(),
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
        rgb, _ = volume_render(
            sigmas_f.reshape(N_c, n_samples),
            colors_f.reshape(N_c, n_samples, 3),
            t_vals,
        )
        all_rgb.append(rgb.cpu())
    return torch.cat(all_rgb, dim=0)


# ── Orbit camera helpers (from starter code) ─────────────────
def look_at_origin(pos):
    """Create a c2w matrix where camera at `pos` looks at the origin."""
    forward = -pos / np.linalg.norm(pos)
    up = np.array([0, 1, 0])
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward  # NOTE: may need -forward for some scenes
    c2w[:3, 3] = pos
    return c2w

def rot_x(phi):
    """Rotation matrix around the Z axis by phi radians."""
    return np.array([
        [math.cos(phi), -math.sin(phi), 0, 0],
        [math.sin(phi),  math.cos(phi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    # ── CONFIG — adjust these for your scene ─────────────────
    DATA_PATH  = "my_object.npz"   # your custom npz
    MODEL_PATH = "my_object_outputs/nerf_model.pth"
    OUT_PATH   = "my_object_outputs/orbit.gif"
    NEAR       = 2.0    # adjust for your scene scale
    FAR        = 6.0    # adjust for your scene scale
    N_SAMPLES  = 64
    NUM_FRAMES = 60     # frames in the orbit GIF
    FPS        = 15

    # START_POS: copy the translation vector (last column, first 3 rows)
    # from one of your training c2ws — this sets the orbit radius correctly.
    # For the lego scene this was roughly [4, 0, 0].
    # For real scenes, look at c2ws_train[0][:3, 3] after loading data.
    START_POS = np.array([4., 0., 0.])   # ← adjust this!

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # ── Load data (just need H, W, K) ────────────────────────
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K = load_data(DATA_PATH)
    H, W = images_train.shape[1], images_train.shape[2]

    # Print a training camera position to help you set START_POS
    c2ws_np = c2ws_train if isinstance(c2ws_train, np.ndarray) else c2ws_train.numpy()
    print(f"Tip — first training camera position: {c2ws_np[0][:3, 3]}")
    print(f"      Set START_POS to a similar radius from origin.")

    # ── Load model ───────────────────────────────────────────
    model = NeRF(L_pts=10, L_dir=4, width=256).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    # ── Render orbit frames ───────────────────────────────────
    dummy_img = torch.zeros(H, W, 3)
    frames = []

    for idx, phi in enumerate(np.linspace(360., 0., NUM_FRAMES, endpoint=False)):
        print(f"Rendering frame {idx+1}/{NUM_FRAMES} (phi={phi:.1f})...")

        # Build orbit camera pose
        c2w_np   = look_at_origin(START_POS)
        c2w_np   = rot_x(phi / 180. * np.pi) @ c2w_np
        c2w      = torch.from_numpy(c2w_np).float()

        # Get rays for this camera
        rays = image_to_rays(dummy_img, c2w, K, device=device)  # (H, W, 6)
        rays_o = rays[:, :, :3].reshape(-1, 3)
        rays_d = rays[:, :, 3:].reshape(-1, 3)

        # Render
        pred = render_image(
            model, rays_o, rays_d,
            NEAR, FAR, N_SAMPLES, device=device,
        ).reshape(H, W, 3).numpy()

        frame = np.clip(pred * 255, 0, 255).astype(np.uint8)
        frames.append(frame)

    imageio.mimsave(OUT_PATH, frames, fps=FPS)
    print(f"Saved orbit GIF to {OUT_PATH}")