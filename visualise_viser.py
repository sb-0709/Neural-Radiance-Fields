import viser
import time
import numpy as np
import torch
import tyro
from dataclasses import dataclass

from dataset_3d import load_data, RaysData
from rendering import sample_along_rays

def get_device() -> str:
    """Return the best available device: cuda > mps (Apple Silicon) > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@dataclass
class Config:
    data_path: str = "lego_200x200.npz"
    near: float = 2.0          # default near plane for lego
    far: float = 6.0           # default far plane for lego
    num_samples_along_ray: int = 32
    num_rays: int = 100        # keep ≤ 100 so the visualization isn't too crowded
    device: str = get_device()
    port: int = 8080


def main(cfg: Config):
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K = load_data(data_path=cfg.data_path)

    H, W = images_train.shape[1], images_train.shape[2]

    # FIX 1: load_data returns numpy arrays — convert to torch tensors first
    images_train_t = torch.from_numpy(images_train).float()
    c2ws_train_t   = torch.from_numpy(c2ws_train).float()
    K_t            = K  # already a tensor from load_data

    dataset = RaysData(
        images_train_t.to(cfg.device),
        K_t.to(cfg.device),
        c2ws_train_t.to(cfg.device),
        device=cfg.device,
    )

    # Verify uvs aren't flipped
    uvs_start = 0
    uvs_end = 40_000
    sample_uvs = dataset.uvs[uvs_start:uvs_end].cpu()
    assert torch.all(
        images_train_t[0, sample_uvs[:, 1], sample_uvs[:, 0]]
        == dataset.gt_rgbs[uvs_start:uvs_end].cpu()
    ), "UVs assertion FAILED — check image_to_rays pixel ordering!"
    print("UVs assertion passed!")

    # Sample random rays from the first image only
    num_pixels_per_image = H * W
    indices = np.random.randint(low=0, high=num_pixels_per_image, size=cfg.num_rays)

    rays_o = dataset.rays_o[indices].to(cfg.device)
    rays_d = dataset.rays_d[indices].to(cfg.device)

    # FIX 2: our sample_along_rays uses n_samples (not num_samples_along_ray)
    # and doesn't take a device arg — points are already on the same device as rays
    pts, t_vals = sample_along_rays(
        r_os=rays_o,
        r_ds=rays_d,
        near=cfg.near,
        far=cfg.far,
        n_samples=cfg.num_samples_along_ray,
        perturb=True,
    )  # pts: (num_rays, num_samples, 3)

    # Convert to numpy for viser
    rays_o_np  = rays_o.cpu().detach().numpy()
    rays_d_np  = rays_d.cpu().detach().numpy()
    points_np  = pts.cpu().detach().numpy()
    images_np  = images_train_t.cpu().detach().numpy()
    c2ws_np    = c2ws_train_t.cpu().detach().numpy()
    K_np       = K_t.cpu().detach().numpy()

    server = viser.ViserServer(port=cfg.port, share=True)

    fov    = float(2 * np.arctan2(H / 2, K_np[0, 0]))
    aspect = float(W / H)

    # Add all camera frustums
    for i, (image, c2w) in enumerate(zip(images_np, c2ws_np)):
        server.scene.add_camera_frustum(
            f"/cameras/{i}",
            fov=fov,
            aspect=aspect,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image,
        )

    # Add ray lines
    for i, (o, d) in enumerate(zip(rays_o_np, rays_d_np)):
        positions = np.stack((o, o + d * cfg.far))
        server.scene.add_spline_catmull_rom(
            f"/rays/{i}",
            positions=positions,
        )

    # Add sample points along rays
    server.scene.add_point_cloud(
        "/samples",
        colors=np.zeros_like(points_np).reshape(-1, 3),
        points=points_np.reshape(-1, 3),
        point_size=0.03,
    )

    print(f"Viser server running on port {cfg.port}. Press Ctrl+C to stop.")
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)