"""
dataset_3d.py — Part 2 implementation
======================================
This is the COMPLETE implementation of dataset_3d.py, filling in:
  - image_to_rays()       (Part 2.2)
  - images_to_rays()      (Part 2.2)
  - RaysData.__init__()   (Part 2.3)
  - RaysData.__len__()    (Part 2.3)
  - RaysData.sample_rays()(Part 2.3)

The functions load_data(), pixel_to_camera(), and pixels_to_rays()
are already implemented in the starter code and are included here
unchanged.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


# ─────────────────────────────────────────────────────────────
# DEVICE DETECTION — auto-picks cuda > mps > cpu
# ─────────────────────────────────────────────────────────────
def get_device() -> str:
    """Return the best available device: cuda > mps (Apple Silicon) > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Module-level default so every function shares it
_DEFAULT_DEVICE = get_device()


# ─────────────────────────────────────────────────────────────
# PROVIDED: load_data
# ─────────────────────────────────────────────────────────────
def load_data(data_path: str):
    """Load and preprocess dataset from an npz file.

    Args:
        data_path: str representing the path to the .npz data file

    Returns:
        images_train: torch.Tensor of shape (num_train, H, W, 3) normalized to [0, 1]
        c2ws_train:   torch.Tensor of shape (num_train, 4, 4)
        images_val:   torch.Tensor of shape (num_val, H, W, 3) normalized to [0, 1]
        c2ws_val:     torch.Tensor of shape (num_val, 4, 4)
        c2ws_test:    torch.Tensor of shape (num_test, 4, 4)
        K:            torch.Tensor of shape (3, 3)
    """
    data = np.load(data_path)

    images_train = data["images_train"] / 255.0   # (100, 200, 200, 3)
    c2ws_train   = data["c2ws_train"]             # (100, 4, 4)
    images_val   = data["images_val"] / 255.0     # (10, 200, 200, 3)
    c2ws_val     = data["c2ws_val"]               # (10, 4, 4)
    c2ws_test    = data["c2ws_test"]              # (60, 4, 4)
    focal        = data["focal"]                  # float

    h, w = images_train.shape[1], images_train.shape[2]
    o_x = w / 2
    o_y = h / 2
    K = torch.as_tensor([
        [focal.item(), 0,            o_x],
        [0,            focal.item(), o_y],
        [0,            0,            1  ],
    ])

    return images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K


# ─────────────────────────────────────────────────────────────
# PROVIDED: pixel_to_camera
# ─────────────────────────────────────────────────────────────
def pixel_to_camera(
    K: torch.Tensor,
    uvs: torch.Tensor,
    s: float = 1,
    device: str = _DEFAULT_DEVICE,
):
    """Pixel to camera transformation on a batch of pixels.

    Args:
        K:   (3, 3) camera intrinsics
        uvs: (num_pixels, 3) homogeneous pixel coordinates
        s:   scaling factor (depth)
        device: device string

    Returns:
        (num_pixels, 3) camera-space coordinates
    """
    K_inv = torch.linalg.inv(K).to(device)
    unnormalized_x_cs = (K_inv @ uvs.T).T  # (num_pixels, 3)
    return s * unnormalized_x_cs


# ─────────────────────────────────────────────────────────────
# PROVIDED: pixels_to_rays
# ─────────────────────────────────────────────────────────────
def pixels_to_rays(
    K: torch.Tensor,
    c2w: torch.Tensor,
    uvs: torch.Tensor,
    verbose: bool = False,
    device: str = _DEFAULT_DEVICE,
):
    """Convert pixels to rays.

    Args:
        K:    (3, 3) camera intrinsics
        c2w:  (4, 4) camera-to-world matrix
        uvs:  (num_pixels, 2) pixel coordinates
        verbose: print debug info
        device: device string

    Returns:
        r_os: (num_pixels, 3) ray origins
        r_ds: (num_pixels, 3) ray directions (unit vectors)
    """
    # Use float32 throughout — float64 is not supported on MPS
    K   = K.to(torch.float32).to(device)
    c2w = c2w.to(torch.float32).to(device)
    uvs = uvs.to(torch.float32).to(device)

    if verbose:
        print(f"K shape: {K.shape} should be (3,3)")
        print(f"c2w shape: {c2w.shape} should be (4,4)")
        print(f"uvs shape: {uvs.shape} should be (num_pixels, 2)")

    num_pixels = uvs.shape[0]
    R    = c2w[:3, :3]                                       # (3, 3)
    r_os = c2w[:3, 3].unsqueeze(0).expand(num_pixels, -1)   # (num_pixels, 3)

    homog_uvs = torch.hstack(
        (uvs, torch.ones(num_pixels, 1, device=device))
    )  # (num_pixels, 3)

    K_inv = torch.linalg.inv(K)
    M     = R @ K_inv                          # (3, 3)
    dirs  = (M @ homog_uvs.T).T               # (num_pixels, 3)
    r_ds  = dirs / torch.linalg.norm(dirs, dim=1, keepdim=True)

    return r_os, r_ds  # each (num_pixels, 3)


# ─────────────────────────────────────────────────────────────
# IMPL: image_to_rays  (Part 2.2)
# ─────────────────────────────────────────────────────────────
def image_to_rays(
    image: torch.Tensor,
    c2w: torch.Tensor,
    K: torch.Tensor,
    verbose: bool = False,
    device: str = _DEFAULT_DEVICE,
):
    """Convert a single image to rays.

    For every pixel (u, v) in the image we compute the ray that passes
    through that pixel given the camera intrinsics K and the camera pose c2w.

    Important detail:  we add 0.5 to integer pixel coordinates so that
    the ray passes through the *center* of each pixel (standard convention).

    Args:
        image: (H, W, 3) float tensor — used only for its shape here
        c2w:   (4, 4) camera-to-world matrix
        K:     (3, 3) camera intrinsics
        verbose: print debug info
        device: device string

    Returns:
        rays: (H, W, 6) where rays[:, :, :3] are origins and
                             rays[:, :, 3:] are directions
    """
    H, W = image.shape[:2]

    # Build a grid of pixel centers: (u + 0.5, v + 0.5) for u in [0,W), v in [0,H)
    # u = column (x), v = row (y)
    us = torch.arange(W, dtype=torch.float32, device=device) + 0.5  # (W,)
    vs = torch.arange(H, dtype=torch.float32, device=device) + 0.5  # (H,)

    # torch.meshgrid returns (H, W) grids
    grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")  # both (H, W)

    # Flatten to (H*W, 2) — each row is [u, v]
    uvs_flat = torch.stack([grid_u.reshape(-1), grid_v.reshape(-1)], dim=1)  # (H*W, 2)

    # Get ray origins and directions using the provided helper
    r_os_flat, r_ds_flat = pixels_to_rays(K, c2w, uvs_flat, verbose=verbose, device=device)
    # r_os_flat: (H*W, 3), r_ds_flat: (H*W, 3)

    # Cast back to float32 for model compatibility
    r_os_flat = r_os_flat.to(torch.float32)
    r_ds_flat = r_ds_flat.to(torch.float32)

    # Reshape back to (H, W, 3) for origins and directions
    r_os = r_os_flat.reshape(H, W, 3)
    r_ds = r_ds_flat.reshape(H, W, 3)

    # Concatenate along last dim → (H, W, 6)
    rays = torch.cat([r_os, r_ds], dim=-1)  # (H, W, 6)
    return rays


# ─────────────────────────────────────────────────────────────
# IMPL: images_to_rays  (Part 2.2)
# ─────────────────────────────────────────────────────────────
def images_to_rays(
    images: torch.Tensor,
    c2ws: torch.Tensor,
    K: torch.Tensor,
    verbose: bool = False,
    device: str = _DEFAULT_DEVICE,
):
    """Convert a batch of images to rays.

    Calls image_to_rays() for each image and stacks the results.

    Args:
        images: (num_images, H, W, 3)
        c2ws:   (num_images, 4, 4)
        K:      (3, 3)
        verbose: print debug info
        device: device string

    Returns:
        all_rays: (num_images, H, W, 6)
                   [:, :, :, :3] = origins
                   [:, :, :, 3:] = directions
    """
    num_images = images.shape[0]
    rays_list = []

    for i in range(num_images):
        # image_to_rays only uses image.shape so we can pass images[i] directly
        rays_i = image_to_rays(images[i], c2ws[i], K, verbose=verbose, device=device)
        rays_list.append(rays_i)  # (H, W, 6)

    all_rays = torch.stack(rays_list, dim=0)  # (num_images, H, W, 6)
    return all_rays


# ─────────────────────────────────────────────────────────────
# IMPL: RaysData  (Part 2.3)
# ─────────────────────────────────────────────────────────────
class RaysData(Dataset):
    """
    Dataset that precomputes rays for every pixel across all images,
    then supports random ray sampling at training time.
    """

    def __init__(
        self,
        images: torch.Tensor,
        K: torch.Tensor,
        c2ws: torch.Tensor,
        split: str = "train",
        device: str = _DEFAULT_DEVICE,
    ):
        """
        Precompute all rays and flatten them.

        Args:
            images: (num_images, H, W, 3)  float in [0, 1]
            K:      (3, 3) camera intrinsics
            c2ws:   (num_images, 4, 4)
            split:  "train", "val", or "test"
            device: device string

        After __init__ the following attributes are set:
            self.uvs:     (N, 2)   integer pixel coordinates (x=u, y=v)
                          N = num_images * H * W
            self.rays_o:  (N, 3)   ray origins
            self.rays_d:  (N, 3)   ray directions
            self.gt_rgbs: (N, 3)   ground truth colors
        """
        self.images    = images
        self.K         = K
        self.c2ws      = c2ws
        self.split     = split
        self.device    = device

        num_images, H, W, _ = images.shape
        self.H = H
        self.W = W
        self.num_images = num_images

        # ── Build integer pixel coordinates ──────────────────
        # For a single image: uvs go (u=0,v=0), (u=1,v=0), ..., (u=W-1,v=H-1)
        # Convention: u = column (x), v = row (y)
        us = torch.arange(W)   # (W,)
        vs = torch.arange(H)   # (H,)
        grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")  # (H, W) each

        # Pixel centers for a single image: shape (H, W, 2)
        uvs_single = torch.stack([grid_u, grid_v], dim=-1)  # (H, W, 2)

        # Repeat for num_images → (num_images, H, W, 2) → flatten → (N, 2)
        uvs_all = uvs_single.unsqueeze(0).expand(num_images, -1, -1, -1)
        self.uvs = uvs_all.reshape(-1, 2)  # (N, 2)  integer pixel coords

        # ── Compute all rays ──────────────────────────────────
        # images_to_rays returns (num_images, H, W, 6)
        # We pass images as torch.Tensor with float dtype
        images_t = images if isinstance(images, torch.Tensor) else torch.from_numpy(images).float()
        c2ws_t   = c2ws   if isinstance(c2ws,   torch.Tensor) else torch.from_numpy(c2ws).float()

        all_rays = images_to_rays(images_t, c2ws_t, K, device=device)
        # all_rays: (num_images, H, W, 6)

        # Flatten to (N, 6)
        all_rays_flat = all_rays.reshape(-1, 6)

        self.rays_o = all_rays_flat[:, :3].cpu()  # (N, 3) — store on CPU to save VRAM
        self.rays_d = all_rays_flat[:, 3:].cpu()  # (N, 3)

        # ── Ground truth colors ───────────────────────────────
        # images: (num_images, H, W, 3) → (N, 3)
        self.gt_rgbs = images_t.reshape(-1, 3).cpu()  # (N, 3)

    def __len__(self):
        """Return the total number of rays: num_images * H * W."""
        return self.gt_rgbs.shape[0]

    def sample_rays(self, num_rays: int):
        """
        Randomly sample num_rays from the dataset.

        Args:
            num_rays: number of rays to sample

        Returns:
            r_os:    (num_rays, 3)  ray origins
            r_ds:    (num_rays, 3)  ray directions
            gt_rgbs: (num_rays, 3)  ground truth colors
        """
        # Draw random indices without replacement (use with replacement if num_rays > N)
        N = len(self)
        indices = torch.randint(0, N, (num_rays,))

        r_os    = self.rays_o[indices].to(self.device)
        r_ds    = self.rays_d[indices].to(self.device)
        gt_rgbs = self.gt_rgbs[indices].to(self.device)

        return r_os, r_ds, gt_rgbs