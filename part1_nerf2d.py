"""
Part 1: Fit a Neural Field to a 2D Image
=========================================
This file implements:
  1. Sinusoidal Positional Encoding (PE)
  2. MLP network (takes 2D pixel coords -> RGB color)
  3. Image Dataloader (randomly samples N pixels per iteration)
  4. Training loop with MSE loss, Adam optimizer, and PSNR metric
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO


# ─────────────────────────────────────────────────────────────
# 1. POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.

    For an input x (any number of dimensions), produces:
        PE(x) = [x, sin(2^0 * pi * x), cos(2^0 * pi * x),
                    sin(2^1 * pi * x), cos(2^1 * pi * x),
                    ...
                    sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)]

    Input shape:  (..., D)
    Output shape: (..., D + 2*D*L)

    For 2D input (D=2) with L=10:
        output dim = 2 + 2*2*10 = 42
    """
    def __init__(self, L: int = 10):
        """
        Args:
            L: number of frequency levels (max frequency = 2^(L-1))
        """
        super().__init__()
        self.L = L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., D) input coordinates, values in [0, 1]
        Returns:
            encoded: (..., D + 2*D*L) positionally encoded coordinates
        """
        # Start with the original coordinates
        encoded = [x]
        for k in range(self.L):
            freq = (2 ** k) * torch.pi
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)  # (..., D*(1 + 2L))


# ─────────────────────────────────────────────────────────────
# 2. MLP NETWORK (2D → RGB)
# ─────────────────────────────────────────────────────────────
class NeuralField2D(nn.Module):
    """
    A simple MLP that maps 2D pixel coordinates (u, v) → RGB color.

    Architecture:
        PE(2D) → Linear → ReLU → Linear → ReLU → ... → Linear → Sigmoid
                  ↑ num_layers hidden layers of width `width`

    The Sigmoid at the end ensures output is in [0, 1], matching
    normalized pixel colors.
    """
    def __init__(
        self,
        L: int = 10,          # positional encoding frequency levels
        width: int = 256,     # hidden layer width
        num_layers: int = 4,  # number of hidden layers
    ):
        super().__init__()
        self.pe = PositionalEncoding(L=L)

        # Input dimension after PE: 2 + 2*2*L = 2*(1 + 2L)
        input_dim = 2 + 2 * 2 * L  # = 42 for L=10

        layers = []
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, 3))  # output: R, G, B
        layers.append(nn.Sigmoid())         # clamp output to [0, 1]

        self.mlp = nn.Sequential(*layers)

    def forward(self, uvs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            uvs: (N, 2) pixel coordinates normalized to [0, 1]
        Returns:
            colors: (N, 3) predicted RGB values in [0, 1]
        """
        encoded = self.pe(uvs)   # (N, 42)
        return self.mlp(encoded) # (N, 3)


# ─────────────────────────────────────────────────────────────
# 3. DATALOADER
# ─────────────────────────────────────────────────────────────
class ImageDataset:
    """
    Dataloader that randomly samples N pixels from an image each iteration.

    Returns:
        uvs:    (N, 2) normalized pixel coordinates in [0, 1]
        colors: (N, 3) normalized pixel colors in [0, 1]
    """
    def __init__(self, image: np.ndarray):
        """
        Args:
            image: (H, W, 3) uint8 numpy array
        """
        H, W, _ = image.shape
        self.H = H
        self.W = W

        # Normalize colors to [0, 1]
        colors = image.astype(np.float32) / 255.0  # (H, W, 3)

        # Build a grid of all (u, v) pixel coordinates
        # u = column index (x), v = row index (y)
        us = np.arange(W)  # [0, 1, ..., W-1]
        vs = np.arange(H)  # [0, 1, ..., H-1]
        uu, vv = np.meshgrid(us, vs)  # each (H, W)

        # Normalize coordinates to [0, 1]
        uvs = np.stack([uu / W, vv / H], axis=-1)  # (H, W, 2)

        # Flatten to (H*W, 2) and (H*W, 3)
        self.all_uvs = torch.from_numpy(uvs.reshape(-1, 2)).float()
        self.all_colors = torch.from_numpy(colors.reshape(-1, 3)).float()

    def sample(self, num_pixels: int, device: str = "cuda"):
        """
        Randomly sample num_pixels pixels.

        Args:
            num_pixels: number of pixels to sample
            device: torch device string
        Returns:
            uvs:    (num_pixels, 2) normalized coords
            colors: (num_pixels, 3) normalized colors
        """
        indices = torch.randint(0, len(self.all_uvs), (num_pixels,))
        return (
            self.all_uvs[indices].to(device),
            self.all_colors[indices].to(device),
        )

    def all_pixels(self, device: str = "cuda"):
        """Return all pixels (used for full-image evaluation)."""
        return self.all_uvs.to(device), self.all_colors.to(device)


# ─────────────────────────────────────────────────────────────
# 4. METRICS
# ─────────────────────────────────────────────────────────────
def compute_psnr(mse: float) -> float:
    """
    Compute Peak Signal-to-Noise Ratio from MSE.
    Assumes pixel values are in [0, 1].

    PSNR = 10 * log10(1 / MSE)
    """
    return 10.0 * np.log10(1.0 / (mse + 1e-10))


# ─────────────────────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def train_2d_nerf(
    image: np.ndarray,
    num_iters: int = 2000,
    batch_size: int = 10_000,
    lr: float = 1e-2,
    L: int = 10,
    width: int = 256,
    num_layers: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_interval: int = 250,   # save visualization every N iters
    title: str = "training",
):
    """
    Train a 2D Neural Field to fit an image.

    Args:
        image:         (H, W, 3) uint8 numpy image
        num_iters:     number of training iterations
        batch_size:    number of pixels sampled per iteration
        lr:            Adam learning rate
        L:             positional encoding frequency levels
        width:         MLP hidden width
        num_layers:    number of MLP hidden layers
        device:        torch device
        save_interval: save visualization every N iterations
        title:         prefix for saved figures

    Returns:
        model: trained NeuralField2D
        psnr_history: list of (iteration, psnr) tuples
    """
    print(f"Training on device: {device}")
    print(f"Image shape: {image.shape}")
    print(f"Architecture: L={L}, width={width}, num_layers={num_layers}")

    # ── Setup ──────────────────────────────────────────────
    dataset = ImageDataset(image)
    model = NeuralField2D(L=L, width=width, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    H, W = image.shape[:2]
    psnr_history = []

    # Collect snapshots for visualization
    snapshot_iters = [1, 100, 250, 500, 1000, 2000, 3000]
    snapshots = {}

    # ── Training loop ──────────────────────────────────────
    for iteration in range(1, num_iters + 1):
        model.train()

        # Sample a random batch of pixels
        uvs, gt_colors = dataset.sample(batch_size, device=device)

        # Forward pass
        pred_colors = model(uvs)  # (N, 3)

        # Compute loss
        loss = loss_fn(pred_colors, gt_colors)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Logging ─────────────────────────────────────────
        if iteration % 100 == 0 or iteration == 1:
            mse_val = loss.item()
            psnr = compute_psnr(mse_val)
            psnr_history.append((iteration, psnr))
            print(f"Iter {iteration:5d} | MSE: {mse_val:.6f} | PSNR: {psnr:.2f} dB")

        # ── Save snapshot ────────────────────────────────────
        if iteration in snapshot_iters or iteration == num_iters:
            model.eval()
            with torch.no_grad():
                all_uvs, _ = dataset.all_pixels(device=device)
                pred_full = model(all_uvs)  # (H*W, 3)
                pred_img = pred_full.cpu().numpy().reshape(H, W, 3)
                pred_img = np.clip(pred_img, 0, 1)
                snapshots[iteration] = pred_img

    # ── Plot training progression ──────────────────────────
    n_snaps = len(snapshots)
    fig, axes = plt.subplots(1, n_snaps + 1, figsize=(3 * (n_snaps + 1), 3))
    axes[0].imshow(image)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    for ax, (it, img) in zip(axes[1:], sorted(snapshots.items())):
        ax.imshow(img)
        ax.set_title(f"Iter {it}")
        ax.axis("off")
    plt.suptitle(f"{title} | L={L}, width={width}")
    plt.tight_layout()
    plt.savefig(f"{title}_progression.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved progression plot to {title}_progression.png")

    # ── Plot PSNR curve ─────────────────────────────────────
    iters, psnrs = zip(*psnr_history)
    plt.figure(figsize=(8, 4))
    plt.plot(iters, psnrs, marker="o", markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("PSNR (dB)")
    plt.title(f"PSNR over Training | L={L}, width={width}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title}_psnr.png", dpi=100)
    plt.close()
    print(f"Saved PSNR curve to {title}_psnr.png")

    return model, psnr_history


# ─────────────────────────────────────────────────────────────
# 6. HYPERPARAMETER GRID (2x2)
# ─────────────────────────────────────────────────────────────
def run_hyperparameter_grid(image: np.ndarray, device: str = "cpu"):
    """
    Run a 2x2 grid of experiments over:
        - 2 choices of L (positional encoding frequencies): e.g. 4 and 10
        - 2 choices of width: e.g. 64 and 256
    Saves a 2x2 grid figure of final rendered images.
    """
    L_values = [4, 10]
    width_values = [64, 256]
    results = {}

    for L in L_values:
        for width in width_values:
            print(f"\n{'='*50}")
            print(f"Running: L={L}, width={width}")
            model, _ = train_2d_nerf(
                image,
                num_iters=1000,
                batch_size=10_000,
                L=L,
                width=width,
                device=device,
                title=f"L{L}_w{width}",
            )
            # Render full image
            H, W = image.shape[:2]
            dataset = ImageDataset(image)
            model.eval()
            with torch.no_grad():
                all_uvs, _ = dataset.all_pixels(device=device)
                pred = model(all_uvs).cpu().numpy().reshape(H, W, 3)
            results[(L, width)] = np.clip(pred, 0, 1)

    # Plot 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, L in enumerate(L_values):
        for j, width in enumerate(width_values):
            ax = axes[i][j]
            ax.imshow(results[(L, width)])
            ax.set_title(f"L={L}, width={width}")
            ax.axis("off")
    plt.suptitle("2×2 Hyperparameter Grid (1000 iters)")
    plt.tight_layout()
    plt.savefig("hyperparameter_grid.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("Saved hyperparameter grid to hyperparameter_grid.png")


# ─────────────────────────────────────────────────────────────
# MAIN: Run Part 1
# ─────────────────────────────────────────────────────────────
 
def load_image(path: str, size: int = 400) -> np.ndarray:
    """Load an image from a local file path and resize to size x size."""
    img_pil = Image.open(path).convert("RGB")
    img_pil = img_pil.resize((size, size), Image.LANCZOS)
    return np.array(img_pil)
 
 
if __name__ == "__main__":
    # Device detection: cuda > mps (Apple M-series) > cpu
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
 
    # ── Load the provided test image ─────────────────────────
    # Download the fox image once and save it as "fox.jpg" in your project folder:
    #   curl -L "https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg" -o fox.jpg
    # OR drag any .jpg into your project folder and update the path below.
    TEST_IMAGE_PATH = "img/fox.jpg"   # update if your filename differs
 
    print(f"Loading test image from {TEST_IMAGE_PATH}...")
    image = load_image(TEST_IMAGE_PATH, size=400)
    print(f"Test image loaded: {image.shape}")
 
    # ── Part 1 main run on test image ────────────────────────
    print("\n=== Part 1: Training 2D Neural Field (test image) ===")
    model, psnr_history = train_2d_nerf(
        image,
        num_iters=2000,
        batch_size=10_000,
        lr=1e-2,
        L=10,
        width=256,
        num_layers=4,
        device=device,
        save_interval=250,
        title="test_image",
    )
 
    # ── Part 1 on YOUR OWN image ─────────────────────────────
    # Save any image you like to your project folder and update the path below.
    MY_IMAGE_PATH = "img/cat.jpg"   
 
    print(f"\nLoading own image from {MY_IMAGE_PATH}...")
    my_image = load_image(MY_IMAGE_PATH, size=400)
    print(f"Own image loaded: {my_image.shape}")
 
    print("\n=== Part 1: Training 2D Neural Field (own image) ===")
    model2, _ = train_2d_nerf(
        my_image,
        num_iters=2000,
        batch_size=10_000,
        lr=1e-2,
        L=10,
        width=256,
        num_layers=4,
        device=device,
        save_interval=250,
        title="my_image",
    )
 
    # ── Part 1 hyperparameter grid (runs on test image) ──────
    print("\n=== Part 1: Hyperparameter Grid ===")
    run_hyperparameter_grid(image, device=device)