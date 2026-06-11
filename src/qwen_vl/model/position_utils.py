"""
Utility functions for 2D sinusoidal positional embeddings.

WHY THIS FILE EXISTS
--------------------
When we fuse 3D geometry features (from VGGT) into 2D vision features (from the Qwen
ViT), or into language token hidden states in the LM decoder, we use cross-attention
where 2D visual tokens act as queries and 3D geometry tokens act as keys/values.

Without positional information, the cross-attention treats every patch token as
interchangeable — a token at the top-left corner "looks the same" as one at the
bottom-right. This causes the model to mix spatial information incorrectly.

We solve this by adding a 2D sinusoidal positional embedding to every patch token
before cross-attention so the model knows WHERE each patch lives in the image grid.

WHY SINUSOIDAL (not learned)?
Sinusoidal embeddings generalise to image resolutions never seen during training —
important for navigation where the robot may encounter differently-sized images.
Learned embeddings are fixed-size and break on out-of-distribution resolutions.

FORMULA
-------
For a patch at grid position (h, w):

    embed[h, w, 2k]   = sin(h / 10000^(2k / D_half))   for k=0..D/4-1
    embed[h, w, 2k+1] = cos(h / 10000^(2k / D_half))   for k=0..D/4-1
    embed[h, w, D/2 + 2k]   = sin(w / 10000^(2k / D_half))
    embed[h, w, D/2 + 2k+1] = cos(w / 10000^(2k / D_half))

The first half of the embedding encodes the row index h.
The second half encodes the column index w.
Each coordinate is encoded by D/4 sin-cos pairs at increasing frequencies
(from 1/1 to 1/10000), giving the model both low-frequency (coarse) and
high-frequency (fine) spatial signals.
"""

import torch


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    Compute 1D sinusoidal positional embedding for a set of scalar positions.

    Args:
        embed_dim (int):
            Output embedding dimension D.  Must be even.
            Each position is encoded as D/2 sin values followed by D/2 cos values,
            covering frequencies omega_k = 1 / 10000^(2k / D) for k=0..D/2-1.
        pos (torch.Tensor):
            Arbitrary-shape tensor of scalar positions (e.g. row indices [H] or
            column indices [W]).  Will be flattened internally to shape [M].

    Returns:
        torch.Tensor of shape [M, D]:
            Row i contains the sinusoidal embedding for position pos.flatten()[i].

    Dimensions walkthrough (example: D=256, H=37 rows):
        pos       : [37]            — row indices 0, 1, ..., 36
        omega     : [128]           — D/2 = 128 frequencies
        out       : [37, 128]       — outer product pos × omega  (each row × each freq)
        emb_sin   : [37, 128]       — sin of each (position, frequency) pair
        emb_cos   : [37, 128]       — cos of each (position, frequency) pair
        return    : [37, 256]       — concat([sin, cos], dim=1)
    """
    assert embed_dim % 2 == 0, "embed_dim must be even to split into sin/cos halves"

    # Build D/2 frequencies: omega_k = 1 / 10000^(k / (D/2)) for k = 0..D/2-1
    # This gives a geometric sequence of frequencies from 1.0 down to 1/10000.
    # Shape: [D/2]
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0        # normalise k to [0, 1)
    omega = 1.0 / (10000 ** omega)  # omega_k = 1 / 10000^(k/(D/2))  →  [D/2]

    # Flatten positions to 1D: [M]
    pos = pos.flatten()

    # Outer product: each position × each frequency → shape [M, D/2]
    # out[m, k] = pos[m] * omega[k]
    out = torch.einsum("m,d->md", pos, omega)  # [M, D/2]

    # Apply sin and cos to get complementary representations.
    # sin captures phase; cos adds orthogonality so nearby positions differ.
    emb_sin = torch.sin(out)  # [M, D/2]
    emb_cos = torch.cos(out)  # [M, D/2]

    # Concatenate along the feature axis: [M, D]
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb  # [M, D]


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D sinusoidal positional embedding from a pre-built coordinate grid.

    Args:
        embed_dim (int):
            Total embedding dimension D.  Must be even.
            The first D/2 dimensions encode the row (H) coordinate;
            the second D/2 dimensions encode the column (W) coordinate.
        grid (torch.Tensor of shape [2, H, W]):
            grid[0] contains row indices (H axis),
            grid[1] contains column indices (W axis).

    Returns:
        torch.Tensor of shape [H*W, D].

    Dimensions walkthrough (example: H=37, W=37, D=1280):
        grid         : [2, 37, 37]
        emb_h        : [37*37, 640]  — D/2=640 dims encoding row index
        emb_w        : [37*37, 640]  — D/2=640 dims encoding col index
        return       : [37*37, 1280] — concat([emb_h, emb_w], dim=1)

    Why split D evenly between H and W?
        The model can devote equal representational capacity to both spatial axes,
        which is appropriate for the roughly square image patches used here.
    """
    assert embed_dim % 2 == 0

    # Encode row indices (grid[0]) using the first half of embed_dim
    # grid[0] shape: [H, W]  →  get_1d_sincos returns [H*W, D/2]
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # [H*W, D/2]

    # Encode column indices (grid[1]) using the second half
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # [H*W, D/2]

    # Concatenate along feature axis to get [H*W, D]
    emb = torch.cat([emb_h, emb_w], dim=1)  # [H*W, D]
    return emb


def get_2d_sincos_pos_embed(
    height: int,
    width: int,
    embed_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute 2D sinusoidal positional embeddings for an H×W patch grid.

    This is the main entry point used by fusion modules.  It builds the
    integer coordinate grid (row, col) and delegates to
    get_2d_sincos_pos_embed_from_grid.

    Args:
        height (int):
            Number of patch rows (H).  For Qwen2.5-VL with patch_size=14 and
            spatial_merge_size=2: H = image_height // (14 * 2).
        width (int):
            Number of patch columns (W).
        embed_dim (int):
            Embedding dimension.  Should match the hidden dimension of the
            module that will add these embeddings to the patch features.
        device (torch.device):
            Target device (must match the patch tensors that will use these
            embeddings to avoid device-mismatch errors).

    Returns:
        torch.Tensor of shape [H*W, embed_dim]:
            Position embedding for each of the H*W spatial positions.
            Row i*W + j contains the embedding for grid position (i, j).

    Example usage in cross-attention fusion:
        # patch tokens:   [B, H*W, C]
        # pos_embed:       [H*W, C]  →  unsqueeze(0) → [1, H*W, C]
        pos = get_2d_sincos_pos_embed(H, W, C, device)
        query = patch_features_2d + pos.unsqueeze(0)   # broadcast over batch
        key   = geo_features_3d   + pos.unsqueeze(0)
    """
    # Build integer row/col index grids
    # grid_h: [H]  values 0, 1, ..., H-1
    # grid_w: [W]  values 0, 1, ..., W-1
    grid_h = torch.arange(height, dtype=torch.float32, device=device)
    grid_w = torch.arange(width,  dtype=torch.float32, device=device)

    # meshgrid → each is [H, W]
    # grid[0, i, j] = i  (row index)
    # grid[1, i, j] = j  (col index)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")  # tuple of two [H, W] tensors
    grid = torch.stack(grid, dim=0)  # [2, H, W]

    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)  # [H*W, embed_dim]
