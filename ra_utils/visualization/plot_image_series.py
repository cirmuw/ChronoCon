
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime


def plot_patch_series(data_batch, pat_id: str, axes=None):

    # Select entries for that patient
    m = np.array(data_batch["patient_id"]) == pat_id
    pat_ids = np.array(data_batch["patient_id"])[m]
    dates_str = np.array(data_batch["date_str"])[m]
    scores = np.array(data_batch["score"])[m]
    sides = np.array(data_batch["left_or_right"])[m]
    score_types = np.array(data_batch["score_type"])[m]
    roi_names = np.array(data_batch["roi_name"])[m]
    images = [img for (cnt, img) in enumerate(data_batch["img"]) if m[cnt]]

    # Convert dates to datetime objects for sorting
    dates_dt = np.array([datetime.strptime(d, "%Y%m%d") for d in dates_str])
    sorted_indices = np.argsort(dates_dt)

    # Sort everything
    dates_sorted = dates_str[sorted_indices]
    scores_sorted = scores[sorted_indices]
    sides_sorted = sides[sorted_indices]
    score_types_sorted = score_types[sorted_indices]
    roi_names_sorted = roi_names[sorted_indices]
    images_sorted = [images[i] for i in sorted_indices]

    # Separate left and right images
    L_indices = [i for i, side in enumerate(sides_sorted) if side == "L"]
    R_indices = [i for i, side in enumerate(sides_sorted) if side == "R"]

    n_L, n_R = len(L_indices), len(R_indices)
    max_n = max(n_L, n_R)

    fig, axes = plt.subplots(
        2, max_n,
        figsize=(4 * max_n, 10),
        gridspec_kw={"height_ratios": [1, 1]}
    )

    # Ensure axes is 2D
    if max_n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    # Plot left images (row 0)
    for i, idx in enumerate(L_indices):
        ax = axes[0, i]
        ax.imshow(images_sorted[idx].cpu().numpy().squeeze(), cmap="gray")
        ax.set_title(f"Date: {dates_sorted[idx]}\nScore: {scores_sorted[idx]}\nType: {score_types_sorted[idx]}")
        ax.axis('off')
    for j in range(n_L, max_n):
        axes[0, j].axis('off')

    # Plot right images (row 1)
    for i, idx in enumerate(R_indices):
        ax = axes[1, i]
        ax.imshow(images_sorted[idx].cpu().numpy().squeeze(), cmap="gray")
        ax.set_title(f"Date: {dates_sorted[idx]}\nScore: {scores_sorted[idx]}\nType: {score_types_sorted[idx]}")
        ax.axis('off')
    for j in range(n_R, max_n):
        axes[1, j].axis('off')

    # Add row labels "L" and "R"
    fig.text(0.03, 0.73, "L", fontsize=30, va='center', ha='center', weight='bold')
    fig.text(0.03, 0.27, "R", fontsize=30, va='center', ha='center', weight='bold')

    fig.suptitle(f'Patient ID: {pat_id}', fontsize=18)
    plt.subplots_adjust(hspace=0.5, left=0.05)
    plt.show()
    
    
    
    



def plot_triplet_dataloader_examples(
    batch,
    N=5,
    image_keys=("img", "img_pos", "img_neg"),
    text_info_keys=('score', 'score_type', 'JSN_or_ERO', 'extremity',
                    'patient_id', 'date_str', 'left_or_right', 'roi_name'),
    axis=None,
):
    """
    Plot a grid of images from a dataloader batch.

    Parameters
    ----------
    batch : dict
        Mapping of keys -> tensors/arrays/lists. Image tensors are expected
        as (B, C, H, W) or (B, H, W). Text info keys can be tensors, arrays, or lists.
    N : int
        Number of columns (samples).
    image_keys : sequence of str
        Row order of image keys to plot. Missing keys are skipped gracefully.
        Example calls:
          - ("img_original", "img", "img_pos")  # your 3×N example
          - default: ("img", "img_pos", "img_neg")
    text_info_keys : sequence of str
        Keys whose values will be summarized in the *top row* titles for each column.
    axis : None or matplotlib Axes array
        If None, a new figure is created. Otherwise must be shaped (rows, N) or 1D of length rows*N.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes array
    """
    try:
        import torch  # only to type-check tensors
        _has_torch = True
    except Exception:
        _has_torch = False

    # Keep only image keys that actually exist in the batch
    img_keys_present = [k for k in image_keys if k in batch]
    if not img_keys_present:
        raise ValueError(f"No requested image_keys found in batch. Asked for {image_keys}, "
                         f"available: {list(batch.keys())}")

    # Determine how many samples we can safely show across all image rows
    def _blen(x):
        if _has_torch and hasattr(x, "shape"):
            return int(x.shape[0])
        if isinstance(x, np.ndarray):
            return int(x.shape[0])
        return len(x)

    max_cols_from_rows = min(_blen(batch[k]) for k in img_keys_present)
    ncols = min(N, max_cols_from_rows)

    # Prepare axes
    nrows = len(img_keys_present)
    if axis is None:
        fig_w = max(2.4 * ncols, 6.0)     # width in inches
        fig_h = max(2.4 * nrows, 3.0)     # height in inches
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    else:
        # Accept (nrows, ncols) array or flat array
        axes = axis
        if np.ndim(axes) == 1:
            if len(axes) != nrows * ncols:
                raise ValueError("Provided axis length does not match rows*cols.")
            axes = np.asarray(axes).reshape(nrows, ncols)
        elif np.shape(axes) != (nrows, ncols):
            raise ValueError(f"Provided axis has shape {np.shape(axes)}, expected {(nrows, ncols)}.")
        fig = axes.flat[0].figure

    # Helpers
    def _to_numpy_img(x):
        """Convert a single sample to 2D numpy array for imshow."""
        if _has_torch and "torch" in str(type(x)):
            x = x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            x = x
        else:
            x = np.array(x)

        # Accept (H,W) or (C,H,W); squeeze 1-channel if present
        if x.ndim == 3:
            # Assume (C,H,W); if (H,W,C) we try to transpose if C is small
            if x.shape[0] <= 4 and x.shape[0] == 1:
                x = x[0]
            elif x.shape[-1] <= 4 and x.shape[-1] == 1:  # (H,W,1)
                x = x[..., 0]
            elif x.shape[0] <= 4:  # multi-channel; take first for grayscale
                x = x[0]
            else:
                # Fallback: if it's (H,W,C) with big C, just take channel 0
                x = x[..., 0]
        elif x.ndim == 4:
            raise ValueError("Got a 4D tensor for a single sample; index the batch dimension first.")
        return x.astype(np.float32, copy=False)

    def _scale01(img):
        """Min-max per-image scaling to [0,1] for display (robust to constants)."""
        vmin = np.nanmin(img)
        vmax = np.nanmax(img)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return np.zeros_like(img, dtype=np.float32)
        return (img - vmin) / (vmax - vmin)

    def _get_item(val, idx):
        """Extract a scalar/string at idx from arrays/tensors/lists."""
        if _has_torch and "torch" in str(type(val)):
            v = val[idx]
            if v.ndim == 0:
                try:
                    return v.item()
                except Exception:
                    return str(v.detach().cpu().numpy())
            return str(v.detach().cpu().numpy())
        if isinstance(val, np.ndarray):
            v = val[idx]
            if np.isscalar(v):
                return v.item() if hasattr(v, "item") else v
            return str(v)
        if isinstance(val, (list, tuple)):
            v = val[idx]
            return v
        # Fallback for singletons
        return val

    def _format_info(idx):
        parts = []
        for k in text_info_keys or []:
            if k in batch:
                try:
                    v = _get_item(batch[k], idx)
                except Exception:
                    v = "<?>"
                parts.append(f"{k}: {v}")
        s = " | ".join(map(str, parts))
        # keep titles compact
        return s if len(s) <= 160 else (s[:157] + "…")

    # Draw
    for r, key in enumerate(img_keys_present):
        imgs = batch[key]
        for c in range(ncols):
            ax = axes[r, c]
            # grab single sample
            if _has_torch and "torch" in str(type(imgs)):
                sample = imgs[c]
            elif isinstance(imgs, np.ndarray):
                sample = imgs[c]
            else:
                sample = imgs[c]
            # squeeze channel dim if present
            if hasattr(sample, "shape") and len(getattr(sample, "shape")) >= 3:
                # expect (C,H,W); if (1,H,W), squeeze channel
                pass
            img2d = _to_numpy_img(sample)
            ax.imshow(_scale01(img2d), cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(key, rotation=90, fontsize=9)
            if r == 0:
                title = _format_info(c) if text_info_keys else ""
                if title:
                    ax.set_title(title, fontsize=8, loc="left")

    plt.tight_layout()
    return fig, axes
