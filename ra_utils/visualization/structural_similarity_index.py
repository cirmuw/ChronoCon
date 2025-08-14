import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from torchmetrics.functional import structural_similarity_index_measure as ssim

def ssim_matrix_torchmetrics(imgs: torch.Tensor,
                             data_range: float | None = None,
                             chunk_size: int | None = None,
                             **ssim_kwargs) -> torch.Tensor:
    """
    imgs: (Nt, 1, H, W) float tensor, ideally in [0,1]
    returns: (Nt, Nt) SSIM matrix
    """
    assert imgs.ndim == 4 and imgs.size(1) == 1, "imgs must be (Nt,1,H,W)"
    imgs = imgs.float()
    Nt = imgs.size(0)
    device = imgs.device

    # single scalar data_range expected
    if data_range is None:
        rng = imgs.max() - imgs.min()
        data_range = float(rng.detach().item()) if rng > 0 else 1.0

    # force no reduction to get one SSIM per pair
    ssim_kwargs = dict(ssim_kwargs)
    ssim_kwargs.setdefault("reduction", "none")

    if chunk_size is None:
        # Full pairwise (fastest, more memory)
        A = imgs.repeat_interleave(Nt, dim=0)   # (Nt*Nt, 1, H, W)
        B = imgs.repeat(Nt, 1, 1, 1)            # (Nt*Nt, 1, H, W)
        vals = ssim(A, B, data_range=data_range, **ssim_kwargs)  # (Nt*Nt,)
        M = vals.view(Nt, Nt)
    else:
        # Memory-friendly chunked version
        M = torch.empty((Nt, Nt), device=device, dtype=imgs.dtype)
        for i in range(0, Nt, chunk_size):
            i2 = min(i + chunk_size, Nt)
            A = imgs[i:i2].repeat_interleave(Nt, dim=0)   # ((i2-i)*Nt, 1, H, W)
            B = imgs.repeat(i2 - i, 1, 1, 1)              # ((i2-i)*Nt, 1, H, W)
            vals = ssim(A, B, data_range=data_range, **ssim_kwargs)  # ((i2-i)*Nt,)
            M[i:i2] = vals.view(i2 - i, Nt)

    # Numerical niceties (optional)
    M = (M + M.T) / 2  # enforce symmetry
    M.fill_diagonal_(1.0)
    return M





def compute_and_plot_ssi(data_batch, pat_id,
                         date_fmt_in="%Y%m%d",
                         show_dates_fmt="%Y-%m-%d",
                         chunk_size: int | None = None,
                         data_range: float | None = 1.0,
                         ssim_kwargs=None):
    """
    Build pairwise SSIM heatmaps (L and R) over time for a given patient.

    Expects data_batch with keys:
      - "patient_id", "date_str", "left_or_right", "roi_name", "score", "score_type"
      - "img": list/array of tensors shaped (1,H,W) or (C,H,W) with C=1

    Args:
      pat_id: patient identifier to filter
      date_fmt_in: format of data_batch["date_str"] (default '%Y%m%d')
      show_dates_fmt: how to show dates on axes (default '%Y-%m-%d')
      chunk_size: memory-friendly chunking for SSIM (None = full vectorized)
      data_range: SSIM data_range (set None to auto)
      ssim_kwargs: extra kwargs for torchmetrics SSIM (e.g., kernel_size=11, sigma=1.5)

    Returns:
      dict with keys {'L': (ssim_matrix, labels), 'R': (ssim_matrix, labels)}
    """
    ssim_kwargs = ssim_kwargs or {}

    # Mask and slice all patient entries
    pids = np.asarray(data_batch["patient_id"])
    m = (pids == pat_id)
    if not np.any(m):
        raise ValueError(f"No entries found for patient_id={pat_id!r}")

    dates_str = np.asarray(data_batch["date_str"])[m]
    sides = np.asarray(data_batch["left_or_right"])[m]
    # optional extras if you need them later
    # scores = np.asarray(data_batch["score"])[m]
    # score_types = np.asarray(data_batch["score_type"])[m]
    # roi_names = np.asarray(data_batch["roi_name"])[m]
    imgs_list = [img for (i, img) in enumerate(data_batch["img"]) if m[i]]

    # Parse and sort by date
    dates_dt = np.array([datetime.strptime(d, date_fmt_in) for d in dates_str])
    order = np.argsort(dates_dt)
    dates_dt = dates_dt[order]
    sides = sides[order]
    imgs_list = [imgs_list[i] for i in order]

    # Format labels
    labels = [dt.strftime(show_dates_fmt) for dt in dates_dt]

    # Split indices by side
    idx_L = [i for i, s in enumerate(sides) if s == "L"]
    idx_R = [i for i, s in enumerate(sides) if s == "R"]

    # Helper to stack (Nt,1,H,W)
    def _stack(imgs_sel):
        if len(imgs_sel) == 0:
            return None
        t = torch.stack([im if im.ndim == 3 else im.unsqueeze(0) for im in imgs_sel], dim=0)
        if t.size(1) != 1:
            # if (H,W) given, add channel dim; if (C,H,W) with C>1, select first channel
            t = t[:, :1, ...]
        return t

    imgs_L = _stack([imgs_list[i] for i in idx_L])
    imgs_R = _stack([imgs_list[i] for i in idx_R])
    labels_L = [labels[i] for i in idx_L]
    labels_R = [labels[i] for i in idx_R]

    # Compute SSIM matrices
    out = {}
    if imgs_L is not None and imgs_L.size(0) >= 2:
        M_L = ssim_matrix_torchmetrics(imgs_L, data_range=data_range,
                                        chunk_size=chunk_size, **ssim_kwargs).cpu().numpy()
        out["L"] = (M_L, labels_L)
    else:
        out["L"] = (None, labels_L)

    if imgs_R is not None and imgs_R.size(0) >= 2:
        M_R = ssim_matrix_torchmetrics(imgs_R, data_range=data_range,
                                        chunk_size=chunk_size, **ssim_kwargs).cpu().numpy()
        out["R"] = (M_R, labels_R)
    else:
        out["R"] = (None, labels_R)

    # ---- Plot: 2x1 (side-by-side) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), squeeze=False)
    axes = axes[0]

    for ax, side in zip(axes, ["L", "R"]):
        M, lab = out[side]
        if M is None:
            ax.text(0.5, 0.5, f"No/insufficient {side} images", ha="center", va="center")
            ax.set_axis_off()
            continue
        hm = sns.heatmap(M, annot=True, fmt=".2f", cmap="coolwarm",
                         cbar_kws={"label": "SSIM"}, square=True,
                         xticklabels=lab, yticklabels=lab, ax=ax)
        ax.set_title(f"Patient {pat_id} — {side}", fontsize=13)
        ax.set_xlabel("Time")
        ax.set_ylabel("Time")
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    fig.suptitle(f"Pairwise SSIM over time — Patient {pat_id}", fontsize=15)
    plt.tight_layout()
    plt.show()

    return out
