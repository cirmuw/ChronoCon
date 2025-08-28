from typing import Dict, Any, Optional, Tuple, List

def _norm_side(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in {"LEFT", "L"}:  return "L"
    if s in {"RIGHT", "R"}: return "R"
    return s

def _norm_str(x: Any) -> str:
    return str(x).strip()

def _matches(record: Any, query: Dict[str, Any]) -> bool:
    """
    record: a dict-like row/item (from dataset.data)
    query: normalized dict of key->value to match
    """
    def _get(rec, key):
        if isinstance(rec, dict):
            return rec.get(key, None)
        return getattr(rec, key, None)

    for k, v in query.items():
        rv = _get(record, k)
        # laterality/side handling
        if k.lower() in {"left_or_right", "side", "laterality"}:
            if _norm_side(rv) != _norm_side(v):
                return False
        else:
            if _norm_str(rv) != _norm_str(v):
                return False
    return True

def _coerce_key(query_key: str, available_keys: List[str]) -> str:
    """
    Try to map user key to a column present in dataset.data
    (handles common aliases for laterality).
    """
    if query_key in available_keys:
        return query_key
    # handle typical laterality aliases
    if query_key.lower() in {"left_or_right", "side", "laterality"}:
        for alt in ["left_or_right", "side", "laterality", "LR", "LeftRight", "L_or_R"]:
            if alt in available_keys:
                return alt
    # best effort: return original; mismatch will be handled by _matches
    return query_key

def find_case_from_dataloader(
    dl,
    case_data: Dict[str, Any],
    key_map: Optional[Dict[str, str]] = None,
    expect_unique: bool = True,
) -> Dict[str, Any]:
    """
    Find a single case by matching metadata in `dl.dataset.data`, then return the
    *transformed* sample via `dl.dataset[idx]`.

    Parameters
    ----------
    dl : torch.utils.data.DataLoader
        Your existing dataloader (e.g., val_loaders["H_JSN_PIPIII"]).
    case_data : dict
        e.g., {"patient_id": "600", "date_str": "20070223", "left_or_right": "L"}
    key_map : dict, optional
        Optional map from query keys -> dataset.data keys if they differ.
        e.g., {"left_or_right": "side"} if your dataset uses "side".
    expect_unique : bool
        If True, warn when multiple matches and return the first.

    Returns
    -------
    out : dict
        {"idx": int, "meta": dict-like raw row from dataset.data, "sample": transformed_sample}
    """
    ds = dl.dataset
    if not hasattr(ds, "data"):
        raise AttributeError("This dataset has no `.data` attribute. Provide a dataset with accessible metadata.")

    data = ds.data
    # Normalize query and align keys to what's available in dataset.data
    # Support DataFrame or list-of-dicts/records.
    try:
        import pandas as pd
        is_df = isinstance(data, pd.DataFrame)
    except Exception:
        is_df = False

    # Collect available keys/columns
    if is_df:
        available_keys = list(data.columns)
    else:
        if isinstance(data, (list, tuple)) and len(data) > 0:
            first = data[0]
            if isinstance(first, dict):
                available_keys = list(first.keys())
            else:
                # fallback: try dir()-ish attributes
                available_keys = [k for k in dir(first) if not k.startswith("_")]
        else:
            available_keys = []

    key_map = key_map or {}
    # Build a normalized query dict matching the dataset's keys
    query = {}
    for k, v in case_data.items():
        # allow explicit key_map overrides first
        mapped = key_map.get(k, k)
        # if still not present, try common aliases
        mapped = _coerce_key(mapped, available_keys) if available_keys else mapped
        query[mapped] = v

    # Find matching index/indices
    matches: List[int] = []
    if is_df:
        df = data.copy()
        # Normalize columns for robust matching
        for col in df.columns:
            if col.lower() in {"left_or_right", "side", "laterality"}:
                df[col] = df[col].astype(str).str.strip().str.upper().replace({"LEFT": "L", "RIGHT": "R"})
            else:
                df[col] = df[col].astype(str).str.strip()

        mask = None
        for k, v in query.items():
            if k not in df.columns:
                continue
            colmask = (df[k] == (_norm_side(v) if k.lower() in {"left_or_right", "side", "laterality"} else _norm_str(v)))
            mask = colmask if mask is None else (mask & colmask)
        if mask is None:
            raise ValueError("None of the query keys matched dataset.data columns.")
        matches = df.index[mask].tolist()
    else:
        if not isinstance(data, (list, tuple)):
            raise TypeError("dataset.data is neither a DataFrame nor a list/tuple of records.")
        for i, rec in enumerate(data):
            if _matches(rec, query):
                matches.append(i)

    if not matches:
        raise LookupError(f"No case matched query={case_data} (resolved keys: {query}).")

    if expect_unique and len(matches) > 1:
        # Not fatal; we just take the first match
        print(f"[find_case_from_dataloader] Warning: {len(matches)} matches found, returning the first.")

    idx = matches[0]
    transformed_sample = ds[idx]  # <-- applies dataset transforms
    raw_meta = data.iloc[idx].to_dict() if is_df else data[idx]

    return {"idx": idx, "meta": raw_meta, "sample": transformed_sample}


# Code to be refactored if it works... 
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Literal
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Helpers
# -------------------------------
def _reduce_channels(t: torch.Tensor, how: Literal["mean","max","l2"] = "mean") -> torch.Tensor:
    """
    t: [B, C, H, W] -> returns [B, 1, H, W]
    """
    if how == "mean":
        r = t.mean(dim=1, keepdim=True)
    elif how == "max":
        r, _ = t.max(dim=1, keepdim=True)
    elif how == "l2":
        r = torch.norm(t, dim=1, keepdim=True)
    else:
        raise ValueError(f"Unknown reduction: {how}")
    return r

def _minmax_norm(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-image min-max over HxW (preserves batch & channel dims).
    Input: [B,1,H,W]  Output: [B,1,H,W] in [0,1]
    """
    B, C, H, W = t.shape
    t_ = t.view(B, C, -1)
    t_min = t_.min(dim=-1, keepdim=True).values
    t_max = t_.max(dim=-1, keepdim=True).values
    t_norm = (t_ - t_min) / (t_max - t_min + eps)
    return t_norm.view(B, C, H, W)

def _to_numpy_image(img: torch.Tensor) -> np.ndarray:
    """
    img: [C,H,W] tensor in any range; returns displayable HxW or HxWx3 float in [0,1].
    Handles 1-channel and 3-channel.
    """
    img = img.detach().cpu().float()
    if img.ndim != 3:
        raise ValueError("Expected image tensor of shape [C,H,W].")
    C, H, W = img.shape
    if C == 1:
        arr = img[0]
        # normalize to [0,1] for display
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin)
        else:
            arr = torch.zeros_like(arr)
        return arr.numpy()
    elif C == 3:
        arr = img
        # per-channel min-max, then clip
        vmin = arr.view(3, -1).min(dim=1).values[:, None, None]
        vmax = arr.view(3, -1).max(dim=1).values[:, None, None]
        arr = (arr - vmin) / (vmax - vmin + 1e-8)
        return arr.permute(1,2,0).numpy()
    else:
        # take first 3 channels if more (rare here)
        arr = img[:3]
        vmin = arr.view(3, -1).min(dim=1).values[:, None, None]
        vmax = arr.view(3, -1).max(dim=1).values[:, None, None]
        arr = (arr - vmin) / (vmax - vmin + 1e-8)
        return arr.permute(1,2,0).numpy()

def _overlay(ax, base_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45, cmap: str = "inferno"):
    """
    base_img: HxW or HxWx3 in [0,1]; heatmap: HxW in [0,1]
    """
    ax.imshow(base_img, interpolation="nearest")
    ax.imshow(heatmap, interpolation="nearest", alpha=alpha, cmap=cmap)
    ax.axis("off")

# -------------------------------
# Hook registration
# -------------------------------
class MTANAttentionTap:
    """
    Context manager to capture attention masks and (optionally) top feature map from your MTANReconv4 during forward().
    """
    def __init__(self, model, capture_feature_map: bool = True):
        self.model = model
        self.capture_feature_map = capture_feature_map
        self.handles = []
        self.captured: Dict[str, List[torch.Tensor]] = {
            "att1": [], "att2": [], "att3": [], "att4": [], "u4": []
        }

    def __enter__(self):
        # Register hooks on every task-specific attention module; only the active one will fire.
        for m in self.model.encoder_att_1.values():
            self.handles.append(m.register_forward_hook(self._make_hook("att1")))
        for m in self.model.encoder_att_2.values():
            self.handles.append(m.register_forward_hook(self._make_hook("att2")))
        for m in self.model.encoder_att_3.values():
            self.handles.append(m.register_forward_hook(self._make_hook("att3")))
        for m in self.model.encoder_att_4.values():
            self.handles.append(m.register_forward_hook(self._make_hook("att4")))

        if self.capture_feature_map:
            # Capture output of the last shared residual block (u_4_t)
            self.handles.append(self.model.shared_layer4_t.register_forward_hook(self._make_hook("u4")))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _make_hook(self, key: str):
        def hook(mod, inp, out):
            # out is attention mask for att* (shape [B, C, H, W])
            # or feature map for u4 (shape [B, C, H, W])
            self.captured[key].append(out.detach())
        return hook



###----

@torch.no_grad()
def visualize_mtan_attention(
    model,
    x: torch.Tensor,
    score_type: List[str],
    *,
    reduction: Literal["mean","max","l2"] = "mean",
    upsample_mode: Literal["bilinear","nearest"] = "bilinear",
    overlay_alpha: float = 0.45,
    fig_cols: int = 5,
    title_prefix: str = "",
    capture_feature_map: bool = True,
    return_arrays: bool = False,
):
    """
    Run a forward pass with hooks to capture MTAN attention masks (levels 1..4),
    upsample them to input size, and overlay on the original images.

    Parameters
    ----------
    model : MTANReconv4
    x : torch.Tensor
        Shape [B, C, H, W]. Works with 1 or 3 channels.
    score_type : list[str]
        Must match batch length and your model's expectation (same active path for all items).
    reduction : {'mean','max','l2'}
        Reduce per-channel attention to a single heatmap per layer.
    upsample_mode : {'bilinear','nearest'}
    overlay_alpha : float
    fig_cols : int
        Number of columns in the output grid (Original + Att1 + Att2 + Att3 + Att4 = 5 by default).
    title_prefix : str
        Optional string to prepend in subplot titles.
    capture_feature_map : bool
        If True, also capture u_4_t and return a reduced+upsampled map (not overlaid by default).
    return_arrays : bool
        If True, returns dict of numpy arrays for upsampled attention maps per layer.

    Returns
    -------
    fig : matplotlib.figure.Figure
    arrays (optional) : Dict[str, np.ndarray]
        Each: shape [B, H, W] in [0,1] for 'att1'..'att4' (and 'u4' if requested).
    """
    model_was_training = model.training
    model.eval()

    B, C, H, W = x.shape
    device = next(model.parameters()).device
    x = x.to(device)

    with MTANAttentionTap(model, capture_feature_map=capture_feature_map) as tap:
        _ = model(x, score_type=score_type)  # normal forward

        def _pick_last(key):
            return tap.captured[key][-1] if len(tap.captured[key]) > 0 else None

        att1 = _pick_last("att1")
        att2 = _pick_last("att2")
        att3 = _pick_last("att3")
        att4 = _pick_last("att4")
        u4   = _pick_last("u4") if capture_feature_map else None

    # Reduce channel dimension -> [B,1,h,w]
    att_maps = {}
    for name, t in [("att1", att1), ("att2", att2), ("att3", att3), ("att4", att4)]:
        if t is not None:
            r = _reduce_channels(t, how=reduction)           # [B,1,h,w]
            up = F.interpolate(r, size=(H, W), mode=upsample_mode, align_corners=False)  # [B,1,H,W]
            att_maps[name] = _minmax_norm(up)                # [B,1,H,W] in [0,1]
        else:
            att_maps[name] = None

    u4_map = None
    if capture_feature_map and (u4 is not None):
        r = _reduce_channels(u4, how=reduction)
        up = F.interpolate(r, size=(H, W), mode=upsample_mode, align_corners=False)
        u4_map = _minmax_norm(up)

    # Build figure: columns = Original + 4 attention overlays
    cols = fig_cols
    rows = B
    fig, axes = plt.subplots(rows, cols, figsize=(3.0*cols, 3.2*rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)  # [1, cols]

    col_titles = ["Original", "Att-1", "Att-2", "Att-3", "Att-4"]

    for b in range(B):
        base_img = _to_numpy_image(x[b])
        for c_idx, name in enumerate(["original", "att1", "att2", "att3", "att4"]):
            ax = axes[b, c_idx]
            if name == "original":
                ax.imshow(base_img, interpolation="nearest")
                ax.axis("off")
                ttl = f"{title_prefix}Original"
            else:
                amap = att_maps[name]
                if amap is None:
                    ax.imshow(base_img, interpolation="nearest")
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, color="w")
                    ttl = f"{title_prefix}{name.upper()} (N/A)"
                    ax.axis("off")
                else:
                    heat = amap[b, 0].detach().cpu().numpy()
                    _overlay(ax, base_img, heat, alpha=overlay_alpha, cmap="inferno")
                    ttl = f"{title_prefix}{name.upper()}"
            if b == 0:  # top row: add more readable titles
                ax.set_title(ttl, fontsize=11)

    plt.tight_layout()

    if model_was_training:
        model.train()

    if return_arrays:
        out_np = {}
        for k, v in att_maps.items():
            if v is not None:
                out_np[k] = v.detach().cpu().numpy()[:, 0, ...]  # [B,H,W]
            else:
                out_np[k] = None
        if u4_map is not None:
            out_np["u4"] = u4_map.detach().cpu().numpy()[:, 0, ...]
        else:
            out_np["u4"] = None
        return fig, out_np
    return fig


def plot_fn(dataloader, case_data, model_AE, device="cuda"):
    result = find_case_from_dataloader(dataloader, case_data)
    idx     = result["idx"]
    meta    = result["meta"]       # raw row from dl.dataset.data
    sample  = result["sample"]     # TRANSFORMED sample (what you want)

    model = model_AE.auto_encoder.eval()
    preprocessor = model_AE.preprocessor.to(device)
    imgs, score_types = sample["img"].unsqueeze(0), [sample["score_type"]]  # imgs: [B,C,H,W], score_types: list[str] len=B

    preprocessor.to(device)
    model.to(device)
    imgs.to(device)

    imgs = preprocessor(imgs)

    model.to(device)
    imgs.to(device)

    fig, arrays = visualize_mtan_attention(
        model,
        imgs,
        score_type=score_types,
        upsample_mode="bilinear",
        overlay_alpha=0.3,
        title_prefix="MTAN ",
        capture_feature_map=True,
        return_arrays=True,
    )
    plt.show()


def case_data_by_index(idx: int, data : list,  keys = ["patient_id","date_str", "left_or_right"]):
    r = {k: v for k,v in data[idx].items() if k in keys}
    return r

