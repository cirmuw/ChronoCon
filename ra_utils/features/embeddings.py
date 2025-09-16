from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union, Tuple
import numpy as np
import torch
import pandas as pd

@torch.no_grad()
def generate_embeddings(
    model_AE: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    model_c = None, 
    *,
    device: str = "cuda",
    transform=lambda x: x,
    include_keys: Optional[Iterable[str]] = None,
    return_numpy: bool = True,
) -> Dict[str, Dict[str, Union[np.ndarray, List[str], torch.Tensor]]]:
    """
    Run the autoencoder's encoder on each dataloader and collect per-sample metadata.

    Parameters
    ----------
    model_AE
        Your AE whose forward returns (X_pred, z) and accepts (images, score_types).
    dataloaders
        Mapping like {"val": val_loader, "test": test_loader, ...}.
    device
        Torch device for inference.
    transform
        Optional transform to apply to X before feeding the model (mirrors your val loop).
    include_keys
        Extra keys to extract from each batch. If None, a sensible default set is used.
        Any key that is missing in a given batch will just be skipped.
        Defaults to:
            [
                "file_name", "patient_id", "patient_scoretype_key", "date_str",
                "extremity", "JSN_or_ERO", "score_type"
            ]
    return_numpy
        If True, embeddings and tensors are returned as numpy arrays.
        If False, embeddings are returned as CPU torch tensors.

    Returns
    -------
    per_loader : dict
        {
          "<loader_name>": {
            "z": (N, D) array/tensor,
            "score": (N,) array/tensor (int or float),
            "score_type": list[str],
            "patient_scoretype_key": np.ndarray[str] (if present),
            "date_str": np.ndarray[str] (if present),
            "file_name": list[str]/np.ndarray[str] (if present),
            "patient_id": list[str]/np.ndarray[str] (if present),
            "extremity": list[str]/np.ndarray[str] (if present),
            "JSN_or_ERO": list[str]/np.ndarray[str] (if present),
            "loader_name": list[str] (length N, for convenience)
          },
          ...
        }
    """
    # Default extras mirrors what you already use downstream
    if include_keys is None:
        include_keys = [
            "file_name",
            "patient_id",
            "patient_scoretype_key",
            "date_str",
            "extremity",
            "JSN_or_ERO",
            "score_type",
        ]

    model_AE.eval().to(device)

    per_loader: Dict[str, Dict[str, list]] = {}

    for loader_name, loader in dataloaders.items():
        store = defaultdict(list)

        for batch in loader:
            # --- inputs -------------------------------------------------------
            X = batch["img"].to(device)
            s_type = batch["score_type"]                 # list[str] length B
            y = batch["score"].to(device)                # (B,)

            # --- forward ------------------------------------------------------
            X_pred, z = model_AE(transform(X), s_type)   # z: (B, D)

            # --- collect embeddings / targets / meta --------------------------
            if return_numpy:
                z_np = z.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
            else:
                z_np = z.detach().cpu()
                y_np = y.detach().cpu()

            store["z"].append(z_np)
            store["score"].append(y_np)

            # Always collect score_type (string list)
            # It might be a tuple in some datasets — normalize to list[str]
            if isinstance(s_type, (list, tuple)):
                store["score_type"].extend(list(s_type))
            else:
                # If a single scalar slipped through, broadcast to batch size
                B = z.shape[0]
                store["score_type"].extend([str(s_type)] * B)

            # Optional extras (best-effort)
            for k in include_keys:
                if k == "score_type":
                    continue  # already handled
                if k in batch:
                    v = batch[k]
                    # Tidy up types to flat python/np types
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().numpy()
                    # Ensure we extend by samples, not append the whole object
                    if isinstance(v, (list, tuple, np.ndarray)):
                        store[k].extend(list(v))
                    else:
                        # If it's a scalar, broadcast to batch size
                        B = z.shape[0]
                        store[k].extend([v] * B)

        # --- stack everything for this loader --------------------------------
        out = {}
        if len(store["z"]) == 0:
            # empty loader guard
            per_loader[loader_name] = {}
            continue

        # z and score come batched; stack vertically
        if return_numpy:
            out["z"] = np.concatenate(store["z"], axis=0)                 # (N, D)
            out["score"] = np.concatenate(store["score"], axis=0)         # (N,)
        else:
            out["z"] = torch.cat(store["z"], dim=0)                       # (N, D)
            out["score"] = torch.cat(store["score"], dim=0)               # (N,)

        # Strings / misc fields are already extended per-sample
        for k, v in store.items():
            if k in {"z", "score"}:
                continue
            out[k] = np.array(v) if isinstance(v, list) else v

        # Convenience column: where this came from
        N = out["z"].shape[0] if return_numpy else out["z"].shape[0]
        out["loader_name"] = np.array([loader_name] * N)

        per_loader[loader_name] = out

    return per_loader




def embeddings_to_dataframe(per_loader: Dict[str, Dict[str, Union[np.ndarray, torch.Tensor]]]) -> pd.DataFrame:
    """
    Flatten the output of `generate_embeddings` into a single pandas DataFrame.

    The latent dimensions are expanded into columns z_0, z_1, ..., z_{D-1}.
    """
    frames = []
    for loader_name, d in per_loader.items():
        if not d or "z" not in d:
            continue

        # Ensure numpy
        z = d["z"].detach().cpu().numpy() if torch.is_tensor(d["z"]) else np.asarray(d["z"])
        score = d["score"].detach().cpu().numpy() if torch.is_tensor(d["score"]) else np.asarray(d["score"])
        D = z.shape[1]

        # Build base frame
        data = {f"z_{i}": z[:, i] for i in range(D)}
        data["score"] = score

        # Attach known metadata if present
        for key in [
            "score_type", "patient_scoretype_key", "patient_id", "date_str",
            "file_name", "extremity", "JSN_or_ERO", "loader_name"
        ]:
            if key in d:
                data[key] = d[key]

        frames.append(pd.DataFrame(data))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
