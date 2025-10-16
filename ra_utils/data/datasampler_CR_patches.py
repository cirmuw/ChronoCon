from collections import defaultdict
import random
from torch.utils.data import Sampler, DataLoader, BatchSampler



# ---------------------------------------------------------------------
#  Samplers
# ---------------------------------------------------------------------
from collections import defaultdict
import random, math
from torch.utils.data import Sampler, DataLoader, WeightedRandomSampler
import torch



# utils.py  (or wherever your samplers live)
from collections import defaultdict
import random, math
from torch.utils.data import Sampler
import numpy as np


class WeightedPatientBatchSampler(Sampler):
    """
    • Draws patients with replacement according to `patient_weights`.
    • Fills a batch with as many images of that patient as possible,
      but tops up with other patients so the batch has `batch_size`
      elements (except possibly the final batch if drop_last=False).
    """

    def __init__(self, dataset, patient_weights, batch_size, drop_last=False):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.drop_last  = drop_last

        # patient_id → [idx, idx, …]
        self.patient2idx = defaultdict(list)
        for idx, item in enumerate(dataset.data):
            self.patient2idx[item["patient_id"]].append(idx)

        self.patient_ids = list(self.patient2idx.keys())
        self.weights     = np.array(
            [max(patient_weights.get(pid, 1.0), 0.0) for pid in self.patient_ids],
            dtype=float
        )
        # guarantee the weights sum > 0
        if self.weights.sum() == 0:
            self.weights[:] = 1.0

    # ------------------------------------------------------------------
    def __iter__(self):
        # fresh copy & shuffle inside each patient every epoch
        buckets = {p: idxs.copy() for p, idxs in self.patient2idx.items()}
        for idxs in buckets.values():
            random.shuffle(idxs)

        current_batch = []

        while any(buckets.values()):
            # (re)fill batch until we hit batch_size
            if len(current_batch) < self.batch_size:
                # sample a patient with replacement, weighted
                pid = random.choices(self.patient_ids, weights=self.weights, k=1)[0]
                if not buckets[pid]:
                    continue  # this patient is empty; pick again

                n_take = min(len(buckets[pid]),
                             self.batch_size - len(current_batch))
                current_batch.extend(buckets[pid][: n_take])
                del buckets[pid][: n_take]

            # emit if full
            if len(current_batch) == self.batch_size:
                yield current_batch
                current_batch = []

        # leftovers
        if current_batch and not self.drop_last:
            yield current_batch

    # ------------------------------------------------------------------
    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)



#### GroupedWeightedRandomSampler
import math
import random
from collections import defaultdict, Counter
from typing import List, Dict, Sequence, Tuple, Optional, Iterable, Callable, Union

import numpy as np
import torch
from torch.utils.data import BatchSampler


def _ceil_mean(arr: Sequence[float]) -> int:
    """ceil(mean(arr)) as an int; empty -> 0"""
    return 0 if not arr else int(math.ceil(float(np.mean(arr))))


def _build_group_key(row: Dict, group_identifiers: Sequence[str]) -> Tuple:
    """Make a hashable group key from row fields."""
    return tuple(row[g] for g in group_identifiers)


class GroupedWeightedRandomBatchSampler(BatchSampler):
    """
    Mostly-homogeneous batches with label-balanced group selection and graceful top-up.

    - Picks a PRIMARY group each batch (weighted by inverse-frequency of an aggregated label).
    - Tries to fill >= max(ceil(min_primary_fraction*B), min_primary_items) from the primary group.
    - If primary can't fill the batch, tops up from other groups (weighted), capped by
      `max_groups_per_batch` distinct groups per batch.
    - When `replacement=True`, can re-use samples (optionally capped via `repeats_per_group_cap`)
      to reach the desired primary fraction/size.

    Assumes dataset has attribute `.data` which is a list[dict] of metadata rows containing:
      - the `group_identifiers` fields (e.g. "patient_id", "score_type", "left_or_right")
      - the `target_col` (e.g. "score") to aggregate for label balancing

    Args
    ----
    dataset: any object with `.data: list[dict]` aligned to indexing
    batch_size: int > 0
    group_identifiers: fields defining a group
    target_col: column used to build labels for balancing (default "score")
    group_score_agg: "mean_ceil" or a callable(list[float]) -> int
    replacement: allow sampling within a group with replacement when its pool empties
    drop_last: drop final incomplete batch
    seed: RNG seed
    min_primary_fraction: target fraction from primary group (0..1)
    max_groups_per_batch: max distinct groups allowed in a single batch (>=1)
    min_primary_items: hard minimum #items from primary (overrides fraction if larger); None to disable
    repeats_per_group_cap: when replacement=True, limit per-sample index repeats in a single batch;
                           None means no explicit cap (other than batch_size)
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        group_identifiers: Sequence[str],
        target_col: str = "score",
        group_score_agg: Union[str, Callable[[List[float]], int]] = "mean_ceil",
        replacement: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        min_primary_fraction: float = 0.7,
        max_groups_per_batch: int = 3,
        min_primary_items: Optional[int] = None,
        repeats_per_group_cap: Optional[int] = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not (0.0 <= min_primary_fraction <= 1.0):
            raise ValueError("min_primary_fraction must be in [0, 1]")
        if max_groups_per_batch < 1:
            raise ValueError("max_groups_per_batch must be >= 1")

        self.dataset = dataset
        self.data = getattr(dataset, "data", None)
        if self.data is None:
            raise ValueError("dataset must have attribute `.data` (list of dicts)")

        self.batch_size = int(batch_size)
        self.group_identifiers = list(group_identifiers)
        self.target_col = target_col
        self.replacement = bool(replacement)
        self.drop_last = bool(drop_last)
        self.min_primary_fraction = float(min_primary_fraction)
        self.max_groups_per_batch = int(max_groups_per_batch)
        self.min_primary_items = None if min_primary_items is None else int(min_primary_items)
        self.repeats_per_group_cap = None if repeats_per_group_cap is None else int(repeats_per_group_cap)
        self.rng = random.Random(int(seed))

        # 1) Build groups: key -> list of dataset indices
        groups: Dict[Tuple, List[int]] = defaultdict(list)
        for idx, row in enumerate(self.data):
            key = _build_group_key(row, self.group_identifiers)
            groups[key].append(idx)
        self.groups: Dict[Tuple, List[int]] = dict(groups)

        if len(self.groups) == 0:
            raise ValueError("No groups found — check `group_identifiers` and dataset.data")

        # 2) Determine group labels for balancing (via aggregated target)
        if callable(group_score_agg):
            agg_fn = group_score_agg
        else:
            if group_score_agg != "mean_ceil":
                raise ValueError(f"Unknown group_score_agg='{group_score_agg}' (use 'mean_ceil' or a callable)")
            agg_fn = _ceil_mean

        self.group_label: Dict[Tuple, int] = {}
        for key, idxs in self.groups.items():
            vals = [float(self.data[i][self.target_col]) for i in idxs]
            self.group_label[key] = int(agg_fn(vals))

        # 3) Inverse-frequency weighting over group labels -> per-group weight
        label_counts = Counter(self.group_label.values())
        # Guard against divide-by-zero (shouldn't happen, but be safe)
        class_weight = {lab: (1.0 / c if c > 0 else 0.0) for lab, c in label_counts.items()}
        self.group_weight = {k: float(class_weight[self.group_label[k]]) for k in self.groups}

        # 4) Pools for within-group draws (for without-replacement behavior)
        self._pools: Dict[Tuple, List[int]] = {}
        for key, idxs in self.groups.items():
            pool = idxs.copy()
            self.rng.shuffle(pool)
            self._pools[key] = pool

        # 5) Cache keys/probs for weighted picks
        self._group_keys: List[Tuple] = list(self.groups.keys())
        self._group_probs: List[float] = self._normalize([self.group_weight[k] for k in self._group_keys])

        # 6) Length estimate (approx): number of full batches; if not drop_last, add one if remainder
        total = len(self.data)
        self._len = total // self.batch_size
        if not self.drop_last and (total % self.batch_size):
            self._len += 1

    @staticmethod
    def _normalize(weights: List[float]) -> List[float]:
        s = float(sum(weights))
        return [1.0 / len(weights)] * len(weights) if s <= 0 else [w / s for w in weights]

    def _pick_group_weighted(self, exclude: Optional[set] = None) -> Optional[Tuple]:
        keys, probs = [], []
        for k, p in zip(self._group_keys, self._group_probs):
            if exclude and k in exclude:
                continue
            keys.append(k)
            probs.append(p)
        if not keys:
            return None
        # simple roulette wheel
        r = self.rng.random()
        acc = 0.0
        for k, p in zip(keys, self._normalize(probs)):
            acc += p
            if r <= acc:
                return k
        return keys[-1]

    def _draw_from_group(
        self,
        key: Tuple,
        need: int,
        per_batch_repeat_counter: Optional[Dict[int, int]] = None,
    ) -> List[int]:
        """
        Draw up to `need` indices from group `key`.
        - Without replacement: pop from shuffled pool until empty.
        - With replacement: when pool empties, refill (reshuffle). If `repeats_per_group_cap`
          is set, avoid drawing an index more than that many times within the *current* batch.
        """
        if need <= 0:
            return []

        drawn: List[int] = []
        pool = self._pools[key]

        def can_use(idx: int) -> bool:
            if not self.replacement:
                return True
            if self.repeats_per_group_cap is None or per_batch_repeat_counter is None:
                return True
            return per_batch_repeat_counter.get(idx, 0) < self.repeats_per_group_cap

        # To avoid potential infinite loops with replacement+caps, bound attempts
        max_attempts = max(50, 5 * need)

        attempts = 0
        while need > 0 and attempts < max_attempts:
            attempts += 1

            if pool:
                candidate = pool.pop()
                if can_use(candidate):
                    drawn.append(candidate)
                    need -= 1
                    if per_batch_repeat_counter is not None:
                        per_batch_repeat_counter[candidate] = per_batch_repeat_counter.get(candidate, 0) + 1
                # if can't use (cap hit), skip; continue to try others
            else:
                if self.replacement:
                    # refill the pool by reshuffling the full group
                    pool[:] = self.groups[key]
                    self.rng.shuffle(pool)
                else:
                    break  # can't draw more from this group without replacement

        return drawn

    def __iter__(self) -> Iterable[List[int]]:
        produced = 0
        target_total = self._len * self.batch_size if self.drop_last else len(self.data)

        while produced < target_total:
            # Per-batch repeat counter (only used when replacement=True and cap provided)
            per_batch_repeat_counter: Optional[Dict[int, int]] = {} if (self.replacement and self.repeats_per_group_cap is not None) else None

            # 1) Pick primary group
            primary = self._pick_group_weighted()
            if primary is None:
                break

            batch: List[int] = []
            groups_used = {primary}
            need = self.batch_size

            # desired minimum from primary (fraction and/or hard floor)
            want_primary = int(math.ceil(self.min_primary_fraction * self.batch_size))
            if self.min_primary_items is not None:
                want_primary = max(want_primary, int(self.min_primary_items))

            # 2) Draw from primary first
            primary_draw = self._draw_from_group(primary, min(need, want_primary), per_batch_repeat_counter)
            batch.extend(primary_draw)
            need -= len(primary_draw)

            # If we still haven't reached want_primary and replacement is allowed, try to top up more from primary
            if self.replacement and len(batch) < want_primary and need > 0:
                extra = self._draw_from_group(primary, min(need, want_primary - len(batch)), per_batch_repeat_counter)
                batch.extend(extra)
                need -= len(extra)

            # 3) Top-up from other groups (respect max_groups_per_batch)
            attempts = 0
            while need > 0 and len(groups_used) < self.max_groups_per_batch and attempts < 10 * self.max_groups_per_batch:
                attempts += 1
                g = self._pick_group_weighted(exclude=groups_used)
                if g is None:
                    break
                draw = self._draw_from_group(g, need, per_batch_repeat_counter)
                if draw:
                    batch.extend(draw)
                    need -= len(draw)
                    groups_used.add(g)
                else:
                    # couldn't draw from this group (e.g., empty w/o replacement) -> skip it
                    groups_used.add(g)

            # 4) If still short and replacement is True, allow draws from any group to fill
            attempts = 0
            while need > 0 and self.replacement and attempts < 20:
                attempts += 1
                g = self._pick_group_weighted()
                if g is None:
                    break
                draw = self._draw_from_group(g, need, per_batch_repeat_counter)
                if draw:
                    batch.extend(draw)
                    need -= len(draw)

            # 5) Yield or drop
            if len(batch) == self.batch_size:
                yield batch
                produced += self.batch_size
            else:
                if not self.drop_last and batch:
                    yield batch
                    produced += len(batch)
                # else: drop incomplete and continue

    def __len__(self) -> int:
        return self._len


### Sampler in two stages. 
# Stage 1: uniform or mostly uniform regarding score types 
#  -> e.g. PIPII_JSN, PIPII_ERO, PIPIII_JSN, ... ) should appear on average equally often. 
# Stage 2: Oversample on score values (within the same score type)  
#  -> frequency of JSN=4 should be equally often JSN=0



import math
import random
from collections import defaultdict, Counter
from typing import Dict, List, Sequence, Tuple, Optional, Iterable, Union, Callable

import numpy as np
from torch.utils.data import BatchSampler

def _build_group_key(row: Dict, cols: Sequence[str]) -> Tuple:
    return tuple(row[c] for c in cols)

def _ceil_mean(vals: List[float]) -> int:
    if not vals:
        return 0
    return int(math.ceil(float(np.mean(vals))))

def _roulette_idx(rng: random.Random, weights: List[float]) -> int:
    s = float(sum(weights))
    if s <= 0 or not np.isfinite(s):
        return rng.randrange(len(weights))
    r = rng.random() * s
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1

class GroupLevelWeightedBatchSampler(BatchSampler):
    """
    Batch sampler that:
      • Forms groups by `group_identifiers` (e.g. ["patient_id","score_type","left_or_right"]).
      • Computes a **group label** from `target_col` via `group_score_agg` (default: ceil(mean(score))).
      • **Oversamples groups** according to label-level weights (e.g., inverse frequency^alpha),
        so groups with higher labels appear more often.
      • Emits batches by concatenating **whole groups** (no splitting, no within-group repeats).
        If adding the next group would exceed batch_size, it starts a new batch.

    Notes
    -----
    - If any single group has size > batch_size → raise ValueError (cannot keep group intact).
    - Group selection can be with or without replacement across an "epoch" (`group_replacement`).
      Within one batch, the same group is used at most once.

    Args
    ----
    dataset: object with `.data: list[dict]`
    batch_size: int
    group_identifiers: fields defining a group
    target_col: numeric score column used to build per-group label
    group_score_agg: "mean_ceil" or callable(list[float]) -> int
    weight_mode: "inverse_freq" or "custom"
    alpha: tempering for inverse frequency weights in [0,1] (0=no reweight, 1=full inverse)
    epsilon: smoothing for counts (>=0)
    prior_per_label: optional dict {label:int -> prior:float} (used when weight_mode="custom" or to scale inverse_freq)
    group_replacement: sample groups with replacement across batches (True) or without (False)
    drop_last: drop final short batch
    seed: RNG seed
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        group_identifiers: Sequence[str],
        *,
        target_col: str = "score",
        group_score_agg: Union[str, Callable[[List[float]], int]] = "mean_ceil",
        weight_mode: str = "inverse_freq",
        alpha: float = 1.0,
        epsilon: float = 1.0,
        prior_per_label: Optional[Dict[int, float]] = None,
        group_replacement: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1]")

        self.dataset = dataset
        self.data: List[Dict] = getattr(dataset, "data", None)
        if self.data is None:
            raise ValueError("dataset must have attribute `.data` (list of dicts)")

        self.batch_size = int(batch_size)
        self.group_identifiers = list(group_identifiers)
        self.target_col = target_col
        self.weight_mode = weight_mode
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.prior_per_label = prior_per_label or {}
        self.group_replacement = bool(group_replacement)
        self.drop_last = bool(drop_last)
        self.rng = random.Random(int(seed))

        # ---- 1) Build groups: key -> list of indices
        groups: Dict[Tuple, List[int]] = defaultdict(list)
        for idx, row in enumerate(self.data):
            key = _build_group_key(row, self.group_identifiers)
            groups[key].append(idx)
        self.groups: Dict[Tuple, List[int]] = dict(groups)
        if not self.groups:
            raise ValueError("No groups found; check `group_identifiers`.")

        # ---- 2) Compute per-group label from target_col
        if callable(group_score_agg):
            agg_fn = group_score_agg
        else:
            if group_score_agg != "mean_ceil":
                raise ValueError("Unsupported group_score_agg; use 'mean_ceil' or provide a callable.")
            agg_fn = _ceil_mean

        self.group_label: Dict[Tuple, int] = {}
        for key, idxs in self.groups.items():
            vals = [float(self.data[i][self.target_col]) for i in idxs]
            self.group_label[key] = int(agg_fn(vals))

        # ---- 3) Build label counts and group weights (oversampling on group level)
        label_counts = Counter(self.group_label.values())  # frequency over groups
        self.labels_sorted = sorted(label_counts.keys())

        weights: List[float] = []
        self._group_keys: List[Tuple] = list(self.groups.keys())

        for key in self._group_keys:
            lab = self.group_label[key]
            if self.weight_mode == "inverse_freq":
                # inverse count with tempering and optional prior scaling
                base = (label_counts[lab] + self.epsilon)
                inv = base ** (-self.alpha)
                w = inv * float(self.prior_per_label.get(lab, 1.0))
            elif self.weight_mode == "custom":
                w = float(self.prior_per_label.get(lab, 1.0))
            else:
                raise ValueError(f"Unknown weight_mode '{self.weight_mode}'")
            weights.append(w)

        # normalize to probabilities for group selection
        total_w = float(sum(weights))
        if total_w <= 0 or not np.isfinite(total_w):
            self._group_probs = [1.0 / len(weights)] * len(weights)
        else:
            self._group_probs = [w / total_w for w in weights]

        # ---- 4) Sanity check: no group larger than batch_size (cannot keep intact)
        self.group_sizes: Dict[Tuple, int] = {k: len(v) for k, v in self.groups.items()}
        too_big = [k for k, sz in self.group_sizes.items() if sz > self.batch_size]
        if too_big:
            bad = too_big[0]
            raise ValueError(f"Group {bad} size {self.group_sizes[bad]} > batch_size {self.batch_size}. "
                             f"Cannot keep group intact. Reduce grouping granularity or increase batch_size.")

        # ---- 5) Length estimate (approx): total_samples / batch_size (+1 if remainder and not drop_last)
        total_samples = len(self.data)
        self._len = total_samples // self.batch_size
        if not self.drop_last and total_samples % self.batch_size:
            self._len += 1

    def __len__(self) -> int:
        return self._len

    def _pick_group_idx_weighted(self, mask_available: Optional[List[bool]] = None) -> Optional[int]:
        """Pick a group index by weighted roulette; optionally mask out used groups."""
        probs = self._group_probs
        if mask_available is not None:
            avail_indices = [i for i, ok in enumerate(mask_available) if ok]
            if not avail_indices:
                return None
            sub_weights = [probs[i] for i in avail_indices]
            j = _roulette_idx(self.rng, sub_weights)
            return avail_indices[j]
        else:
            return _roulette_idx(self.rng, probs)

    def __iter__(self) -> Iterable[List[int]]:
        produced = 0
        target_total = self._len * self.batch_size if self.drop_last else len(self.data)

        # For group_replacement=False, we keep an epoch-level "available" mask
        epoch_available = [True] * len(self._group_keys)

        while produced < target_total:
            batch: List[int] = []
            batch_groups_used = set()

            # keep picking whole groups until we fill (or decide to end)
            while True:
                # stop if we cannot add any group without exceeding
                remaining = self.batch_size - len(batch)
                if remaining <= 0:
                    break

                # build a list of candidate groups that fit into remaining and are allowed this batch
                # (avoid repeating the same group within a batch)
                candidates_mask = []
                for i, key in enumerate(self._group_keys):
                    if key in batch_groups_used:
                        candidates_mask.append(False)
                        continue
                    if not self.group_replacement and not epoch_available[i]:
                        candidates_mask.append(False)
                        continue
                    if self.group_sizes[key] <= remaining:
                        candidates_mask.append(True)
                    else:
                        candidates_mask.append(False)

                # if no candidate fits:
                if not any(candidates_mask):
                    # if batch is empty (e.g., remaining < smallest group), we must start a new batch
                    # to avoid splitting a group. So end this batch loop.
                    break

                # pick a candidate by weights (restricted to mask)
                gi = self._pick_group_idx_weighted(mask_available=candidates_mask)
                if gi is None:
                    break
                gkey = self._group_keys[gi]
                # append whole group
                batch.extend(self.groups[gkey])
                batch_groups_used.add(gkey)
                if not self.group_replacement:
                    epoch_available[gi] = False

                # continue loop to try adding more groups (if space remains)

            if len(batch) == self.batch_size:
                yield batch
                produced += self.batch_size
            else:
                if not self.drop_last and len(batch) > 0:
                    yield batch
                    produced += len(batch)
                # else: drop too-small batch and continue

            # If we've exhausted all groups in epoch_available (no replacement), reset for another pass
            if not self.group_replacement and not any(epoch_available):
                epoch_available = [True] * len(self._group_keys)





class GroupLevelTypePriorBatchSampler(BatchSampler):
    """
    Hierarchical group-level batch sampler for contrastive trajectories.

    Guarantees:
      • Keeps each group (defined by `group_identifiers`, e.g. ["patient_id","score_type","left_or_right"])
        intact within a batch (no splitting, no within-batch repeats).
      • Enforces a score_type prior π(t): "uniform" | "match_val" | "blend".
      • Within each chosen type, oversamples groups by inverse-frequency (tempered) of the
        group label computed from `target_col` via `group_score_agg` (default ceil(mean(score))).
      • Emits batches by concatenating whole groups until `batch_size` is reached; if the next group
        does not fit, starts a new batch.

    Args
    ----
    dataset: object with `.data: list[dict]` aligned with __getitem__
    batch_size: int > 0
    group_identifiers: fields defining a group, e.g. ["patient_id","score_type","left_or_right"]
    target_col: numeric score column for group-label aggregation
    group_score_agg: "mean_ceil" or callable(list[float]) -> int
    # type prior
    prior_mode: "uniform" | "match_val" | "blend"
    val_type_freq: optional dict {score_type: proportion}, used for match/blend
    blend_mix: in [0,1], π = blend_mix * uniform + (1-blend_mix) * match_val
    # within-type balancing (over group labels)
    alpha: in [0,1], tempering for inverse-frequency (0=no balance, 1=full inverse)
    epsilon: >=0, smoothing for counts
    # batching behavior
    size_correction: bool, if True downweights large groups within type (weight /= group_size)
    group_replacement: if True, groups may be reused across batches within an epoch
    drop_last: drop final short batch
    seed: RNG seed
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        group_identifiers: Sequence[str],
        *,
        target_col: str = "score",
        group_score_agg: Union[str, Callable[[List[float]], int]] = "mean_ceil",
        prior_mode: str = "uniform",
        val_type_freq: Optional[Dict[str, float]] = None,
        blend_mix: float = 0.5,
        alpha: float = 1.0,
        epsilon: float = 1.0,
        size_correction: bool = False,
        group_replacement: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1]")
        if not (0.0 <= blend_mix <= 1.0):
            raise ValueError("blend_mix must be in [0,1]")

        self.dataset = dataset
        self.data: List[Dict] = getattr(dataset, "data", None)
        if self.data is None:
            raise ValueError("dataset must have attribute `.data` (list of dicts)")

        self.batch_size = int(batch_size)
        self.group_identifiers = list(group_identifiers)
        self.target_col = target_col
        self.prior_mode = str(prior_mode)
        self.val_type_freq = val_type_freq or {}
        self.blend_mix = float(blend_mix)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.size_correction = bool(size_correction)
        self.group_replacement = bool(group_replacement)
        self.drop_last = bool(drop_last)
        self.rng = random.Random(int(seed))

        # --- 1) Build groups: key -> list[idx]
        groups: Dict[Tuple, List[int]] = defaultdict(list)
        for idx, row in enumerate(self.data):
            key = _build_group_key(row, self.group_identifiers)
            groups[key].append(idx)
        self.groups: Dict[Tuple, List[int]] = dict(groups)
        if not self.groups:
            raise ValueError("No groups found; check `group_identifiers`.")

        # --- 2) Group sizes and guard (no splitting allowed)
        self.group_sizes: Dict[Tuple, int] = {k: len(v) for k, v in self.groups.items()}
        too_big = [k for k, sz in self.group_sizes.items() if sz > self.batch_size]
        if too_big:
            k = too_big[0]
            raise ValueError(
                f"Group {k} has size {self.group_sizes[k]} > batch_size {self.batch_size}. "
                f"Increase batch_size or reduce grouping granularity."
            )

        # --- 3) Compute per-group label from target_col
        if callable(group_score_agg):
            agg_fn = group_score_agg
        elif group_score_agg == "mean_ceil":
            agg_fn = _ceil_mean
        else:
            raise ValueError("Unsupported group_score_agg; use 'mean_ceil' or provide a callable.")

        self.group_label: Dict[Tuple, int] = {}
        for key, idxs in self.groups.items():
            vals = [float(self.data[i][self.target_col]) for i in idxs]
            self.group_label[key] = int(agg_fn(vals))

        # --- 4) Indexing helpers
        self._group_keys: List[Tuple] = list(self.groups.keys())
        # map type for each group (assumes score_type is in the row; recommended that it is in the key)
        self.group_type: Dict[Tuple, str] = {}
        for key, idxs in self.groups.items():
            t = self.data[idxs[0]]["score_type"]
            self.group_type[key] = t

        # --- 5) Type-level structures
        self.types = sorted(set(self.group_type.values()))
        # map: type -> list of group indices (into self._group_keys)
        self.type_to_group_idxs: Dict[str, List[int]] = defaultdict(list)
        for gi, gkey in enumerate(self._group_keys):
            t = self.group_type[gkey]
            self.type_to_group_idxs[t].append(gi)

        # --- 6) Build type prior π(t)
        if self.prior_mode == "uniform":
            self.type_prior = {t: 1.0 / len(self.types) for t in self.types}
        elif self.prior_mode == "match_val":
            vf = self.val_type_freq
            s = sum(vf.get(t, 0.0) for t in self.types)
            if s <= 0:
                self.type_prior = {t: 1.0 / len(self.types) for t in self.types}
            else:
                self.type_prior = {t: vf.get(t, 0.0) / s for t in self.types}
        elif self.prior_mode == "blend":
            u = {t: 1.0 / len(self.types) for t in self.types}
            vf = self.val_type_freq
            s = sum(vf.get(t, 0.0) for t in self.types)
            mv = {t: (vf.get(t, 0.0) / s if s > 0 else u[t]) for t in self.types}
            self.type_prior = {t: self.blend_mix * u[t] + (1 - self.blend_mix) * mv[t] for t in self.types}
            # renormalize
            s2 = sum(self.type_prior.values())
            self.type_prior = {t: self.type_prior[t] / s2 for t in self.types}
        else:
            raise ValueError(f"Unknown prior_mode '{self.prior_mode}'")

        # --- 7) Within-type group weights via inverse freq of group labels (tempered)
        # Count per-type label frequencies across groups
        type_label_counts: Dict[str, Counter] = defaultdict(Counter)
        for gkey, lab in self.group_label.items():
            t = self.group_type[gkey]
            type_label_counts[t][lab] += 1

        # Precompute normalized weights: for each type t, dict {group_index -> weight}
        self.within_type_group_weight: Dict[str, Dict[int, float]] = {}
        for t, gis in self.type_to_group_idxs.items():
            counts = type_label_counts[t]
            w_pairs = []
            for gi in gis:
                gkey = self._group_keys[gi]
                lab = self.group_label[gkey]
                base = (counts[lab] + self.epsilon)
                w = (base ** (-self.alpha))
                if self.size_correction:
                    w /= max(1, self.group_sizes[gkey])
                w_pairs.append((gi, float(w)))
            s = float(sum(w for _, w in w_pairs))
            if s <= 0:
                w_pairs = [(gi, 1.0 / max(1, len(w_pairs))) for gi, _ in w_pairs]
            else:
                w_pairs = [(gi, w / s) for gi, w in w_pairs]
            self.within_type_group_weight[t] = dict(w_pairs)

        # --- 8) Length estimate (approximate)
        total_samples = len(self.data)
        self._len = total_samples // self.batch_size
        if not self.drop_last and total_samples % self.batch_size:
            self._len += 1

    # ---- utilities ----
    def _roulette_from_pairs(self, rng: random.Random, pairs: List[Tuple[int, float]]) -> int:
        """pairs: list[(index, weight)] -> chosen index"""
        s = float(sum(w for _, w in pairs))
        if s <= 0 or not np.isfinite(s):
            return pairs[rng.randrange(len(pairs))][0]
        r = rng.random() * s
        acc = 0.0
        for idx, w in pairs:
            acc += w
            if r <= acc:
                return idx
        return pairs[-1][0]

    def _pick_type_with_soft_quota(self, batch_type_counts: Dict[str, float]) -> str:
        """
        Pick a score_type by prior, softly nudging toward underrepresented types within the current batch.
        Target per-batch type fraction is uniform: 1/|T|.
        """
        tgt = 1.0 / max(1, len(self.types))
        pairs: List[Tuple[str, float]] = []
        for t in self.types:
            prior = self.type_prior[t]
            # penalize types that exceeded target count in this batch
            mult = 1.0 / (1.0 + max(0.0, batch_type_counts.get(t, 0.0) - tgt))
            pairs.append((t, prior * mult))
        # convert to indexable pairs
        idx_pairs = list(enumerate([w for _, w in pairs]))
        chosen_idx = _roulette_idx(self.rng, [w for _, w in idx_pairs])
        return self.types[chosen_idx]

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterable[List[int]]:
        produced = 0
        target_total = self._len * self.batch_size if self.drop_last else len(self.data)

        # epoch-level mask when group_replacement=False
        epoch_available = [True] * len(self._group_keys)

        while produced < target_total:
            batch: List[int] = []
            batch_groups_used = set()
            batch_type_counts: Dict[str, float] = {}

            while True:
                remaining = self.batch_size - len(batch)
                if remaining <= 0:
                    break

                # 1) choose a type by prior (soft quotas inside the batch)
                t = self._pick_type_with_soft_quota(batch_type_counts)

                # 2) build candidates among groups of that type that fit & are allowed
                cand_pairs: List[Tuple[int, float]] = []
                for gi in self.type_to_group_idxs[t]:
                    gkey = self._group_keys[gi]
                    if gkey in batch_groups_used:
                        continue
                    if not self.group_replacement and not epoch_available[gi]:
                        continue
                    if self.group_sizes[gkey] <= remaining:
                        w = self.within_type_group_weight[t][gi]
                        cand_pairs.append((gi, w))

                # If none fit for this type, try other types (fallback)
                picked_gi = None
                picked_type = t
                if not cand_pairs:
                    for t2 in self.types:
                        if t2 == t:
                            continue
                        cand2: List[Tuple[int, float]] = []
                        for gi in self.type_to_group_idxs[t2]:
                            gkey = self._group_keys[gi]
                            if gkey in batch_groups_used:
                                continue
                            if not self.group_replacement and not epoch_available[gi]:
                                continue
                            if self.group_sizes[gkey] <= remaining:
                                w = self.within_type_group_weight[t2][gi]
                                cand2.append((gi, w))
                        if cand2:
                            picked_gi = self._roulette_from_pairs(self.rng, cand2)
                            picked_type = t2
                            break
                else:
                    picked_gi = self._roulette_from_pairs(self.rng, cand_pairs)

                if picked_gi is None:
                    # nothing fits → finish the batch
                    break

                gkey = self._group_keys[picked_gi]
                batch.extend(self.groups[gkey])
                batch_groups_used.add(gkey)
                batch_type_counts[picked_type] = batch_type_counts.get(picked_type, 0.0) + 1.0
                if not self.group_replacement:
                    epoch_available[picked_gi] = False

            if len(batch) == self.batch_size:
                yield batch
                produced += self.batch_size
            else:
                if not self.drop_last and batch:
                    yield batch
                    produced += len(batch)
                # else: drop incomplete batch and continue

            # reset availability if we exhausted all groups without replacement
            if not self.group_replacement and not any(epoch_available):
                epoch_available = [True] * len(self._group_keys)











def _val_group_key(row: Dict, cols: Sequence[str]) -> Tuple:
    return tuple(row[c] for c in cols)

class GroupPackingBatchSampler(BatchSampler):
    """
    Validation sampler:
      - No weighting, no repeats, no shuffling (deterministic).
      - Packs whole groups into batches; groups never split across batches.
      - If a group exceeds batch_size -> raises (you should increase batch_size or change grouping).
    """
    def __init__(
        self,
        dataset,
        batch_size: int,
        group_identifiers: Sequence[str],
        drop_last: bool = False,
        sort_groups_by: Optional[str] = "key",  # "key" | "size_desc" | None
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.dataset = dataset
        self.data = getattr(dataset, "data", None)
        if self.data is None:
            raise ValueError("dataset must expose `.data` list[dict]")
        self.batch_size = int(batch_size)
        self.group_identifiers = list(group_identifiers)
        self.drop_last = bool(drop_last)
        self.sort_groups_by = sort_groups_by

        # Build groups deterministically
        groups: Dict[Tuple, List[int]] = defaultdict(list)
        for idx, row in enumerate(self.data):
            groups[_val_group_key(row, self.group_identifiers)].append(idx)
        # deterministic ordering
        items = list(groups.items())
        if self.sort_groups_by == "size_desc":
            items.sort(key=lambda kv: (-len(kv[1]), kv[0]))
        elif self.sort_groups_by == "key":
            items.sort(key=lambda kv: kv[0])
        else:
            # keep insertion order (already deterministic for fixed data order)
            pass

        self.group_keys = [k for k, _ in items]
        self.group_indices = {k: v for k, v in items}

        # guard: group must fit the batch
        too_big = [k for k in self.group_keys if len(self.group_indices[k]) > self.batch_size]
        if too_big:
            k = too_big[0]
            raise ValueError(
                f"Validation group {k} has size {len(self.group_indices[k])} > batch_size {self.batch_size}. "
                f"Increase batch size or relax grouping."
            )

        # length estimate
        total = len(self.data)
        self._len = total // self.batch_size + (0 if (self.drop_last or total % self.batch_size == 0) else 1)

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterable[List[int]]:
        batch: List[int] = []
        for k in self.group_keys:
            idxs = self.group_indices[k]
            if len(batch) + len(idxs) <= self.batch_size:
                batch.extend(idxs)
            else:
                # flush
                if batch:
                    yield batch
                batch = idxs.copy()
        if batch and not self.drop_last:
            yield batch






# class PatientBatchSampler(Sampler):
#     """
#     Groups indices by patient and yields mini-batches that
#     (almost) always contain a single patient.
#     """

#     def __init__(self, patient_ids, batch_size, drop_last=False):
#         self.batch_size = batch_size
#         self.drop_last  = drop_last

#         self.patient2idx = defaultdict(list)
#         for idx, pid in enumerate(patient_ids):
#             self.patient2idx[pid].append(idx)

#     # --------------------------------------------------
#     def _epoch_batches(self):
#         # shuffle within every patient
#         for bucket in self.patient2idx.values():
#             random.shuffle(bucket)

#         patient_order = list(self.patient2idx.keys())
#         random.shuffle(patient_order)

#         for pid in patient_order:
#             bucket = self.patient2idx[pid]
#             for i in range(0, len(bucket), self.batch_size):
#                 chunk = bucket[i : i + self.batch_size]
#                 if len(chunk) == self.batch_size or not self.drop_last:
#                     yield chunk

#     # --------------------------------------------------
#     def __iter__(self):
#         yield from self._epoch_batches()

#     def __len__(self):
#         n = sum(len(v) for v in self.patient2idx.values())
#         return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)




class PatientBatchSampler(Sampler):
    """
    Yields lists of indices so that (almost) every batch
    contains samples from a single patient.
    """
    def __init__(self, patient_ids, batch_size, drop_last=False):
        self.batch_size  = batch_size
        self.drop_last   = drop_last

        # Map patient_id → list[index]
        self.patient2idx = defaultdict(list)
        for idx, pid in enumerate(patient_ids):
            self.patient2idx[pid].append(idx)

        self._build_epoch_batches()

    def _build_epoch_batches(self):
        # Shuffle *within* each patient
        for idx_list in self.patient2idx.values():
            random.shuffle(idx_list)

        # Pick a new random order of patients each epoch
        self.epoch_batches = []
        patient_order = list(self.patient2idx.keys())
        random.shuffle(patient_order)

        for pid in patient_order:
            idxs = self.patient2idx[pid]
            # chunk into batch_size pieces
            for i in range(0, len(idxs), self.batch_size):
                chunk = idxs[i : i + self.batch_size]
                if len(chunk) == self.batch_size or not self.drop_last:
                    self.epoch_batches.append(chunk)

    def __iter__(self):
        # Re-shuffle every epoch
        self._build_epoch_batches()
        for batch in self.epoch_batches:
            yield batch

    def __len__(self):
        return len(self.epoch_batches)