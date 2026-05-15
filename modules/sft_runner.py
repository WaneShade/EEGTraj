# -*- coding: utf-8 -*-
"""
train_multitask_sft_v3.py

State-aware shared-chain SFT for TUAB + TUEV + HMC + SEED.

Validation / test collection
----------------------------
- Validation defaults to class-balanced subsets for fast, stable monitoring.
- Test defaults to a proportional subset of the official test split, which is
  much faster than full decoding while staying closer to full-test class
  proportions than a balanced subset.
- Set --test_eval_mode full if you want the exact full-split evaluation.
"""

import argparse
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from downstream_dataset import HMCLoader, SEEDDataset, TUABLoader, TUEVLoader
from model.model import GPTConfig
from model.model_neurolm import NeuroLM

try:
    from run_logger import setup_run_logging
except Exception:
    setup_run_logging = None

from modules.neuro_utils import (
    build_tokmap,
    build_trie,
    compute_decision_features,
    encode_eeg_only,
    list_pkls,
    make_loader,
    morpho_label,
    sample_or_greedy_allowed,
    set_seed,
    source_label,
    spatial_label,
    temporal_label,
    trie_next_allowed,
    trie_step,
    weighted_nll_loss,
)


TUEV_LABELS = ["A", "B", "C", "D", "E", "F"]
TUAB_LABELS = ["normal", "abnormal"]
HMC_LABELS = ["wake", "n1", "n2", "n3", "rem"]
SEED_LABELS = ["positive", "neutral", "negative"]

SOURCE_VALUES = ["cerebral_event", "noncerebral", "background_like", "uncertain"]
STATE_VALUES = ["wake_like", "transition_like", "stable_sleep_like", "rem_like", "na", "uncertain"]
TEMPORAL_VALUES = [
    "isolated_transient",
    "periodic_repeating",
    "slow_drift",
    "broadband_irregular",
    "stable_rhythm",
    "state_transition",
    "none_or_uncertain",
]
SPATIAL_VALUES = ["generalized", "lateralized", "frontal_dominant", "focal_local", "diffuse_mixed", "na"]
MORPHO_VALUES = [
    "spike_sharp_complex",
    "drift_like",
    "noise_like",
    "background_rhythm",
    "slow_wave_like",
    "spindle_kcomplex_like",
    "mixed_low_voltage",
    "uncertain",
    "na",
]

TUEV_OPTION_LINES = (
    "(A) spike and slow wave.\n"
    "(B) generalized periodic epileptiform discharge.\n"
    "(C) periodic lateralized epileptiform discharge.\n"
    "(D) eye movement.\n"
    "(E) artifact.\n"
    "(F) background."
)
TUAB_OPTION_LINES = "(normal) normal EEG segment.\n(abnormal) abnormal EEG segment."
HMC_OPTION_LINES = "(wake) Wake.\n(n1) NREM-1.\n(n2) NREM-2.\n(n3) NREM-3.\n(rem) REM."
SEED_OPTION_LINES = "(positive) Positive emotion.\n(neutral) Neutral emotion.\n(negative) Negative emotion."

FS = 200.0
TASK_NAME_TO_OFFSET = {"TUAB": 0, "TUEV": 1, "HMC": 2, "SEED": 3}
TASK_WEIGHT_SCALES = {
    "TUAB": {"task_weight": 1.0, "source_weight": 1.0, "state_weight": 1.0, "temporal_weight": 1.0, "spatial_weight": 0.8, "morpho_weight": 1.0, "answer_weight": 1.0},
    "TUEV": {"task_weight": 1.0, "source_weight": 1.25, "state_weight": 0.2, "temporal_weight": 1.0, "spatial_weight": 1.0, "morpho_weight": 1.0, "answer_weight": 1.2},
    "HMC": {"task_weight": 1.0, "source_weight": 0.8, "state_weight": 1.4, "temporal_weight": 1.15, "spatial_weight": 0.5, "morpho_weight": 1.2, "answer_weight": 1.0},
    "SEED": {"task_weight": 1.0, "source_weight": 0.9, "state_weight": 0.8, "temporal_weight": 1.0, "spatial_weight": 0.9, "morpho_weight": 0.9, "answer_weight": 1.0},
}


@dataclass
class TaskSpec:
    name: str
    labels: List[str]
    loader_cls: Optional[object]
    root: str
    train_dir: str = ""
    val_dir: str = ""
    test_dir: str = ""
    split_kind: str = "pkl"
    seed_h5_path: str = ""


@dataclass
class DecodeResult:
    pred_idx: int
    pred_label: str
    slots: Dict[str, str]
    raw_tail: str
    score_abnormal: Optional[float] = None


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def label_space_for_task(task_name: str) -> List[str]:
    if task_name == "TUAB":
        return TUAB_LABELS
    if task_name == "TUEV":
        return TUEV_LABELS
    if task_name == "HMC":
        return HMC_LABELS
    if task_name == "SEED":
        return SEED_LABELS
    raise ValueError(task_name)


def resolve_split_dir(root: str, split: str) -> str:
    cands = {
        "train": ["processed_train", "train", "train_processed"],
        "val": ["processed_eval", "processed_val", "eval", "val"],
        "test": ["processed_test", "test"],
    }[split]
    for sub in cands:
        p = os.path.join(root, sub)
        if os.path.isdir(p):
            return p
    return os.path.join(root, cands[0])


def resolve_seed_h5(root: str) -> str:
    cands = [
        os.path.join(root, "seed-3.hdf5"),
        os.path.join(root, "h5data", "seed-3.hdf5"),
    ]
    for path in cands:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Could not find SEED HDF5 under {root}. Tried: {cands}"
    )


SEED_SYMBOL_TO_IDX = {
    "H": 0,
    "N": 1,
    "S": 2,
    "positive": 0,
    "neutral": 1,
    "negative": 2,
    "Positive": 0,
    "Neutral": 1,
    "Negative": 2,
}


def parse_seed_symbol(raw) -> int:
    if isinstance(raw, np.ndarray):
        arr = np.asarray(raw).reshape(-1)
        if arr.size != 1:
            raise TypeError(f"Unsupported SEED label array shape: {arr.shape}")
        raw = arr[0]
    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode("utf-8")
    key = str(raw).strip()
    if key not in SEED_SYMBOL_TO_IDX:
        raise KeyError(f"Unsupported SEED label symbol: {key}")
    return int(SEED_SYMBOL_TO_IDX[key])


def parse_label_scalar(y, n_classes: Optional[int] = None) -> int:
    if isinstance(y, torch.Tensor):
        arr = y.detach().cpu().numpy()
    elif isinstance(y, np.ndarray):
        arr = y
    elif isinstance(y, (list, tuple)):
        arr = np.asarray(y)
    else:
        return int(y)

    arr = np.asarray(arr)
    if arr.ndim == 0 or arr.size == 1:
        return int(arr.reshape(-1)[0])

    flat = arr.reshape(-1)
    if (
        n_classes is not None
        and flat.size == int(n_classes)
        and np.issubdtype(flat.dtype, np.number)
        and np.all(np.isfinite(flat))
    ):
        return int(np.argmax(flat))

    if np.issubdtype(flat.dtype, np.number):
        if np.allclose(flat, flat[0]):
            return int(round(float(flat[0])))
        return int(round(float(flat[0])))

    raise TypeError(f"Unsupported label format: type={type(y)}, shape={arr.shape}")


def load_label_from_pkl(task_name: str, root: str, fn: str) -> int:
    sample = pickle.load(open(os.path.join(root, fn), "rb"))
    if task_name == "TUEV":
        return int(sample["label"][0] - 1)
    if task_name == "TUAB":
        return parse_label_scalar(sample["y"], n_classes=2)
    if task_name == "HMC":
        return parse_label_scalar(sample["y"], n_classes=5)
    raise ValueError(task_name)


def scan_labels_for_sampler(task_name: str, root: str, files: List[str], n_classes: int) -> Tuple[List[int], List[float]]:
    counts = [0] * n_classes
    labels = []
    for fn in files:
        y = load_label_from_pkl(task_name, root, fn)
        if 0 <= y < n_classes:
            labels.append(y)
            counts[y] += 1
    weights = [1.0 / max(1, counts[y]) for y in labels]
    return counts, weights


def build_label_buckets(task_name: str, root: str, files: List[str], n_classes: int) -> Tuple[Dict[int, List[str]], List[int]]:
    buckets: Dict[int, List[str]] = {i: [] for i in range(n_classes)}
    counts = [0] * n_classes
    for fn in files:
        y = load_label_from_pkl(task_name, root, fn)
        if 0 <= y < n_classes:
            buckets[y].append(fn)
            counts[y] += 1
    return buckets, counts


def build_seed_dataset(h5_path: str, split: str, eeg_max_len: int, text_max_len: int) -> SEEDDataset:
    split_map = {
        "train": (0.0, 0.6),
        "val": (0.6, 0.8),
        "test": (0.8, 1.0),
    }
    if split not in split_map:
        raise ValueError(split)
    trial_start, trial_end = split_map[split]
    return SEEDDataset(
        Path(h5_path),
        window_size=800,
        stride_size=800,
        trial_start_percentage=trial_start,
        trial_end_percentage=trial_end,
        is_instruct=True,
        is_val=True,
        eeg_max_len=eeg_max_len,
        text_max_len=text_max_len,
    )


def build_seed_index_buckets(ds: SEEDDataset) -> Tuple[Dict[int, List[int]], List[int]]:
    buckets: Dict[int, List[int]] = {i: [] for i in range(len(SEED_LABELS))}
    counts = [0] * len(SEED_LABELS)

    global_idxes = list(getattr(ds, "_SEEDDataset__global_idxes"))
    local_idxess = list(getattr(ds, "_SEEDDataset__local_idxess"))
    labelss = list(getattr(ds, "_SEEDDataset__labelss"))
    total_len = int(len(ds))

    for subject_id, local_idxes in enumerate(local_idxess):
        global_start = int(global_idxes[subject_id])
        if subject_id + 1 < len(global_idxes):
            subject_total = int(global_idxes[subject_id + 1] - global_start)
        else:
            subject_total = int(total_len - global_start)

        local_idxes = [int(x) for x in local_idxes]
        labels_subject = list(labelss[subject_id])
        for trial_id, local_start in enumerate(local_idxes):
            local_end = local_idxes[trial_id + 1] if trial_id + 1 < len(local_idxes) else subject_total
            n_items = int(local_end - local_start)
            if n_items <= 0:
                continue
            y = parse_seed_symbol(labels_subject[trial_id])
            start = global_start + int(local_start)
            end = global_start + int(local_end)
            buckets[y].extend(range(start, end))
            counts[y] += n_items
    return buckets, counts


def seed_sampler_weights_from_buckets(ds: SEEDDataset, buckets: Dict[int, List[int]]) -> List[float]:
    weights = [0.0] * len(ds)
    for cls_idx, indices in buckets.items():
        cls_weight = 1.0 / max(1, len(indices))
        for idx in indices:
            weights[int(idx)] = float(cls_weight)
    return weights


def _allocate_proportional_counts(
    available: List[int],
    max_samples: int,
    min_per_class: int,
    rng: random.Random,
) -> List[int]:
    total_available = int(sum(available))
    if max_samples <= 0 or max_samples >= total_available:
        return list(available)

    n_classes = len(available)
    base = [0] * n_classes
    if int(min_per_class) > 0:
        cand = [min(int(a), int(min_per_class)) if int(a) > 0 else 0 for a in available]
        if sum(cand) <= max_samples:
            base = cand

    remaining = int(max_samples) - int(sum(base))
    if remaining <= 0:
        return base

    residual = [max(0, int(a) - int(base[i])) for i, a in enumerate(available)]
    residual_total = int(sum(residual))
    if residual_total <= 0:
        return base

    raw = [remaining * (float(r) / float(residual_total)) for r in residual]
    extra = [min(residual[i], int(math.floor(raw[i]))) for i in range(n_classes)]
    counts = [base[i] + extra[i] for i in range(n_classes)]
    leftover = remaining - int(sum(extra))
    if leftover <= 0:
        return counts

    order = list(range(n_classes))
    order.sort(key=lambda i: ((raw[i] - math.floor(raw[i])), rng.random()), reverse=True)
    while leftover > 0:
        advanced = False
        for i in order:
            if counts[i] < available[i]:
                counts[i] += 1
                leftover -= 1
                advanced = True
                if leftover <= 0:
                    break
        if not advanced:
            break
    return counts


def sample_files_for_eval(
    buckets: Dict[int, List[str]],
    mode: str,
    per_class: int,
    seed: int,
    max_samples: int = 0,
    min_per_class: int = 0,
) -> Tuple[List[str], List[int]]:
    rng = random.Random(seed)
    n_classes = len(buckets)
    chosen: List[str] = []
    counts = [0] * n_classes
    if mode == "full" or per_class is None or int(per_class) <= 0:
        for c in range(n_classes):
            xs = list(buckets[c])
            chosen.extend(xs)
            counts[c] = len(xs)
        chosen.sort()
        return chosen, counts

    if mode == "proportional":
        available = [len(buckets[c]) for c in range(n_classes)]
        quotas = _allocate_proportional_counts(available, int(max_samples), int(min_per_class), rng)
        for c in range(n_classes):
            xs = list(buckets[c])
            take_n = int(quotas[c])
            take = xs if take_n >= len(xs) else rng.sample(xs, take_n)
            chosen.extend(take)
            counts[c] = len(take)
        rng.shuffle(chosen)
        return chosen, counts

    for c in range(n_classes):
        xs = list(buckets[c])
        take = xs if len(xs) <= per_class else rng.sample(xs, per_class)
        chosen.extend(take)
        counts[c] = len(take)
    rng.shuffle(chosen)
    return chosen, counts


def maybe_make_loader(ds, batch_size, num_workers, weights, balanced, drop_last, shuffle):
    sampler = None
    if balanced and weights is not None and len(weights) == len(ds):
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return make_loader(ds, batch_size, num_workers, sampler=sampler, shuffle=(shuffle and sampler is None), drop_last=drop_last)


def build_eval_loader_for_files(loader_cls, split_dir: str, files: List[str], eeg_max_len: int, text_max_len: int, batch_size: int, num_workers: int):
    ds = loader_cls(split_dir, files, eeg_max_len=eeg_max_len, text_max_len=text_max_len, is_instruct=True, is_val=True)
    return maybe_make_loader(ds, batch_size, num_workers, None, False, False, False)


def build_eval_loader_for_indices(ds, indices: List[int], batch_size: int, num_workers: int):
    subset = Subset(ds, list(indices))
    return maybe_make_loader(subset, batch_size, num_workers, None, False, False, False)


def build_eval_loader(
    task_name: str,
    loader_cls,
    split_dir: str,
    buckets: Dict[int, List[str]],
    mode: str,
    per_class: int,
    seed: int,
    eeg_max_len: int,
    text_max_len: int,
    batch_size: int,
    num_workers: int,
    max_samples: int = 0,
    min_per_class: int = 0,
):
    files, counts = sample_files_for_eval(
        buckets,
        mode=mode,
        per_class=per_class,
        seed=seed,
        max_samples=max_samples,
        min_per_class=min_per_class,
    )
    loader = build_eval_loader_for_files(loader_cls, split_dir, files, eeg_max_len, text_max_len, batch_size, num_workers)
    return loader, files, counts


def build_task_spec(name: str, root: str) -> TaskSpec:
    root = str(root)
    if name == "TUEV":
        return TaskSpec(name="TUEV", labels=TUEV_LABELS, loader_cls=TUEVLoader, root=root, train_dir=resolve_split_dir(root, "train"), val_dir=resolve_split_dir(root, "val"), test_dir=resolve_split_dir(root, "test"))
    if name == "TUAB":
        return TaskSpec(name="TUAB", labels=TUAB_LABELS, loader_cls=TUABLoader, root=root, train_dir=resolve_split_dir(root, "train"), val_dir=resolve_split_dir(root, "val"), test_dir=resolve_split_dir(root, "test"))
    if name == "HMC":
        return TaskSpec(name="HMC", labels=HMC_LABELS, loader_cls=HMCLoader, root=root, train_dir=resolve_split_dir(root, "train"), val_dir=resolve_split_dir(root, "val"), test_dir=resolve_split_dir(root, "test"))
    if name == "SEED":
        return TaskSpec(name="SEED", labels=SEED_LABELS, loader_cls=None, root=root, split_kind="seed_h5", seed_h5_path=resolve_seed_h5(root))
    raise ValueError(name)


def _valid_eeg_len_from_mask(eeg_mask: torch.Tensor) -> torch.Tensor:
    return eeg_mask.long().sum(dim=1)


def build_gpt_mask_multitask(X_eeg_tokens: torch.Tensor, X_text: torch.Tensor, eeg_mask: torch.Tensor, input_time: torch.Tensor) -> torch.Tensor:
    device = X_eeg_tokens.device
    B = X_eeg_tokens.size(0)
    eeg_len = X_eeg_tokens.size(1)
    text_len = X_text.size(1)
    total = eeg_len + text_len

    gpt_mask = torch.tril(torch.ones((B, total, total), device=device, dtype=torch.bool)).unsqueeze(1)
    valid_eeg_lens = _valid_eeg_len_from_mask(eeg_mask)
    for b in range(B):
        valid_len = int(valid_eeg_lens[b].item())
        if valid_len > 0:
            times_b = input_time[b, :valid_len].detach().cpu().tolist()
            start = 0
            while start < valid_len:
                t = times_b[start]
                end = start + 1
                while end < valid_len and times_b[end] == t:
                    end += 1
                gpt_mask[b, :, start:end, start:end] = True
                start = end
        if valid_len < eeg_len:
            gpt_mask[b, :, :, valid_len:eeg_len] = False
            gpt_mask[b, :, valid_len:eeg_len, :] = False
    return gpt_mask


def tokens_to_waveform_multitask(x_eeg_1: torch.Tensor, eeg_mask_1: torch.Tensor, input_time_1: torch.Tensor) -> Optional[torch.Tensor]:
    valid = x_eeg_1[eeg_mask_1]
    if valid.size(0) <= 0:
        return None
    times = input_time_1[eeg_mask_1].long()
    uniq = torch.unique_consecutive(times)
    if uniq.numel() <= 0:
        return None

    blocks: List[torch.Tensor] = []
    n_chans = None
    cursor = 0
    times_list = times.detach().cpu().tolist()
    for t in uniq.detach().cpu().tolist():
        start = cursor
        while cursor < len(times_list) and times_list[cursor] == t:
            cursor += 1
        block = valid[start:cursor]
        if block.size(0) <= 0:
            continue
        if n_chans is None:
            n_chans = int(block.size(0))
        if block.size(0) != n_chans:
            if block.size(0) < n_chans:
                return None
            block = block[:n_chans]
        blocks.append(block)

    if not blocks or n_chans is None:
        return None
    x = torch.stack(blocks, dim=0).permute(1, 0, 2).contiguous()
    return x.view(n_chans, -1)


@torch.no_grad()
def estimate_signal_quality_thresholds_multitask(dl_train: DataLoader, max_batches: int, q1: float, q2: float) -> Tuple[float, float]:
    scores: List[float] = []
    for bi, batch in enumerate(dl_train):
        if bi >= max_batches:
            break
        X_eeg, _t, _Y, _ic, input_time, eeg_mask, _gm = batch
        X_eeg = X_eeg.float().cpu()
        input_time = input_time.long().cpu()
        eeg_mask = eeg_mask.bool().cpu()
        for i in range(X_eeg.size(0)):
            xw = tokens_to_waveform_multitask(X_eeg[i], eeg_mask[i], input_time[i])
            feats = compute_decision_features(xw, t1=0.0, t2=0.0)
            art = float(feats.get("artifact_score", np.nan))
            if np.isfinite(art):
                scores.append(art)
    if len(scores) <= 0:
        return 0.08, 0.30

    arr = np.asarray(scores, dtype=np.float64)
    t1 = float(np.quantile(arr, q1))
    t2 = float(np.quantile(arr, q2))
    print(f"[signal_quality] estimated thresholds: t1(q{q1})={t1:.6f}, t2(q{q2})={t2:.6f}, n_scores={len(scores)}")
    return t1, t2


def _median_band_fraction(x: torch.Tensor, low_hz: float, high_hz: float, fs: float = FS) -> float:
    if x is None or x.numel() <= 0 or x.size(1) < 8:
        return float("nan")
    x = x.float() - x.float().mean(dim=1, keepdim=True)
    spec = torch.fft.rfft(x, dim=1)
    power = (spec.real * spec.real + spec.imag * spec.imag)
    freqs = torch.fft.rfftfreq(x.size(1), d=1.0 / fs).to(x.device)
    band_mask = (freqs >= low_hz) & (freqs < high_hz)
    total_mask = (freqs >= 0.5) & (freqs < 30.0)
    if not torch.any(band_mask) or not torch.any(total_mask):
        return float("nan")
    total = power[:, total_mask].sum(dim=1).clamp(min=1e-12)
    frac = power[:, band_mask].sum(dim=1) / total
    return float(torch.median(frac).item())


def compute_sleep_features(x: Optional[torch.Tensor], feats: Dict[str, float]) -> Dict[str, float]:
    if x is None:
        return {"sigma_ratio": np.nan, "slow_ratio": np.nan, "low_voltage_score": np.nan}

    sigma_ratio = _median_band_fraction(x, 12.0, 15.0, fs=FS)
    delta = float(feats.get("delta", np.nan))
    theta = float(feats.get("theta", np.nan))
    slow_ratio = np.nan if (not np.isfinite(delta) or not np.isfinite(theta)) else float(delta + theta)
    ll = float(feats.get("line_length", np.nan))
    low_voltage_score = np.nan if not np.isfinite(ll) else clamp01((1.18 - ll) / 0.18)
    return {"sigma_ratio": sigma_ratio, "slow_ratio": slow_ratio, "low_voltage_score": low_voltage_score}


def infer_background_state(feats: Dict[str, float], sleep_feats: Dict[str, float]) -> str:
    art = float(feats.get("artifact_score", np.nan))
    hf = float(feats.get("hf", np.nan))
    clip_ratio = float(feats.get("clip_ratio", np.nan))
    flatline_ratio = float(feats.get("flatline_ratio", np.nan))
    alpha = float(feats.get("alpha", np.nan))
    slow_ratio = float(sleep_feats.get("slow_ratio", np.nan))
    sigma_ratio = float(sleep_feats.get("sigma_ratio", np.nan))
    low_voltage = float(sleep_feats.get("low_voltage_score", np.nan))

    if ((np.isfinite(art) and art > 0.40) or (np.isfinite(hf) and hf > 0.18) or (np.isfinite(clip_ratio) and clip_ratio > 0.04) or (np.isfinite(flatline_ratio) and flatline_ratio > 0.30)):
        return "uncertain"
    if (np.isfinite(alpha) and np.isfinite(slow_ratio) and np.isfinite(sigma_ratio) and np.isfinite(low_voltage) and alpha > 0.24 and slow_ratio < 0.45 and sigma_ratio < 0.08 and low_voltage < 0.45):
        return "wake_like"
    if np.isfinite(sigma_ratio) and sigma_ratio > 0.08:
        return "stable_sleep_like"
    if np.isfinite(slow_ratio) and slow_ratio > 0.60:
        return "stable_sleep_like"
    if np.isfinite(low_voltage) and np.isfinite(alpha) and np.isfinite(slow_ratio) and low_voltage > 0.55 and alpha < 0.18 and slow_ratio < 0.55:
        return "rem_like"
    return "transition_like"


def temporal_from_state(state: str, feats: Dict[str, float], sleep_feats: Dict[str, float]) -> str:
    slow_ratio = float(sleep_feats.get("slow_ratio", np.nan))
    if state == "wake_like":
        return "stable_rhythm"
    if state == "stable_sleep_like":
        return "slow_drift" if (np.isfinite(slow_ratio) and slow_ratio > 0.60) else "stable_rhythm"
    if state in {"transition_like", "rem_like"}:
        return "state_transition"
    return "none_or_uncertain"


def spatial_from_state(state: str, feats: Dict[str, float], sleep_feats: Dict[str, float]) -> str:
    frontal_ratio = float(feats.get("frontal_ratio", np.nan))
    slow_ratio = float(sleep_feats.get("slow_ratio", np.nan))
    if state == "wake_like":
        return "na"
    if np.isfinite(frontal_ratio) and np.isfinite(slow_ratio) and frontal_ratio > 1.45 and slow_ratio > 0.55:
        return "frontal_dominant"
    return "diffuse_mixed"


def morpho_from_state(state: str, feats: Dict[str, float], sleep_feats: Dict[str, float]) -> str:
    sigma_ratio = float(sleep_feats.get("sigma_ratio", np.nan))
    slow_ratio = float(sleep_feats.get("slow_ratio", np.nan))
    if state == "wake_like":
        return "background_rhythm"
    if state == "stable_sleep_like":
        if np.isfinite(sigma_ratio) and sigma_ratio > 0.08 and (not np.isfinite(slow_ratio) or slow_ratio < 0.65):
            return "spindle_kcomplex_like"
        return "slow_wave_like"
    if state == "rem_like":
        return "mixed_low_voltage"
    if state == "transition_like":
        if np.isfinite(sigma_ratio) and sigma_ratio > 0.08:
            return "spindle_kcomplex_like"
        if np.isfinite(slow_ratio) and slow_ratio > 0.55:
            return "slow_wave_like"
        return "mixed_low_voltage"
    return "uncertain"


def prompt_str_shared(task_name: str) -> str:
    if task_name == "TUEV":
        q = "Question:\nGiven the EEG segment, choose the correct event type.\n\nOptions:\n" + f"{TUEV_OPTION_LINES}\n\n"
        ans_desc = "ANS=(A|B|C|D|E|F)"
    elif task_name == "TUAB":
        q = "Question:\nGiven the EEG segment, decide whether the segment is normal or abnormal.\n\nOptions:\n" + f"{TUAB_OPTION_LINES}\n\n"
        ans_desc = "ANS=(normal|abnormal)"
    elif task_name == "HMC":
        q = "Question:\nGiven the EEG segment, choose the correct sleep stage.\n\nOptions:\n" + f"{HMC_OPTION_LINES}\n\n"
        ans_desc = "ANS=(wake|n1|n2|n3|rem)"
    elif task_name == "SEED":
        q = "Question:\nGiven the EEG segment, choose the correct emotion class.\n\nOptions:\n" + f"{SEED_OPTION_LINES}\n\n"
        ans_desc = "ANS=(positive|neutral|negative)"
    else:
        raise ValueError(task_name)

    return (
        q
        + "Reasoning policy:\n"
        + "Use one shared decision chain before the answer. The answer must be supported by source type, state cue, temporal organization, spatial distribution, and morphology.\n\n"
        + "Allowed DSL:\n"
        + "TASK=[TUAB|TUEV|HMC|SEED]\n"
        + "SOURCE=[cerebral_event|noncerebral|background_like|uncertain]\n"
        + "STATE=[wake_like|transition_like|stable_sleep_like|rem_like|na|uncertain]\n"
        + "TEMPORAL=[isolated_transient|periodic_repeating|slow_drift|broadband_irregular|stable_rhythm|state_transition|none_or_uncertain]\n"
        + "SPATIAL=[generalized|lateralized|frontal_dominant|focal_local|diffuse_mixed|na]\n"
        + "MORPHO=[spike_sharp_complex|drift_like|noise_like|background_rhythm|slow_wave_like|spindle_kcomplex_like|mixed_low_voltage|uncertain|na]\n"
        + f"{ans_desc}\n\n"
        + "Output DSL exactly:\n"
        + "TASK=[TUAB|TUEV|HMC|SEED]\n"
        + "SOURCE=[cerebral_event|noncerebral|background_like|uncertain]\n"
        + "STATE=[wake_like|transition_like|stable_sleep_like|rem_like|na|uncertain]\n"
        + "TEMPORAL=[isolated_transient|periodic_repeating|slow_drift|broadband_irregular|stable_rhythm|state_transition|none_or_uncertain]\n"
        + "SPATIAL=[generalized|lateralized|frontal_dominant|focal_local|diffuse_mixed|na]\n"
        + "MORPHO=[spike_sharp_complex|drift_like|noise_like|background_rhythm|slow_wave_like|spindle_kcomplex_like|mixed_low_voltage|uncertain|na]\n"
        + f"{ans_desc}\n\n"
        + "DSL:\n"
    )


def completion_dsl_shared(task_name: str, answer_text: str, slots: Dict[str, str]) -> str:
    return (
        f"TASK={task_name}\n"
        f"SOURCE={slots['source']}\n"
        f"STATE={slots['state']}\n"
        f"TEMPORAL={slots['temporal']}\n"
        f"SPATIAL={slots['spatial']}\n"
        f"MORPHO={slots['morpho']}\n"
        f"ANS=({answer_text})\n"
        "<|endoftext|>"
    )


def build_sft_xy_shared(tok, task_name: str, answer_text: str, slots: Dict[str, str], text_max_len: int):
    prompt = prompt_str_shared(task_name)
    completion = completion_dsl_shared(task_name, answer_text, slots)
    full_ids = [tok.sep_id] + tok.enc.encode(prompt + completion, allowed_special={"<|endoftext|>"})
    prompt_ids = [tok.sep_id] + tok.enc.encode(prompt, allowed_special={"<|endoftext|>"})

    if text_max_len and text_max_len > 0:
        full_ids = full_ids[-text_max_len:]
        prompt_ids = prompt_ids[-text_max_len:]
        prompt_len = len(prompt_ids)
        pad_len = text_max_len - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [tok.eos_id] * pad_len
        x = torch.tensor(full_ids, dtype=torch.long)
    else:
        x = torch.tensor(full_ids, dtype=torch.long)
        prompt_len = len(prompt_ids)

    y_text = torch.full_like(x, fill_value=-1)
    base_w = torch.zeros_like(x, dtype=torch.float32)
    L = x.size(0)
    start = max(prompt_len - 1, 0)
    if start < L - 1:
        y_text[start:L - 1] = x[start + 1:L]
        base_w[start:L - 1] = 1.0

    if text_max_len and text_max_len > 0:
        pad_pos = (x == tok.eos_id).nonzero(as_tuple=False).view(-1)
        if pad_pos.numel() > 0:
            first_pad = int(pad_pos[0].item())
            cut = max(first_pad - 1, 0)
            y_text[cut:] = -1
            base_w[cut:] = 0.0
    return x, y_text, base_w, prompt_len


def _find_subseq(hay: List[int], needle: List[int]) -> int:
    if not needle or len(needle) > len(hay):
        return -1
    for i in range(0, len(hay) - len(needle) + 1):
        if hay[i:i + len(needle)] == needle:
            return i
    return -1


def _find_line_span_by_prefix(tok, x_ids: List[int], prefix_text: str, max_span: int = 256) -> Optional[Tuple[int, int]]:
    prefix_ids = tok.enc.encode(prefix_text, allowed_special={"<|endoftext|>"})
    s = _find_subseq(x_ids, prefix_ids)
    if s < 0:
        return None
    token_id_newline = tok.enc.encode("\n")[0]
    e = None
    for j in range(s, min(len(x_ids), s + max_span)):
        if x_ids[j] == token_id_newline:
            e = j + 1
            break
    if e is None:
        e = min(len(x_ids), s + max_span)
    return (s, e)


def task_weight_config(task_name: str, args) -> Dict[str, float]:
    scales = TASK_WEIGHT_SCALES[task_name]
    return {
        "task_weight": args.task_weight * scales["task_weight"],
        "source_weight": args.source_weight * scales["source_weight"],
        "state_weight": args.state_weight * scales["state_weight"],
        "temporal_weight": args.temporal_weight * scales["temporal_weight"],
        "spatial_weight": args.spatial_weight * scales["spatial_weight"],
        "morpho_weight": args.morpho_weight * scales["morpho_weight"],
        "answer_weight": args.answer_weight * scales["answer_weight"],
    }


def build_weight_vector_shared(tok, x_text_1d: torch.Tensor, base_w_1d: torch.Tensor, task_weight: float, source_weight: float, state_weight: float, temporal_weight: float, spatial_weight: float, morpho_weight: float, answer_weight: float) -> torch.Tensor:
    w = base_w_1d.clone()
    x_ids = x_text_1d.tolist()
    prefix_to_weight = {
        "TASK=": float(task_weight),
        "SOURCE=": float(source_weight),
        "STATE=": float(state_weight),
        "TEMPORAL=": float(temporal_weight),
        "SPATIAL=": float(spatial_weight),
        "MORPHO=": float(morpho_weight),
        "ANS=(": float(answer_weight),
    }
    for prefix, weight in prefix_to_weight.items():
        span = _find_line_span_by_prefix(tok, x_ids, prefix)
        if span is None:
            continue
        s, e = span
        active = base_w_1d[s:e] > 0
        w[s:e][active] = weight
    return w


def candidate_labels_from_factors_tuev(slots: Dict[str, str]) -> List[str]:
    s = slots["source"]
    t = slots["temporal"]
    p = slots["spatial"]
    m = slots["morpho"]

    if s == "background_like":
        if m == "background_rhythm" and t in {"stable_rhythm", "none_or_uncertain"}:
            return ["F"]
        if m == "slow_wave_like":
            return ["F", "A"]
        return ["F", "A"]

    if s == "noncerebral":
        if t == "slow_drift" and p == "frontal_dominant" and m == "drift_like":
            return ["D", "E"]
        if t == "broadband_irregular" or m == "noise_like":
            return ["E", "D"]
        return ["D", "E"]

    if s == "cerebral_event":
        if t == "periodic_repeating":
            if p == "generalized":
                return ["B", "C", "A"]
            if p == "lateralized":
                return ["C", "B", "A"]
            return ["A", "B", "C"]
        if t == "isolated_transient":
            if m == "spike_sharp_complex":
                if p in {"focal_local", "lateralized"}:
                    return ["A", "C", "B"]
                return ["A", "B", "C"]
            return ["A", "B", "C"]
        if t == "slow_drift" and m == "slow_wave_like":
            return ["F", "A", "E"]
        return ["A", "B", "C"]

    return ["A", "B", "C", "D", "E", "F"]


def candidate_labels_from_factors_tuab(slots: Dict[str, str]) -> List[str]:
    s = slots["source"]
    st = slots["state"]
    t = slots["temporal"]
    p = slots["spatial"]
    m = slots["morpho"]

    if s == "background_like":
        strong_normal = (
            st in {"wake_like", "transition_like", "stable_sleep_like", "rem_like"}
            and t in {"stable_rhythm", "slow_drift", "state_transition", "none_or_uncertain"}
            and p in {"na", "diffuse_mixed", "generalized", "frontal_dominant"}
            and m in {"background_rhythm", "slow_wave_like", "spindle_kcomplex_like", "mixed_low_voltage", "uncertain"}
        )
        if not strong_normal:
            return ["normal", "abnormal"]
        if st == "wake_like" and m == "background_rhythm" and t == "stable_rhythm":
            return ["normal"]
        if st == "stable_sleep_like" and m in {"spindle_kcomplex_like", "slow_wave_like"}:
            return ["normal"]
        if st in {"transition_like", "rem_like"} and m == "mixed_low_voltage":
            return ["normal", "abnormal"]
        return ["normal", "abnormal"]

    if s in {"noncerebral", "cerebral_event"}:
        return ["abnormal"]

    if t == "broadband_irregular" or m == "noise_like":
        return ["abnormal"]

    return ["abnormal", "normal"]


def candidate_labels_from_factors_hmc(slots: Dict[str, str]) -> List[str]:
    s = slots["source"]
    st = slots["state"]
    t = slots["temporal"]
    m = slots["morpho"]

    if s != "background_like":
        return ["wake", "n1", "rem", "n2", "n3"]

    if st == "wake_like":
        if m == "background_rhythm" and t == "stable_rhythm":
            return ["wake"]
        return ["wake", "n1", "rem"]

    if st == "stable_sleep_like":
        if m == "spindle_kcomplex_like":
            return ["n2", "n1"]
        if m == "slow_wave_like" and t == "slow_drift":
            return ["n3", "n2"]
        if t == "stable_rhythm":
            return ["n2", "n3"]
        return ["n2", "n3", "n1"]

    if st == "rem_like":
        return ["rem", "n1", "wake"]

    if st == "transition_like":
        if m == "mixed_low_voltage":
            return ["n1", "rem", "wake"]
        if m == "spindle_kcomplex_like":
            return ["n2", "n1"]
        if m == "slow_wave_like":
            return ["n3", "n2", "n1"]
        return ["n1", "rem", "n2"]

    return ["wake", "n1", "n2", "n3", "rem"]


def candidate_labels_from_factors_seed(slots: Dict[str, str]) -> List[str]:
    s = slots["source"]
    st = slots["state"]
    t = slots["temporal"]
    p = slots["spatial"]
    m = slots["morpho"]

    if s != "background_like":
        return ["positive", "neutral", "negative"]
    if m == "background_rhythm" and t == "stable_rhythm" and st == "wake_like":
        if p in {"diffuse_mixed", "na"}:
            return ["positive", "neutral"]
        return ["positive", "neutral", "negative"]
    if m == "slow_wave_like" or t == "slow_drift":
        if p == "frontal_dominant":
            return ["negative", "neutral"]
        return ["negative", "neutral", "positive"]
    if st == "transition_like" and m in {"mixed_low_voltage", "uncertain"}:
        return ["neutral", "negative", "positive"]
    return ["neutral", "positive", "negative"]


def candidate_labels_from_factors(task_name: str, slots: Dict[str, str]) -> List[str]:
    if task_name == "TUEV":
        return candidate_labels_from_factors_tuev(slots)
    if task_name == "TUAB":
        return candidate_labels_from_factors_tuab(slots)
    if task_name == "HMC":
        return candidate_labels_from_factors_hmc(slots)
    if task_name == "SEED":
        return candidate_labels_from_factors_seed(slots)
    raise ValueError(task_name)


def canonical_factors_for_label_tuev(y: int) -> Dict[str, str]:
    if y == 0:
        return {"source": "cerebral_event", "state": "na", "temporal": "isolated_transient", "spatial": "focal_local", "morpho": "spike_sharp_complex"}
    if y == 1:
        return {"source": "cerebral_event", "state": "na", "temporal": "periodic_repeating", "spatial": "generalized", "morpho": "spike_sharp_complex"}
    if y == 2:
        return {"source": "cerebral_event", "state": "na", "temporal": "periodic_repeating", "spatial": "lateralized", "morpho": "spike_sharp_complex"}
    if y == 3:
        return {"source": "noncerebral", "state": "na", "temporal": "slow_drift", "spatial": "frontal_dominant", "morpho": "drift_like"}
    if y == 4:
        return {"source": "noncerebral", "state": "na", "temporal": "broadband_irregular", "spatial": "diffuse_mixed", "morpho": "noise_like"}
    return {"source": "background_like", "state": "na", "temporal": "stable_rhythm", "spatial": "na", "morpho": "background_rhythm"}


def canonical_factors_for_label_tuab(y: int) -> Dict[str, str]:
    if int(y) == 0:
        return {"source": "background_like", "state": "wake_like", "temporal": "stable_rhythm", "spatial": "na", "morpho": "background_rhythm"}
    return {"source": "cerebral_event", "state": "na", "temporal": "slow_drift", "spatial": "diffuse_mixed", "morpho": "slow_wave_like"}


def canonical_factors_for_label_hmc(y: int) -> Dict[str, str]:
    if int(y) == 0:
        return {"source": "background_like", "state": "wake_like", "temporal": "stable_rhythm", "spatial": "na", "morpho": "background_rhythm"}
    if int(y) == 1:
        return {"source": "background_like", "state": "transition_like", "temporal": "state_transition", "spatial": "diffuse_mixed", "morpho": "mixed_low_voltage"}
    if int(y) == 2:
        return {"source": "background_like", "state": "stable_sleep_like", "temporal": "stable_rhythm", "spatial": "diffuse_mixed", "morpho": "spindle_kcomplex_like"}
    if int(y) == 3:
        return {"source": "background_like", "state": "stable_sleep_like", "temporal": "slow_drift", "spatial": "diffuse_mixed", "morpho": "slow_wave_like"}
    return {"source": "background_like", "state": "rem_like", "temporal": "state_transition", "spatial": "diffuse_mixed", "morpho": "mixed_low_voltage"}


def canonical_factors_for_label_seed(y: int) -> Dict[str, str]:
    if int(y) == 0:
        return {"source": "background_like", "state": "wake_like", "temporal": "stable_rhythm", "spatial": "diffuse_mixed", "morpho": "background_rhythm"}
    if int(y) == 1:
        return {"source": "background_like", "state": "transition_like", "temporal": "stable_rhythm", "spatial": "diffuse_mixed", "morpho": "mixed_low_voltage"}
    return {"source": "background_like", "state": "transition_like", "temporal": "slow_drift", "spatial": "frontal_dominant", "morpho": "slow_wave_like"}


def canonical_factors_for_label(task_name: str, y: int) -> Dict[str, str]:
    if task_name == "TUEV":
        return canonical_factors_for_label_tuev(y)
    if task_name == "TUAB":
        return canonical_factors_for_label_tuab(y)
    if task_name == "HMC":
        return canonical_factors_for_label_hmc(y)
    if task_name == "SEED":
        return canonical_factors_for_label_seed(y)
    raise ValueError(task_name)


def repair_factors_to_include_label(task_name: str, slots: Dict[str, str], y: int) -> Dict[str, str]:
    gold = label_space_for_task(task_name)[int(y)]
    if gold in candidate_labels_from_factors(task_name, slots):
        return slots

    canon = canonical_factors_for_label(task_name, y)
    repaired = dict(slots)
    for key in ["source", "state", "temporal", "spatial", "morpho"]:
        repaired[key] = canon[key]
        if gold in candidate_labels_from_factors(task_name, repaired):
            return repaired
    return canon


def _map_temporal_extended(feats: Dict[str, float], src: str, temp_old: str, morpho_old: str) -> str:
    if src == "background_like":
        alpha = float(feats.get("alpha", np.nan))
        if morpho_old == "background_rhythm" or (np.isfinite(alpha) and alpha > 0.18):
            return "stable_rhythm"
        return "none_or_uncertain"
    return temp_old


def _map_morpho_extended(feats: Dict[str, float], src: str, temp_new: str, spatial: str, morpho_old: str) -> str:
    if src == "background_like" and temp_new == "stable_rhythm":
        return "background_rhythm"
    delta = float(feats.get("delta", np.nan))
    alpha = float(feats.get("alpha", np.nan))
    if src == "cerebral_event" and temp_new == "slow_drift" and np.isfinite(delta) and np.isfinite(alpha) and delta > max(0.18, alpha + 0.03):
        return "slow_wave_like"
    return morpho_old


def compute_slots_tuev_multitask(x: Optional[torch.Tensor], y: int, t1: float, t2: float) -> Dict[str, str]:
    feats = compute_decision_features(x, t1=t1, t2=t2)
    src = source_label(feats)
    temp_old = temporal_label(feats, src)
    spa = spatial_label(feats, src, temp_old)
    morph_old = morpho_label(feats, src, temp_old, spa)
    temp_new = _map_temporal_extended(feats, src, temp_old, morph_old)
    morph_new = _map_morpho_extended(feats, src, temp_new, spa, morph_old)
    slots = {"source": src, "state": "na", "temporal": temp_new, "spatial": spa, "morpho": morph_new}
    return repair_factors_to_include_label("TUEV", slots, y)


def compute_slots_tuab_multitask(x: Optional[torch.Tensor], y: int, t1: float, t2: float) -> Dict[str, str]:
    feats = compute_decision_features(x, t1=t1, t2=t2)
    sleep_feats = compute_sleep_features(x, feats)

    src = source_label(feats)
    temp_old = temporal_label(feats, src)
    spa_old = spatial_label(feats, src, temp_old)
    morph_old = morpho_label(feats, src, temp_old, spa_old)

    if src == "background_like":
        state = infer_background_state(feats, sleep_feats)
        temp_new = temporal_from_state(state, feats, sleep_feats)
        spa_new = spatial_from_state(state, feats, sleep_feats)
        morph_new = morpho_from_state(state, feats, sleep_feats)
    elif src == "noncerebral":
        state = "na"
        temp_new = "slow_drift" if temp_old == "slow_drift" else "broadband_irregular"
        spa_new = "frontal_dominant" if (temp_new == "slow_drift" and feats["frontal_ratio"] > 1.35) else "diffuse_mixed"
        morph_new = "drift_like" if temp_new == "slow_drift" else "noise_like"
    elif src == "cerebral_event":
        state = "na"
        if temp_old in {"isolated_transient", "periodic_repeating"}:
            temp_new = temp_old
            morph_new = "spike_sharp_complex"
            spa_new = spa_old if spa_old != "na" else "focal_local"
        else:
            delta = float(feats.get("delta", np.nan))
            alpha = float(feats.get("alpha", np.nan))
            use_slow = np.isfinite(delta) and np.isfinite(alpha) and delta > max(0.18, alpha + 0.03)
            temp_new = "slow_drift" if use_slow else "isolated_transient"
            morph_new = "slow_wave_like" if temp_new == "slow_drift" else "spike_sharp_complex"
            spa_new = spa_old if spa_old != "na" else "diffuse_mixed"
    else:
        state_guess = infer_background_state(feats, sleep_feats)
        if state_guess in {"wake_like", "transition_like", "stable_sleep_like", "rem_like"}:
            src = "background_like"
            state = state_guess
            temp_new = temporal_from_state(state, feats, sleep_feats)
            spa_new = spatial_from_state(state, feats, sleep_feats)
            morph_new = morpho_from_state(state, feats, sleep_feats)
        else:
            state = "uncertain"
            temp_new = "none_or_uncertain"
            spa_new = "diffuse_mixed"
            morph_new = "uncertain"

    slots = {"source": src, "state": state, "temporal": temp_new, "spatial": spa_new, "morpho": morph_new}
    return repair_factors_to_include_label("TUAB", slots, y)


def compute_slots_hmc_multitask(x: Optional[torch.Tensor], y: int, t1: float, t2: float) -> Dict[str, str]:
    feats = compute_decision_features(x, t1=t1, t2=t2)
    sleep_feats = compute_sleep_features(x, feats)

    art = float(feats.get("artifact_score", np.nan))
    hf = float(feats.get("hf", np.nan))
    clip_ratio = float(feats.get("clip_ratio", np.nan))
    flatline_ratio = float(feats.get("flatline_ratio", np.nan))
    severe_artifact = (
        (np.isfinite(art) and art > max(t2 * 1.15, t2 + 0.02))
        or (np.isfinite(hf) and hf > 0.18)
        or (np.isfinite(clip_ratio) and clip_ratio > 0.04)
        or (np.isfinite(flatline_ratio) and flatline_ratio > 0.30)
    )

    if severe_artifact:
        slots = {"source": "noncerebral", "state": "uncertain", "temporal": "broadband_irregular", "spatial": "diffuse_mixed", "morpho": "noise_like"}
    else:
        state = infer_background_state(feats, sleep_feats)
        slots = {
            "source": "background_like",
            "state": state,
            "temporal": temporal_from_state(state, feats, sleep_feats),
            "spatial": spatial_from_state(state, feats, sleep_feats),
            "morpho": morpho_from_state(state, feats, sleep_feats),
        }
        if state == "uncertain":
            slots["temporal"] = "none_or_uncertain"
            slots["spatial"] = "diffuse_mixed"
            slots["morpho"] = "uncertain"
    return repair_factors_to_include_label("HMC", slots, y)


def compute_slots_seed_multitask(x: Optional[torch.Tensor], y: int, t1: float, t2: float) -> Dict[str, str]:
    feats = compute_decision_features(x, t1=t1, t2=t2)
    sleep_feats = compute_sleep_features(x, feats)

    art = float(feats.get("artifact_score", np.nan))
    hf = float(feats.get("hf", np.nan))
    clip_ratio = float(feats.get("clip_ratio", np.nan))
    flatline_ratio = float(feats.get("flatline_ratio", np.nan))
    alpha = float(feats.get("alpha", np.nan))
    delta = float(feats.get("delta", np.nan))
    frontal_ratio = float(feats.get("frontal_ratio", np.nan))
    slow_ratio = float(sleep_feats.get("slow_ratio", np.nan))
    low_voltage = float(sleep_feats.get("low_voltage_score", np.nan))

    severe_artifact = (
        (np.isfinite(art) and art > max(t2 * 1.10, t2 + 0.02))
        or (np.isfinite(hf) and hf > 0.18)
        or (np.isfinite(clip_ratio) and clip_ratio > 0.04)
        or (np.isfinite(flatline_ratio) and flatline_ratio > 0.30)
    )
    if severe_artifact:
        slots = {"source": "uncertain", "state": "uncertain", "temporal": "none_or_uncertain", "spatial": "diffuse_mixed", "morpho": "uncertain"}
        return repair_factors_to_include_label("SEED", slots, y)

    src = "background_like"
    use_negative = (
        (np.isfinite(slow_ratio) and slow_ratio > 0.58)
        or (np.isfinite(delta) and np.isfinite(alpha) and delta > max(0.20, alpha + 0.05))
        or (np.isfinite(frontal_ratio) and frontal_ratio > 1.22)
    )
    use_positive = (
        (np.isfinite(alpha) and alpha > 0.20)
        and (not np.isfinite(slow_ratio) or slow_ratio < 0.52)
        and (not np.isfinite(frontal_ratio) or frontal_ratio <= 1.22)
    )

    if use_positive:
        slots = {"source": src, "state": "wake_like", "temporal": "stable_rhythm", "spatial": "diffuse_mixed", "morpho": "background_rhythm"}
    elif use_negative:
        slots = {"source": src, "state": "transition_like", "temporal": "slow_drift", "spatial": "frontal_dominant", "morpho": "slow_wave_like"}
    else:
        morpho = "mixed_low_voltage" if (np.isfinite(low_voltage) and low_voltage > 0.55) else "uncertain"
        slots = {"source": src, "state": "transition_like", "temporal": "stable_rhythm", "spatial": "diffuse_mixed", "morpho": morpho}
    return repair_factors_to_include_label("SEED", slots, y)


def compute_slots(task_name: str, x: Optional[torch.Tensor], y: int, t1: float, t2: float) -> Dict[str, str]:
    if task_name == "TUEV":
        return compute_slots_tuev_multitask(x, y, t1, t2)
    if task_name == "TUAB":
        return compute_slots_tuab_multitask(x, y, t1, t2)
    if task_name == "HMC":
        return compute_slots_hmc_multitask(x, y, t1, t2)
    if task_name == "SEED":
        return compute_slots_seed_multitask(x, y, t1, t2)
    raise ValueError(task_name)


def safe_decode(tok, ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().view(-1).tolist()
    filt = [int(t) for t in ids if 0 <= int(t) <= int(tok.eos_id) and int(t) != int(tok.sep_id)]
    return tok.enc.decode(filt) if len(filt) > 0 else ""


def source_option_values(task_name: str) -> List[str]:
    if task_name == "HMC":
        return ["background_like", "noncerebral", "uncertain"]
    if task_name == "SEED":
        return ["background_like", "uncertain"]
    return list(SOURCE_VALUES)


def state_option_values(task_name: str) -> List[str]:
    if task_name == "TUEV":
        return ["na"]
    if task_name == "HMC":
        return ["wake_like", "transition_like", "stable_sleep_like", "rem_like", "uncertain"]
    if task_name == "SEED":
        return ["wake_like", "transition_like", "uncertain"]
    return ["wake_like", "transition_like", "stable_sleep_like", "rem_like", "na", "uncertain"]


def temporal_option_values(task_name: str) -> List[str]:
    if task_name == "TUEV":
        return ["isolated_transient", "periodic_repeating", "slow_drift", "broadband_irregular", "stable_rhythm", "none_or_uncertain"]
    if task_name == "HMC":
        return ["stable_rhythm", "slow_drift", "state_transition", "broadband_irregular", "none_or_uncertain"]
    if task_name == "SEED":
        return ["stable_rhythm", "slow_drift", "none_or_uncertain"]
    return list(TEMPORAL_VALUES)


def spatial_option_values(task_name: str) -> List[str]:
    if task_name == "HMC":
        return ["diffuse_mixed", "frontal_dominant", "na"]
    if task_name == "SEED":
        return ["diffuse_mixed", "frontal_dominant", "na"]
    return list(SPATIAL_VALUES)


def morpho_option_values(task_name: str) -> List[str]:
    if task_name == "TUEV":
        return ["spike_sharp_complex", "drift_like", "noise_like", "background_rhythm", "slow_wave_like", "uncertain", "na"]
    if task_name == "HMC":
        return ["background_rhythm", "slow_wave_like", "spindle_kcomplex_like", "mixed_low_voltage", "noise_like", "uncertain", "na"]
    if task_name == "SEED":
        return ["background_rhythm", "slow_wave_like", "mixed_low_voltage", "uncertain", "na"]
    return list(MORPHO_VALUES)


@torch.no_grad()
def _decode_one_line_from_options(model, tok, x_eeg_emb_1, X_eeg_1, input_time_1, eeg_mask_1, x_text_1d, option_strs: List[str], temperature: float, greedy: bool):
    device = x_text_1d.device
    trie = build_trie([tok.enc.encode(s, allowed_special={"<|endoftext|>"}) for s in option_strs])
    emitted: List[int] = []
    node = trie
    x_text = x_text_1d.unsqueeze(0)
    while True:
        gpt_mask = build_gpt_mask_multitask(X_eeg_1.unsqueeze(0), x_text, eeg_mask_1.unsqueeze(0), input_time_1.unsqueeze(0))
        logits, _, _ = model.GPT2(
            x_eeg=x_eeg_emb_1.unsqueeze(0),
            x_text=x_text,
            y_text=None,
            eeg_time_idx=input_time_1.unsqueeze(0),
            eeg_mask=eeg_mask_1.unsqueeze(0),
            eeg_text_mask=gpt_mask,
        )
        logits_next = logits[0, -1, :50257]
        allowed = trie_next_allowed(node)
        tid = sample_or_greedy_allowed(logits_next, allowed, temperature, greedy=greedy)
        emitted.append(tid)
        node = trie_step(node, tid)
        x_text = torch.cat([x_text, torch.tensor([[tid]], device=device, dtype=torch.long)], dim=1)
        if node.is_end:
            break
    return x_text[0], safe_decode(tok, emitted)


@torch.no_grad()
def _score_completion_normalized_logprob(
    model,
    tok,
    x_eeg_emb_1: torch.Tensor,
    X_eeg_1: torch.Tensor,
    input_time_1: torch.Tensor,
    eeg_mask_1: torch.Tensor,
    x_text_1d: torch.Tensor,
    completion_text: str,
    temperature: float,
) -> float:
    ids = tok.enc.encode(completion_text, allowed_special={"<|endoftext|>"})
    if not ids:
        return float("-inf")

    x_text = x_text_1d.unsqueeze(0)
    total = 0.0
    for tid in ids:
        gpt_mask = build_gpt_mask_multitask(
            X_eeg_1.unsqueeze(0),
            x_text,
            eeg_mask_1.unsqueeze(0),
            input_time_1.unsqueeze(0),
        )
        logits, _, _ = model.GPT2(
            x_eeg=x_eeg_emb_1.unsqueeze(0),
            x_text=x_text,
            y_text=None,
            eeg_time_idx=input_time_1.unsqueeze(0),
            eeg_mask=eeg_mask_1.unsqueeze(0),
            eeg_text_mask=gpt_mask,
        )
        logits_next = logits[0, -1, :50257] / max(float(temperature), 1e-6)
        total += float(F.log_softmax(logits_next, dim=-1)[int(tid)].item())
        x_text = torch.cat([x_text, torch.tensor([[int(tid)]], device=x_text.device, dtype=torch.long)], dim=1)

    return float(total / max(len(ids), 1))


@torch.no_grad()
def _tuab_abnormal_score_from_context(
    model,
    tok,
    x_eeg_emb_1: torch.Tensor,
    X_eeg_1: torch.Tensor,
    input_time_1: torch.Tensor,
    eeg_mask_1: torch.Tensor,
    x_text_1d: torch.Tensor,
    temperature: float,
) -> Optional[float]:
    scores = [
        _score_completion_normalized_logprob(model, tok, x_eeg_emb_1, X_eeg_1, input_time_1, eeg_mask_1, x_text_1d, "ANS=(normal)\n<|endoftext|>", temperature),
        _score_completion_normalized_logprob(model, tok, x_eeg_emb_1, X_eeg_1, input_time_1, eeg_mask_1, x_text_1d, "ANS=(abnormal)\n<|endoftext|>", temperature),
    ]
    if not np.all(np.isfinite(np.array(scores, dtype=np.float64))):
        return None
    return float(torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0)[1].item())


@torch.no_grad()
def generate_constrained_chain_multitask(model, tok, task_name: str, X_eeg_tokens: torch.Tensor, input_chans: torch.Tensor, input_time: torch.Tensor, eeg_mask: torch.Tensor, text_max_len: int, temperature: float, greedy: bool, print_samples: int = 0):
    device = X_eeg_tokens.device
    B = X_eeg_tokens.size(0)
    x_eeg_emb = encode_eeg_only(model, X_eeg_tokens, input_chans, input_time, eeg_mask)
    prompt_ids = [tok.sep_id] + tok.enc.encode(prompt_str_shared(task_name), allowed_special={"<|endoftext|>"})
    if text_max_len > 0:
        prompt_ids = prompt_ids[-text_max_len:]

    task_opts = [f"TASK={task_name}\n"]
    source_opts = [f"SOURCE={v}\n" for v in source_option_values(task_name)]
    state_opts = [f"STATE={v}\n" for v in state_option_values(task_name)]
    temp_opts = [f"TEMPORAL={v}\n" for v in temporal_option_values(task_name)]
    spatial_opts = [f"SPATIAL={v}\n" for v in spatial_option_values(task_name)]
    morpho_opts = [f"MORPHO={v}\n" for v in morpho_option_values(task_name)]

    results: List[DecodeResult] = []
    shown = 0
    labels = label_space_for_task(task_name)
    for b in range(B):
        x_text = torch.tensor(prompt_ids, device=device, dtype=torch.long)
        x_text, _ = _decode_one_line_from_options(model, tok, x_eeg_emb[b], X_eeg_tokens[b], input_time[b], eeg_mask[b], x_text, task_opts, temperature, greedy)
        x_text, source_line = _decode_one_line_from_options(model, tok, x_eeg_emb[b], X_eeg_tokens[b], input_time[b], eeg_mask[b], x_text, source_opts, temperature, greedy)
        source = source_line.strip().split("=", 1)[1]
        x_text, state_line = _decode_one_line_from_options(model, tok, x_eeg_emb[b], X_eeg_tokens[b], input_time[b], eeg_mask[b], x_text, state_opts, temperature, greedy)
        state = state_line.strip().split("=", 1)[1]
        x_text, temp_line = _decode_one_line_from_options(model, tok, x_eeg_emb[b], X_eeg_tokens[b], input_time[b], eeg_mask[b], x_text, temp_opts, temperature, greedy)
        temporal = temp_line.strip().split("=", 1)[1]
        x_text, spa_line = _decode_one_line_from_options(model, tok, x_eeg_emb[b], X_eeg_tokens[b], input_time[b], eeg_mask[b], x_text, spatial_opts, temperature, greedy)
        spatial = spa_line.strip().split("=", 1)[1]
        x_text, mor_line = _decode_one_line_from_options(model, tok, x_eeg_emb[b], X_eeg_tokens[b], input_time[b], eeg_mask[b], x_text, morpho_opts, temperature, greedy)
        morpho = mor_line.strip().split("=", 1)[1]

        slots = {"source": source, "state": state, "temporal": temporal, "spatial": spatial, "morpho": morpho}
        cand = candidate_labels_from_factors(task_name, slots)
        ans_opts = [f"ANS=({v})\n<|endoftext|>" for v in cand]
        score_abnormal = None
        if task_name == "TUAB":
            score_abnormal = _tuab_abnormal_score_from_context(
                model, tok, x_eeg_emb[b], X_eeg_tokens[b], input_time[b], eeg_mask[b], x_text, temperature
            )
        x_text, ans_line = _decode_one_line_from_options(model, tok, x_eeg_emb[b], X_eeg_tokens[b], input_time[b], eeg_mask[b], x_text, ans_opts, temperature, greedy)
        ans_text = ans_line.strip().split("(", 1)[1].split(")", 1)[0]

        pred_idx = labels.index(ans_text) if ans_text in labels else 0
        tail = safe_decode(tok, x_text)
        if "DSL:\n" in tail:
            tail = tail.split("DSL:\n", 1)[1]
        results.append(DecodeResult(pred_idx=pred_idx, pred_label=ans_text, slots=slots, raw_tail=tail, score_abnormal=score_abnormal))

        if shown < print_samples:
            shown += 1
            print(f"----- sample gen tail ({task_name}, shared-chain SFT v3) -----")
            print(tail)
    return results


def chain_eval_flags(task_name: str, pred_label: str, slots: Dict[str, str]) -> Tuple[bool, bool, bool, List[str]]:
    """Return FV/KCS/CAR flags plus chain-derived candidate labels."""
    slot_options = {
        "source": set(source_option_values(task_name)),
        "state": set(state_option_values(task_name)),
        "temporal": set(temporal_option_values(task_name)),
        "spatial": set(spatial_option_values(task_name)),
        "morpho": set(morpho_option_values(task_name)),
    }
    label_set = set(label_space_for_task(task_name))
    format_ok = pred_label in label_set
    for key, allowed in slot_options.items():
        format_ok = format_ok and key in slots and str(slots.get(key)) in allowed

    cand = candidate_labels_from_factors(task_name, slots) if format_ok else []
    kcs_ok = format_ok and len(cand) > 0 and all(c in label_set for c in cand)
    car_bad = format_ok and pred_label not in cand
    return bool(format_ok), bool(kcs_ok), bool(car_bad), cand


def chain_specificity_score(task_name: str, cand: List[str], format_ok: bool) -> float:
    """How much the chain narrows the task label space."""
    labels = set(label_space_for_task(task_name))
    n_labels = len(labels)
    if (not format_ok) or n_labels <= 1:
        return 0.0
    n_cand = len({c for c in cand if c in labels})
    if n_cand <= 0:
        return 0.0
    n_cand = min(n_cand, n_labels)
    return float(1.0 - (n_cand - 1) / max(n_labels - 1, 1))


@torch.no_grad()
def eval_multitask_loader(
    model,
    tok,
    task_name: str,
    loader,
    device,
    text_max_len: int,
    temperature: float,
    print_samples: int = 0,
    progress_label: str = "",
    progress_every: int = 0,
):
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    tuab_scores: List[float] = []
    incons_count = 0
    format_valid_count = 0
    kcs_count = 0
    car_count = 0
    specificity_sum = 0.0
    n_total = 0
    shown = 0
    t0 = time.time()
    total_batches = len(loader) if hasattr(loader, "__len__") else None

    for batch_idx, batch in enumerate(loader, start=1):
        X_eeg, _t, Y, input_chans, input_time, eeg_mask, _gm = batch
        X_eeg = X_eeg.to(device)
        Y = Y.to(device)
        input_chans = input_chans.to(device)
        input_time = input_time.to(device)
        eeg_mask = eeg_mask.to(device)

        dec = generate_constrained_chain_multitask(
            model=model,
            tok=tok,
            task_name=task_name,
            X_eeg_tokens=X_eeg,
            input_chans=input_chans,
            input_time=input_time,
            eeg_mask=eeg_mask,
            text_max_len=text_max_len,
            temperature=temperature,
            greedy=True,
            print_samples=max(0, print_samples - shown),
        )
        shown += min(max(0, print_samples - shown), len(dec))

        yt = Y.detach().cpu().numpy().astype(np.int64)
        labels = label_space_for_task(task_name)
        for i, one in enumerate(dec):
            y_true.append(int(yt[i]))
            y_pred.append(int(one.pred_idx))
            if task_name == "TUAB" and one.score_abnormal is not None:
                tuab_scores.append(float(one.score_abnormal))
            gold_label = labels[int(yt[i])]
            format_ok, kcs_ok, car_bad, cand = chain_eval_flags(task_name, one.pred_label, one.slots)
            format_valid_count += int(format_ok)
            kcs_count += int(kcs_ok)
            car_count += int(car_bad)
            specificity_sum += chain_specificity_score(task_name, cand, format_ok)
            if gold_label not in cand:
                incons_count += 1
            n_total += 1

        if int(progress_every) > 0 and (
            batch_idx % int(progress_every) == 0
            or (total_batches is not None and batch_idx == total_batches)
        ):
            elapsed = max(time.time() - t0, 1e-9)
            speed = float(n_total) / elapsed
            total_text = str(total_batches) if total_batches is not None else "?"
            prefix = progress_label or task_name
            print(
                f"[progress:{prefix}] batch={batch_idx}/{total_text} "
                f"done={n_total} elapsed={elapsed:.1f}s speed={speed:.2f} samples/s"
            )

    y_true_np = np.array(y_true, dtype=np.int64)
    y_pred_np = np.array(y_pred, dtype=np.int64)
    label_ids = list(range(len(label_space_for_task(task_name))))
    rec = recall_score(y_true_np, y_pred_np, average=None, labels=label_ids, zero_division=0)
    cm = confusion_matrix(y_true_np, y_pred_np, labels=label_ids).tolist()
    out = {
        "n_eval": int(len(y_true_np)),
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_np, y_pred_np)),
        "f1_weighted": float(f1_score(y_true_np, y_pred_np, average="weighted")),
        "format_validity": float(format_valid_count / max(n_total, 1)),
        "knowledge_constraint_satisfaction": float(kcs_count / max(n_total, 1)),
        "chain_answer_contradiction_rate": float(car_count / max(n_total, 1)),
        "specificity": float(specificity_sum / max(n_total, 1)),
        "chain_inconsistency_rate": float(incons_count / max(n_total, 1)),
        "recall_per_class": rec.tolist(),
        "confusion_matrix": cm,
    }
    if len(label_ids) > 2:
        out["cohen_kappa"] = float(cohen_kappa_score(y_true_np, y_pred_np))
    if task_name == "TUAB" and len(tuab_scores) == len(y_true_np) and len(np.unique(y_true_np)) == 2:
        score_np = np.array(tuab_scores, dtype=np.float64)
        out["auc_pr"] = float(average_precision_score(y_true_np, score_np))
        out["auroc"] = float(roc_auc_score(y_true_np, score_np))
    return out


def print_eval(prefix: str, res: Dict[str, object]):
    extra = (
        f" FV={res.get('format_validity', 0.0):.4f}"
        f" KCS={res.get('knowledge_constraint_satisfaction', 0.0):.4f}"
        f" CAR={res.get('chain_answer_contradiction_rate', 0.0):.4f}"
        f" Specificity={res.get('specificity', 0.0):.4f}"
        f" chain_inconsistency_rate={res['chain_inconsistency_rate']:.4f}"
    )
    if "auc_pr" in res and "auroc" in res:
        extra = f" auc_pr={res['auc_pr']:.4f} auroc={res['auroc']:.4f}" + extra
    if "cohen_kappa" in res:
        extra = f" cohen_kappa={res['cohen_kappa']:.4f}" + extra
    print(
        f"[{prefix}] n={res['n_eval']} accuracy={res['accuracy']:.4f} "
        f"balanced_accuracy={res['balanced_accuracy']:.4f} f1_weighted={res['f1_weighted']:.4f}{extra} "
        f"recall={['{:.4f}'.format(x) for x in res['recall_per_class']]}"
    )
    print(f"[{prefix}] confusion_matrix=")
    for row in res["confusion_matrix"]:
        print(row)


def parse_joint_weights(text: str) -> Dict[str, float]:
    parts = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(parts) != 4:
        parts = [0.25, 0.25, 0.25, 0.25]
    total = max(sum(parts), 1e-8)
    vals = [p / total for p in parts]
    return {"TUAB": vals[0], "TUEV": vals[1], "HMC": vals[2], "SEED": vals[3]}


def joint_score(res_tuab: Dict[str, object], res_tuev: Dict[str, object], res_hmc: Dict[str, object], res_seed: Dict[str, object], weights: Dict[str, float]) -> float:
    return (
        weights["TUAB"] * float(res_tuab["balanced_accuracy"])
        + weights["TUEV"] * float(res_tuev["balanced_accuracy"])
        + weights["HMC"] * float(res_hmc["balanced_accuracy"])
        + weights["SEED"] * float(res_seed["balanced_accuracy"])
    )


def next_batch(iters: Dict[str, Iterable], loaders: Dict[str, DataLoader], task_name: str):
    try:
        return next(iters[task_name])
    except StopIteration:
        iters[task_name] = iter(loaders[task_name])
        return next(iters[task_name])


def build_mix_schedule(task_mix: str) -> List[str]:
    parts = [int(x.strip()) for x in task_mix.split(",") if x.strip()]
    if len(parts) != 4 or min(parts) <= 0:
        raise ValueError("--task_mix must look like 1,1,1,1 or 1,2,1,1")
    return ["TUAB"] * parts[0] + ["TUEV"] * parts[1] + ["HMC"] * parts[2] + ["SEED"] * parts[3]


def save_ckpt(path: str, payload: Dict[str, object]) -> None:
    torch.save(payload, path)


def load_ckpt_weights(path: str, model: NeuroLM, device: torch.device) -> Dict[str, object]:
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get("model", ckpt.get("state_dict", {}))
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tuab_root", type=str, required=True)
    ap.add_argument("--tuev_root", type=str, required=True)
    ap.add_argument("--hmc_root", type=str, required=True)
    ap.add_argument("--seed_root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--eeg_max_len", type=int, default=1024)
    ap.add_argument("--text_max_len", type=int, default=768)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.set_defaults(balanced_sampler=False)
    ap.add_argument("--balanced_sampler", dest="balanced_sampler", action="store_true", help="Use class-balanced weighted sampling for training.")
    ap.add_argument("--no_balanced_sampler", dest="balanced_sampler", action="store_false", help="Disable class-balanced weighted sampling for training.")
    ap.add_argument("--task_mix", type=str, default="1,1,1,1", help="TUAB,TUEV,HMC,SEED sampling ratio")
    ap.add_argument("--eval_batch_size", type=int, default=0, help="Evaluation batch size. If <=0, use max(batch_size*2, batch_size*3//2).")
    ap.add_argument("--eval_progress_every", type=int, default=10, help="Print validation/test progress every N batches. Set 0 to disable.")

    ap.add_argument("--sq_estimate_batches", type=int, default=50)
    ap.add_argument("--sq_q1", type=float, default=0.50)
    ap.add_argument("--sq_q2", type=float, default=0.85)

    ap.add_argument("--sft_steps", type=int, default=4000)
    ap.add_argument("--sft_lr", type=float, default=1e-5)
    ap.add_argument("--eval_every", type=int, default=500)

    ap.add_argument("--task_weight", type=float, default=1.0)
    ap.add_argument("--source_weight", type=float, default=1.4)
    ap.add_argument("--state_weight", type=float, default=1.2)
    ap.add_argument("--temporal_weight", type=float, default=1.2)
    ap.add_argument("--spatial_weight", type=float, default=1.0)
    ap.add_argument("--morpho_weight", type=float, default=1.2)
    ap.add_argument("--answer_weight", type=float, default=3.0)

    ap.add_argument("--eval_temperature", type=float, default=0.7)
    ap.add_argument("--print_eval_samples", type=int, default=2)
    ap.add_argument("--val_eval_mode", type=str, default="balanced", choices=["balanced", "full", "proportional"])
    ap.add_argument("--val_eval_per_class", type=int, default=256, help="Validation samples per class when --val_eval_mode=balanced")
    ap.add_argument("--val_eval_max_samples", type=int, default=0, help="Validation max total samples when --val_eval_mode=proportional")
    ap.add_argument("--val_eval_min_per_class", type=int, default=0, help="Validation minimum samples per non-empty class when --val_eval_mode=proportional")
    ap.add_argument("--test_eval_mode", type=str, default="proportional", choices=["balanced", "full", "proportional"])
    ap.add_argument("--test_eval_per_class", type=int, default=500, help="Test samples per class when --test_eval_mode=balanced")
    ap.add_argument("--test_eval_max_samples", type=int, default=12000, help="Test max total samples when --test_eval_mode=proportional")
    ap.add_argument("--test_eval_min_per_class", type=int, default=200, help="Test minimum samples per non-empty class when --test_eval_mode=proportional")
    ap.add_argument("--joint_weights", type=str, default="0.25,0.25,0.25,0.25", help="TUAB,TUEV,HMC,SEED weights for joint validation score")

    ap.add_argument("--save_dir", type=str, default="runs_multitask_sft_v3")
    ap.add_argument("--log_dir", type=str, default="logs")
    ap.add_argument("--log_prefix", type=str, default="multitask_chain_sft_tuab_tuev_hmc_seed_v3")
    args = ap.parse_args()

    set_seed(args.seed)
    if setup_run_logging is not None:
        setup_run_logging(args.log_dir, prefix=args.log_prefix)

    device = torch.device(args.device)
    tok = build_tokmap()
    joint_weights = parse_joint_weights(args.joint_weights)
    task_names = ["TUAB", "TUEV", "HMC", "SEED"]
    specs = {
        "TUAB": build_task_spec("TUAB", args.tuab_root),
        "TUEV": build_task_spec("TUEV", args.tuev_root),
        "HMC": build_task_spec("HMC", args.hmc_root),
        "SEED": build_task_spec("SEED", args.seed_root),
    }

    data = {}
    sampler_weights = {}
    for task_name in ["TUAB", "TUEV", "HMC"]:
        spec = specs[task_name]
        train_files = list_pkls(spec.train_dir)
        val_files = list_pkls(spec.val_dir)
        test_files = list_pkls(spec.test_dir) if os.path.isdir(spec.test_dir) else []
        counts, weights = scan_labels_for_sampler(task_name, spec.train_dir, train_files, len(spec.labels))
        sampler_weights[task_name] = weights
        data[task_name] = {
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files,
            "train_class_counts": counts,
        }
        print(
            f"[{task_name}] train={len(train_files)} val={len(val_files)} test={len(test_files)} "
            f"train_class_counts={counts}"
        )

    ds_seed_train = build_seed_dataset(specs["SEED"].seed_h5_path, "train", args.eeg_max_len, args.text_max_len)
    ds_seed_val = build_seed_dataset(specs["SEED"].seed_h5_path, "val", args.eeg_max_len, args.text_max_len)
    ds_seed_test = build_seed_dataset(specs["SEED"].seed_h5_path, "test", args.eeg_max_len, args.text_max_len)
    seed_train_buckets, seed_train_counts = build_seed_index_buckets(ds_seed_train)
    seed_val_buckets, seed_val_counts = build_seed_index_buckets(ds_seed_val)
    seed_test_buckets, seed_test_counts = build_seed_index_buckets(ds_seed_test)
    sampler_weights["SEED"] = seed_sampler_weights_from_buckets(ds_seed_train, seed_train_buckets)
    data["SEED"] = {
        "train_files": list(range(len(ds_seed_train))),
        "val_files": list(range(len(ds_seed_val))),
        "test_files": list(range(len(ds_seed_test))),
        "train_class_counts": seed_train_counts,
    }
    print(
        f"[SEED] train={len(ds_seed_train)} val={len(ds_seed_val)} test={len(ds_seed_test)} "
        f"train_class_counts={seed_train_counts}"
    )

    ds_train = {
        "TUAB": specs["TUAB"].loader_cls(
            specs["TUAB"].train_dir,
            data["TUAB"]["train_files"],
            eeg_max_len=args.eeg_max_len,
            text_max_len=args.text_max_len,
            is_instruct=True,
            is_val=True,
        ),
        "TUEV": specs["TUEV"].loader_cls(
            specs["TUEV"].train_dir,
            data["TUEV"]["train_files"],
            eeg_max_len=args.eeg_max_len,
            text_max_len=args.text_max_len,
            is_instruct=True,
            is_val=True,
        ),
        "HMC": specs["HMC"].loader_cls(
            specs["HMC"].train_dir,
            data["HMC"]["train_files"],
            eeg_max_len=args.eeg_max_len,
            text_max_len=args.text_max_len,
            is_instruct=True,
            is_val=True,
        ),
        "SEED": ds_seed_train,
    }

    val_buckets = {}
    val_counts = {}
    test_buckets = {}
    test_counts = {}
    for task_name in ["TUAB", "TUEV", "HMC"]:
        spec = specs[task_name]
        val_buckets[task_name], val_counts[task_name] = build_label_buckets(
            task_name,
            spec.val_dir,
            data[task_name]["val_files"],
            len(spec.labels),
        )
        if os.path.isdir(spec.test_dir) and len(data[task_name]["test_files"]) > 0:
            test_buckets[task_name], test_counts[task_name] = build_label_buckets(
                task_name,
                spec.test_dir,
                data[task_name]["test_files"],
                len(spec.labels),
            )
        else:
            test_buckets[task_name] = {i: [] for i in range(len(spec.labels))}
            test_counts[task_name] = [0] * len(spec.labels)
        print(f"[{task_name}] val_class_counts={val_counts[task_name]} test_class_counts={test_counts[task_name]}")
    val_buckets["SEED"], val_counts["SEED"] = seed_val_buckets, seed_val_counts
    test_buckets["SEED"], test_counts["SEED"] = seed_test_buckets, seed_test_counts
    print(f"[SEED] val_class_counts={val_counts['SEED']} test_class_counts={test_counts['SEED']}")

    dl_train = {
        k: maybe_make_loader(
            ds_train[k],
            args.batch_size,
            args.num_workers,
            sampler_weights[k],
            args.balanced_sampler,
            True,
            True,
        )
        for k in task_names
    }
    eval_bs = int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else max(max(1, int(args.batch_size * 3 // 2)), int(args.batch_size) * 2)

    tvals = []
    for task_name in task_names:
        t1_i, t2_i = estimate_signal_quality_thresholds_multitask(
            dl_train[task_name],
            args.sq_estimate_batches,
            args.sq_q1,
            args.sq_q2,
        )
        tvals.append((t1_i, t2_i))
        print(f"[{task_name}] estimated thresholds: t1={t1_i:.6f}, t2={t2_i:.6f}")
    t1 = float(np.mean([x[0] for x in tvals]))
    t2 = float(np.mean([x[1] for x in tvals]))
    print(f"[shared signal_quality] using averaged thresholds: t1={t1:.6f}, t2={t2:.6f}")

    base_ckpt = torch.load(args.ckpt, map_location="cpu")
    model_args = base_ckpt.get("model_args", {})
    model = NeuroLM(GPTConfig(**model_args), init_from="scratch")
    state_dict = base_ckpt.get("model", base_ckpt.get("state_dict", {}))
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"number of parameters: {model.get_num_params()/1e6:.2f}M")
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)

    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.sft_lr)
    train_iters = {k: iter(dl_train[k]) for k in task_names}
    schedule = build_mix_schedule(args.task_mix)

    os.makedirs(args.save_dir, exist_ok=True)
    best_tuab = -1.0
    best_tuev = -1.0
    best_hmc = -1.0
    best_seed = -1.0
    best_joint = -1.0
    last_path = os.path.join(args.save_dir, "multitask_chain_sft_tuab_tuev_hmc_seed_last.pt")
    best_joint_path = os.path.join(args.save_dir, "multitask_chain_sft_joint_best.pt")

    print(
        f"[SFT] start steps={args.sft_steps} lr={args.sft_lr} task_mix={schedule} "
        f"weights(task/source/state/temporal/spatial/morpho/answer)="
        f"{args.task_weight}/{args.source_weight}/{args.state_weight}/{args.temporal_weight}/"
        f"{args.spatial_weight}/{args.morpho_weight}/{args.answer_weight}"
    )
    print(
        f"[eval-plan] val mode={args.val_eval_mode} per_class={args.val_eval_per_class} | "
        f"test mode={args.test_eval_mode} per_class={args.test_eval_per_class} "
        f"max_samples={args.test_eval_max_samples} min_per_class={args.test_eval_min_per_class} | "
        f"eval_batch_size={eval_bs} progress_every={args.eval_progress_every} | "
        f"joint_weights={joint_weights} balanced_sampler={args.balanced_sampler}"
    )

    for step in range(1, args.sft_steps + 1):
        task_name = schedule[(step - 1) % len(schedule)]
        batch = next_batch(train_iters, dl_train, task_name)

        X_eeg, _text_unused, Y, input_chans, input_time, eeg_mask, _gm = batch
        X_eeg = X_eeg.to(device)
        Y = Y.to(device)
        input_chans = input_chans.to(device)
        input_time = input_time.to(device)
        eeg_mask = eeg_mask.to(device)

        X_eeg_cpu = X_eeg.detach().float().cpu()
        input_time_cpu = input_time.detach().long().cpu()
        eeg_mask_cpu = eeg_mask.detach().bool().cpu()

        X_text_list = []
        Y_text_list = []
        W_list = []
        label_space = label_space_for_task(task_name)
        for i in range(X_eeg.size(0)):
            y_i = int(Y[i].item())
            xw = tokens_to_waveform_multitask(X_eeg_cpu[i], eeg_mask_cpu[i], input_time_cpu[i])
            slots = compute_slots(task_name, xw, y_i, t1=t1, t2=t2)
            answer_text = label_space[y_i]
            x_text_i, y_text_i, base_w_i, _ = build_sft_xy_shared(tok, task_name, answer_text, slots, args.text_max_len)
            task_cfg = task_weight_config(task_name, args)
            w_i = build_weight_vector_shared(tok=tok, x_text_1d=x_text_i, base_w_1d=base_w_i, **task_cfg)
            X_text_list.append(x_text_i)
            Y_text_list.append(y_text_i)
            W_list.append(w_i)

        X_text = torch.stack(X_text_list, dim=0).to(device)
        Y_text = torch.stack(Y_text_list, dim=0).to(device)
        weights = torch.stack(W_list, dim=0).to(device)

        gpt_mask = build_gpt_mask_multitask(X_eeg, X_text, eeg_mask, input_time)
        x_eeg_emb = encode_eeg_only(model, X_eeg, input_chans, input_time, eeg_mask)
        y_eeg_dummy = torch.full((X_eeg.size(0), X_eeg.size(1)), -1, device=device, dtype=torch.long)
        logits_all, _, _ = model.GPT2(
            x_eeg=x_eeg_emb,
            y_eeg=y_eeg_dummy,
            x_text=X_text,
            y_text=Y_text,
            eeg_time_idx=input_time,
            eeg_mask=eeg_mask,
            eeg_text_mask=gpt_mask,
        )
        text_len = X_text.size(1)
        logits_text = logits_all[:, -text_len:, :50257]

        opt.zero_grad(set_to_none=True)
        loss = weighted_nll_loss(logits_text, Y_text, weights, vocab_limit=50257)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            print(f"[SFT] step={step} task={task_name} loss={loss.item():.4f}")

        if (step % args.eval_every == 0) or (step == args.sft_steps):
            save_ckpt(last_path, {"model_args": model_args, "model": model.state_dict()})

            val_loaders = {}
            val_results = {}
            for eval_task in task_names:
                if eval_task == "SEED":
                    val_indices_sel, val_counts_sel = sample_files_for_eval(
                        val_buckets[eval_task],
                        mode=args.val_eval_mode,
                        per_class=args.val_eval_per_class,
                        seed=args.seed + 1000 * step + TASK_NAME_TO_OFFSET[eval_task],
                        max_samples=args.val_eval_max_samples,
                        min_per_class=args.val_eval_min_per_class,
                    )
                    val_loader = build_eval_loader_for_indices(ds_seed_val, val_indices_sel, eval_bs, args.num_workers)
                    selected_n = len(val_indices_sel)
                else:
                    val_loader, val_files_sel, val_counts_sel = build_eval_loader(
                        task_name=eval_task,
                        loader_cls=specs[eval_task].loader_cls,
                        split_dir=specs[eval_task].val_dir,
                        buckets=val_buckets[eval_task],
                        mode=args.val_eval_mode,
                        per_class=args.val_eval_per_class,
                        seed=args.seed + 1000 * step + TASK_NAME_TO_OFFSET[eval_task],
                        eeg_max_len=args.eeg_max_len,
                        text_max_len=args.text_max_len,
                        batch_size=eval_bs,
                        num_workers=args.num_workers,
                        max_samples=args.val_eval_max_samples,
                        min_per_class=args.val_eval_min_per_class,
                    )
                    selected_n = len(val_files_sel)
                val_loaders[eval_task] = val_loader
                print(
                    f"[eval-collection] step={step} task={eval_task} mode={args.val_eval_mode} "
                    f"selected={selected_n} per_class={val_counts_sel} "
                    f"max_samples={args.val_eval_max_samples if args.val_eval_mode == 'proportional' else 0} "
                    f"min_per_class={args.val_eval_min_per_class if args.val_eval_mode == 'proportional' else 0}"
                )

            for eval_task in task_names:
                val_results[eval_task] = eval_multitask_loader(
                    model,
                    tok,
                    eval_task,
                    val_loaders[eval_task],
                    device,
                    args.text_max_len,
                    args.eval_temperature,
                    print_samples=args.print_eval_samples,
                    progress_label=f"val/{eval_task}",
                    progress_every=args.eval_progress_every,
                )
                print_eval(f"{eval_task}(val,shared-chain-v3)", val_results[eval_task])

            jscore = joint_score(
                val_results["TUAB"],
                val_results["TUEV"],
                val_results["HMC"],
                val_results["SEED"],
                joint_weights,
            )
            print(f"[joint(val)] score={jscore:.4f}")

            if float(val_results["TUAB"]["balanced_accuracy"]) > best_tuab:
                best_tuab = float(val_results["TUAB"]["balanced_accuracy"])
                path = os.path.join(args.save_dir, "multitask_chain_sft_tuab_best.pt")
                save_ckpt(path, {"model_args": model_args, "model": model.state_dict()})
                print(f"[save] best TUAB -> {path}")
            if float(val_results["TUEV"]["balanced_accuracy"]) > best_tuev:
                best_tuev = float(val_results["TUEV"]["balanced_accuracy"])
                path = os.path.join(args.save_dir, "multitask_chain_sft_tuev_best.pt")
                save_ckpt(path, {"model_args": model_args, "model": model.state_dict()})
                print(f"[save] best TUEV -> {path}")
            if float(val_results["HMC"]["balanced_accuracy"]) > best_hmc:
                best_hmc = float(val_results["HMC"]["balanced_accuracy"])
                path = os.path.join(args.save_dir, "multitask_chain_sft_hmc_best.pt")
                save_ckpt(path, {"model_args": model_args, "model": model.state_dict()})
                print(f"[save] best HMC -> {path}")
            if float(val_results["SEED"]["balanced_accuracy"]) > best_seed:
                best_seed = float(val_results["SEED"]["balanced_accuracy"])
                path = os.path.join(args.save_dir, "multitask_chain_sft_seed_best.pt")
                save_ckpt(path, {"model_args": model_args, "model": model.state_dict()})
                print(f"[save] best SEED -> {path}")
            if jscore > best_joint:
                best_joint = float(jscore)
                save_ckpt(best_joint_path, {"model_args": model_args, "model": model.state_dict()})
                print(f"[save] best JOINT -> {best_joint_path}")

    if os.path.isfile(best_joint_path):
        load_ckpt_weights(best_joint_path, model, device)
        print(f"[test] loaded best_joint checkpoint: {best_joint_path}")
    else:
        print("[test] best_joint checkpoint not found, using current model weights")

    for task_name in task_names:
        if task_name == "SEED":
            test_indices_sel, test_counts_sel = sample_files_for_eval(
                test_buckets[task_name],
                mode=args.test_eval_mode,
                per_class=args.test_eval_per_class,
                seed=args.seed + 3000 + TASK_NAME_TO_OFFSET[task_name],
                max_samples=args.test_eval_max_samples,
                min_per_class=args.test_eval_min_per_class,
            )
            test_loader = build_eval_loader_for_indices(ds_seed_test, test_indices_sel, eval_bs, args.num_workers)
            selected_n = len(test_indices_sel)
        else:
            if not os.path.isdir(specs[task_name].test_dir) or len(data[task_name]["test_files"]) <= 0:
                print(f"[test] skip {task_name}: no test split found under {specs[task_name].test_dir}")
                continue
            test_loader, test_files_sel, test_counts_sel = build_eval_loader(
                task_name=task_name,
                loader_cls=specs[task_name].loader_cls,
                split_dir=specs[task_name].test_dir,
                buckets=test_buckets[task_name],
                mode=args.test_eval_mode,
                per_class=args.test_eval_per_class,
                seed=args.seed + 3000 + TASK_NAME_TO_OFFSET[task_name],
                eeg_max_len=args.eeg_max_len,
                text_max_len=args.text_max_len,
                batch_size=eval_bs,
                num_workers=args.num_workers,
                max_samples=args.test_eval_max_samples,
                min_per_class=args.test_eval_min_per_class,
            )
            selected_n = len(test_files_sel)
        print(
            f"[test-collection] task={task_name} mode={args.test_eval_mode} "
            f"selected={selected_n} per_class={test_counts_sel} "
            f"max_samples={args.test_eval_max_samples if args.test_eval_mode == 'proportional' else 0} "
            f"min_per_class={args.test_eval_min_per_class if args.test_eval_mode == 'proportional' else 0}"
        )
        res = eval_multitask_loader(
            model,
            tok,
            task_name,
            test_loader,
            device,
            args.text_max_len,
            args.eval_temperature,
            print_samples=0,
            progress_label=f"test/{task_name}",
            progress_every=args.eval_progress_every,
        )
        print_eval(f"{task_name}(test,shared-chain-v3)", res)


if __name__ == "__main__":
    main()
