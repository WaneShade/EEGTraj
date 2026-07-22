# -*- coding: utf-8 -*-
"""Shared NeuroLM and EEG utility functions for the main training pipeline."""

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


LABELS = ["A", "B", "C", "D", "E", "F"]

TUEV_NUM_CHANS = 23
EEG_TOKEN_LEN = 200
FS = 200

FRONTAL_IDX = np.array([0, 1, 2, 3, 10, 11, 18, 21, 22], dtype=np.int64)
POSTERIOR_IDX = np.array([6, 7, 8, 9, 14, 15, 20], dtype=np.int64)
LEFT_IDX = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 21], dtype=np.int64)
RIGHT_IDX = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 22], dtype=np.int64)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_pkls(dir_path: str) -> List[str]:
    files = [fn for fn in os.listdir(str(dir_path)) if fn.endswith(".pkl")]
    files.sort()
    return files


@dataclass
class TokMap:
    enc: Any
    eos_id: int
    sep_id: int
    allowed_letter_ids: List[int]
    token_id_rparen: int
    token_id_newline: int


def build_tokmap() -> TokMap:
    enc = tiktoken.get_encoding("gpt2")
    eos_id = 50256
    sep_id = 50257
    allowed = []
    for ch in LABELS:
        ids = enc.encode(ch, allowed_special={"<|endoftext|>"})
        if len(ids) != 1:
            raise RuntimeError(f"Letter '{ch}' is not single-token: {ids}")
        allowed.append(ids[0])
    return TokMap(
        enc=enc,
        eos_id=eos_id,
        sep_id=sep_id,
        allowed_letter_ids=allowed,
        token_id_rparen=enc.encode(")")[0],
        token_id_newline=enc.encode("\n")[0],
    )


def make_loader(ds, batch_size: int, num_workers: int, sampler=None, shuffle: bool = False, drop_last: bool = True):
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def artifact_score_diff_energy(x: Optional[torch.Tensor]) -> float:
    if x is None:
        return np.nan
    x = x.float()
    if x.numel() < 2:
        return np.nan
    x = x - x.mean(dim=1, keepdim=True)
    diff = x[:, 1:] - x[:, :-1]
    num = float((diff ** 2).mean(dim=1).median().item())
    den = float((x ** 2).mean(dim=1).median().item()) + 1e-12
    return num / den


def _fft_psd(x_1d: torch.Tensor, fs: int = FS) -> Tuple[np.ndarray, np.ndarray]:
    x_1d = x_1d - x_1d.mean()
    length = int(x_1d.numel())
    if length < fs:
        return np.array([]), np.array([])
    window = torch.hann_window(length, periodic=False, device=x_1d.device, dtype=x_1d.dtype)
    spectrum = torch.fft.rfft(x_1d * window)
    psd = (spectrum.abs() ** 2).cpu().numpy()
    freqs = np.fft.rfftfreq(length, d=1.0 / fs)
    return freqs, psd


def _dominant_bands(x: Optional[torch.Tensor]) -> Dict[str, float]:
    if x is None:
        return {"delta": np.nan, "theta": np.nan, "alpha": np.nan, "beta": np.nan, "hf": np.nan, "peakiness": np.nan}
    _, length = x.shape
    if length < FS:
        return {"delta": np.nan, "theta": np.nan, "alpha": np.nan, "beta": np.nan, "hf": np.nan, "peakiness": np.nan}

    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "hf": (30.0, 45.0),
    }
    fracs = {k: [] for k in bands}
    peakiness_list = []
    for c in range(x.shape[0]):
        freqs, psd = _fft_psd(x[c], fs=FS)
        if freqs.size == 0:
            continue
        mask = (freqs >= 0.5) & (freqs <= 45.0)
        f = freqs[mask]
        p = psd[mask]
        if p.size < 8:
            continue
        total = float(p.sum()) + 1e-12
        for name, (lo, hi) in bands.items():
            band_mask = (f >= lo) & (f < hi)
            fracs[name].append(float(p[band_mask].sum()) / total)
        peakiness_list.append(float(p.max() / (np.median(p) + 1e-12)))

    return {
        "delta": float(np.median(fracs["delta"])) if fracs["delta"] else np.nan,
        "theta": float(np.median(fracs["theta"])) if fracs["theta"] else np.nan,
        "alpha": float(np.median(fracs["alpha"])) if fracs["alpha"] else np.nan,
        "beta": float(np.median(fracs["beta"])) if fracs["beta"] else np.nan,
        "hf": float(np.median(fracs["hf"])) if fracs["hf"] else np.nan,
        "peakiness": float(np.median(peakiness_list)) if peakiness_list else np.nan,
    }


def _line_lengthiness(x: Optional[torch.Tensor]) -> float:
    if x is None:
        return np.nan
    x = x.float()
    if x.size(1) < 4:
        return np.nan
    num = (x[:, 1:] - x[:, :-1]).abs().median(dim=1).values.median().item()
    den = x.abs().median(dim=1).values.median().item() + 1e-12
    return float(num / den)


def _robust_peak_stats(x: Optional[torch.Tensor]) -> Dict[str, float]:
    if x is None:
        return {"event_rate": np.nan, "periodicity": np.nan, "transient_strength": np.nan}
    x = x.float()
    _, length = x.shape
    if length < FS:
        return {"event_rate": np.nan, "periodicity": np.nan, "transient_strength": np.nan}

    z = x.abs().median(dim=0).values
    med = z.median().item()
    mad = (z - med).abs().median().item() + 1e-8
    thr = med + 3.0 * mad

    z_np = z.cpu().numpy()
    peak_idx = []
    refractory = int(0.20 * FS)
    i = 1
    while i < length - 1:
        if z_np[i] > thr and z_np[i] >= z_np[i - 1] and z_np[i] >= z_np[i + 1]:
            peak_idx.append(i)
            i += refractory
        else:
            i += 1

    duration_s = length / FS
    event_rate = float(len(peak_idx) / max(duration_s, 1e-6))
    if len(peak_idx) >= 3:
        intervals = np.diff(np.array(peak_idx, dtype=np.float64)) / FS
        cv = float(np.std(intervals) / (np.mean(intervals) + 1e-12))
        periodicity = float(np.exp(-cv))
    else:
        periodicity = 0.0

    transient_strength = float((z.max().item() - med) / (mad + 1e-8))
    transient_strength = float(np.tanh(transient_strength / 10.0))
    return {
        "event_rate": event_rate,
        "periodicity": periodicity,
        "transient_strength": transient_strength,
    }


def _spatial_stats(x: Optional[torch.Tensor]) -> Dict[str, float]:
    if x is None:
        return {"entropy_norm": np.nan, "active_frac": np.nan, "frontal_ratio": np.nan, "lateralization": np.nan}
    x = x.float()
    if x.shape[0] != TUEV_NUM_CHANS:
        return {"entropy_norm": np.nan, "active_frac": np.nan, "frontal_ratio": np.nan, "lateralization": np.nan}

    energy = (x ** 2).mean(dim=1).cpu().numpy() + 1e-12
    prob = energy / (energy.sum() + 1e-12)
    entropy = -np.sum(prob * np.log(prob + 1e-12))
    entropy_norm = float(entropy / np.log(len(prob)))
    med_e = float(np.median(energy))
    active_frac = float(np.mean(energy > med_e * 1.2))

    frontal = float(np.mean(energy[FRONTAL_IDX]))
    posterior = float(np.mean(energy[POSTERIOR_IDX])) + 1e-12
    frontal_ratio = frontal / posterior
    left_e = float(np.mean(energy[LEFT_IDX]))
    right_e = float(np.mean(energy[RIGHT_IDX]))
    lateralization = abs(left_e - right_e) / (left_e + right_e + 1e-12)

    return {
        "entropy_norm": entropy_norm,
        "active_frac": active_frac,
        "frontal_ratio": frontal_ratio,
        "lateralization": lateralization,
    }


def compute_decision_features(x: Optional[torch.Tensor], t1: float, t2: float) -> Dict[str, float]:
    if x is None:
        return {
            "artifact_score": np.nan,
            "flatline_ratio": np.nan,
            "clip_ratio": np.nan,
            "line_length": np.nan,
            "delta": np.nan,
            "theta": np.nan,
            "alpha": np.nan,
            "beta": np.nan,
            "hf": np.nan,
            "peakiness": np.nan,
            "event_rate": np.nan,
            "periodicity": np.nan,
            "transient_strength": np.nan,
            "entropy_norm": np.nan,
            "active_frac": np.nan,
            "frontal_ratio": np.nan,
            "lateralization": np.nan,
            "t1": t1,
            "t2": t2,
        }

    x = x.float()
    dx = (x[:, 1:] - x[:, :-1]).abs()
    q99 = float(torch.quantile(x.abs().reshape(-1), 0.99).item()) + 1e-12
    return {
        "artifact_score": artifact_score_diff_energy(x),
        "flatline_ratio": float((dx < 1e-6).float().mean().item()),
        "clip_ratio": float((x.abs() > q99 * 0.98).float().mean().item()),
        "line_length": _line_lengthiness(x),
        **_dominant_bands(x),
        **_robust_peak_stats(x),
        **_spatial_stats(x),
        "t1": t1,
        "t2": t2,
    }


def source_label(feats: Dict[str, float]) -> str:
    art = feats["artifact_score"]
    hf = feats["hf"]
    clip_ratio = feats["clip_ratio"]
    flatline_ratio = feats["flatline_ratio"]
    delta = feats["delta"]
    frontal_ratio = feats["frontal_ratio"]
    ll = feats["line_length"]
    periodicity = feats["periodicity"]
    lateralization = feats["lateralization"]
    event_rate = feats["event_rate"]
    alpha = feats["alpha"]
    t1 = feats["t1"]
    t2 = feats["t2"]

    if flatline_ratio > 0.30 or clip_ratio > 0.03 or art > max(t2 * 1.20, t2 + 0.02) or hf > 0.18:
        return "noncerebral"

    if delta > 0.45 and frontal_ratio > 1.45 and ll < 1.18 and periodicity < 0.45 and lateralization < 0.35:
        return "noncerebral"

    if periodicity > 0.58 or ll > 1.26 or event_rate > 0.55:
        return "cerebral_event"

    if alpha > 0.20 and art < max(t1, 0.08) and ll < 1.15 and hf < 0.10:
        return "background_like"

    if art > t1 or hf > 0.10:
        return "noncerebral"

    return "uncertain"


def temporal_label(feats: Dict[str, float], source: str) -> str:
    periodicity = feats["periodicity"]
    event_rate = feats["event_rate"]
    ll = feats["line_length"]
    delta = feats["delta"]
    hf = feats["hf"]
    art = feats["artifact_score"]
    frontal_ratio = feats["frontal_ratio"]
    t2 = feats["t2"]

    if source == "background_like":
        return "none_or_uncertain"
    if source == "noncerebral":
        if delta > 0.45 and frontal_ratio > 1.45 and periodicity < 0.45 and ll < 1.18:
            return "slow_drift"
        if hf > 0.12 or art > t2 or feats["clip_ratio"] > 0.02:
            return "broadband_irregular"
        return "none_or_uncertain"
    if source == "cerebral_event":
        if periodicity > 0.58 and event_rate > 0.45:
            return "periodic_repeating"
        if ll > 1.24 or event_rate > 0.30:
            return "isolated_transient"
        return "none_or_uncertain"
    return "none_or_uncertain"


def spatial_label(feats: Dict[str, float], source: str, temporal: str) -> str:
    entropy_norm = feats["entropy_norm"]
    active_frac = feats["active_frac"]
    frontal_ratio = feats["frontal_ratio"]
    lateralization = feats["lateralization"]

    if source == "background_like":
        return "na"
    if source == "noncerebral":
        if temporal == "slow_drift" and frontal_ratio > 1.45:
            return "frontal_dominant"
        if lateralization > 0.30 and active_frac > 0.22:
            return "focal_local"
        return "diffuse_mixed"
    if source == "cerebral_event":
        if entropy_norm > 0.82 and lateralization < 0.16 and active_frac > 0.55:
            return "generalized"
        if lateralization > 0.30 and active_frac > 0.22:
            return "lateralized"
        if entropy_norm < 0.72 or active_frac < 0.40:
            return "focal_local"
        return "diffuse_mixed"
    return "na"


def morpho_label(feats: Dict[str, float], source: str, temporal: str, spatial: str) -> str:
    ll = feats["line_length"]
    trans = feats["transient_strength"]
    alpha = feats["alpha"]
    art = feats["artifact_score"]
    hf = feats["hf"]
    t1 = feats["t1"]
    t2 = feats["t2"]

    if source == "background_like":
        if alpha > 0.22 and art < max(t1, 0.08) and ll < 1.12:
            return "background_rhythm"
        return "uncertain"
    if source == "noncerebral":
        if temporal == "slow_drift":
            return "drift_like"
        if temporal == "broadband_irregular" or hf > 0.12 or art > t2:
            return "noise_like"
        return "uncertain"
    if source == "cerebral_event":
        if ll > 1.20 and trans > 0.30:
            return "spike_sharp_complex"
        return "uncertain"
    return "na"


def weighted_nll_loss(logits_text: torch.Tensor, y_text: torch.Tensor, weights: torch.Tensor, vocab_limit: int = 50257) -> torch.Tensor:
    batch, length, vocab = logits_text.shape
    vocab = min(vocab, vocab_limit)
    logits_text = logits_text[:, :, :vocab].contiguous()
    logits_f = logits_text.view(batch * length, vocab)
    y_f = y_text.view(batch * length)
    w_f = weights.view(batch * length)
    mask = (y_f != -1) & (w_f > 0)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits_text.device)
    loss = F.cross_entropy(logits_f[mask], y_f[mask], reduction="none")
    return (loss * w_f[mask]).sum() / (w_f[mask].sum() + 1e-12)


@torch.no_grad()
def encode_eeg_only(model: Any, X_eeg, input_chans, input_time, input_mask):
    im = input_mask.unsqueeze(1).repeat(1, X_eeg.size(1), 1).unsqueeze(1)
    x = model.tokenizer(X_eeg, input_chans, input_time, im, return_all_tokens=True)
    x = model.encode_transform_layer(x)
    x = x + model.pos_embed(input_chans)
    return x


@dataclass
class TrieNode:
    children: Dict[int, "TrieNode"]
    is_end: bool = False

    def __init__(self):
        self.children = {}
        self.is_end = False


def build_trie(seqs: List[List[int]]) -> TrieNode:
    root = TrieNode()
    for seq in seqs:
        cur = root
        for tid in seq:
            if tid not in cur.children:
                cur.children[tid] = TrieNode()
            cur = cur.children[tid]
        cur.is_end = True
    return root


def trie_next_allowed(node: TrieNode) -> List[int]:
    return list(node.children.keys())


def trie_step(node: TrieNode, tid: int) -> TrieNode:
    return node.children[tid]


def sample_or_greedy_allowed(logits_next: torch.Tensor, allowed_ids: List[int], temperature: float, greedy: bool) -> int:
    idx = torch.tensor(allowed_ids, device=logits_next.device, dtype=torch.long)
    logits = logits_next.index_select(0, idx) / max(float(temperature), 1e-6)
    if greedy:
        return int(idx[int(torch.argmax(logits).item())].item())
    probs = F.softmax(logits, dim=-1)
    sampled = int(torch.multinomial(probs, num_samples=1).item())
    return int(idx[sampled].item())

