# -*- coding: utf-8 -*-
"""
Main structured GRPO for TUAB + TUEV + HMC + SEED.

Design notes
------------
- Always initialize from an SFT checkpoint passed by --sft_ckpt.
- This is the paper mainline method:
  Structured SFT -> answer reward -> validity reward -> consistency reward ->
  discriminative reward.
- The answer reward is not gated by chain quality, keeping answer learning as
  the dominant objective while chain rewards shape the intermediate DSL.
- Validity rewards DSL legality and non-empty chain-implied label candidates.
- Consistency rewards agreement between chain-implied candidates and ANS.
- Discriminative reward combines true-label margin and counterfactual
  sensitivity of high-value chain factors.
- Test with the task-specific best checkpoint for each dataset.

Example
-------
CUDA_VISIBLE_DEVICES=0 python multitask_chain_project/scripts/train_grpo.py \
  --sft_ckpt runs_multitask_sft_v3_rerun/multitask_chain_sft_joint_best.pt \
  --tuab_root data/TUAB/processed \
  --tuev_root data/TUEV/processed \
  --hmc_root data/HMC \
  --seed_root data/SEED \
  --device cuda:0 \
  --balanced_sampler \
  --task_mix 1,1,1,1 \
  --batch_size 2 \
  --group_size 4 \
  --grpo_steps 1000 \
  --grpo_lr 2e-6 \
  --slot_temperature 0.9 \
  --ans_temperature 0.8 \
  --eval_temperature 0.7 \
  --kl_coef 0.12 \
  --entropy_coef 0.003 \
  --format_coef 0.03 \
  --bc_coef 0.01 \
  --reward_scheme inv_sqrt_freq \
  --answer_reward_weight 1.00 \
  --validity_reward_weight 0.06 \
  --invalidity_penalty 0.06 \
  --consistency_reward 0.06 \
  --inconsistency_penalty 0.06 \
  --singleton_bonus 0.01 \
  --chain_margin_weight 0.08 \
  --counterfactual_weight 0.08 \
  --wrong_penalty 0.55 \
  --disable_seed_chain_margin \
  --eval_every 250 \
  --val_eval_mode balanced \
  --val_eval_per_class 500 \
  --test_eval_mode proportional \
  --test_checkpoint_mode task_best \
  --save_dir runs_main_structured_grpo \
  --log_dir logs \
  --log_prefix main_structured_grpo
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
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

from model.model import GPTConfig
from model.model_neurolm import NeuroLM

try:
    from run_logger import setup_run_logging
except Exception:
    setup_run_logging = None

from utils import build_tokmap, encode_eeg_only, list_pkls, set_seed
from engine_sft import (
    HMC_LABELS,
    MORPHO_VALUES,
    SEED_LABELS,
    SOURCE_VALUES,
    SPATIAL_VALUES,
    STATE_VALUES,
    TASK_NAME_TO_OFFSET,
    TEMPORAL_VALUES,
    TUEV_LABELS,
    TUAB_LABELS,
    build_eval_loader,
    build_eval_loader_for_indices,
    build_gpt_mask_multitask,
    build_label_buckets,
    build_seed_dataset,
    build_seed_index_buckets,
    build_task_spec,
    build_mix_schedule,
    candidate_labels_from_factors,
    canonical_factors_for_label,
    label_space_for_task,
    load_ckpt_weights,
    maybe_make_loader,
    parse_joint_weights,
    joint_score,
    print_eval,
    prompt_str_shared,
    safe_decode,
    sample_files_for_eval,
    save_ckpt,
    scan_labels_for_sampler,
    seed_sampler_weights_from_buckets,
    source_option_values,
    state_option_values,
    temporal_option_values,
    spatial_option_values,
    morpho_option_values,
)


TASK_FACTOR_WEIGHTS = {
    "TUEV": {"source": 2.0, "state": 0.0, "temporal": 1.5, "spatial": 1.0, "morpho": 0.8},
    "TUAB": {"source": 1.6, "state": 1.2, "temporal": 1.2, "spatial": 0.5, "morpho": 1.0},
    "HMC": {"source": 0.8, "state": 1.8, "temporal": 1.2, "spatial": 0.4, "morpho": 1.4},
    "SEED": {"source": 0.9, "state": 0.8, "temporal": 1.0, "spatial": 0.9, "morpho": 0.9},
}


def make_class_weights(counts: List[int], scheme: str) -> List[float]:
    c = np.array(counts, dtype=np.float64)
    p = c / max(float(c.sum()), 1.0)
    if scheme == "uniform":
        w = np.ones_like(p)
    elif scheme == "inv_freq":
        w = 1.0 / np.maximum(p, 1e-12)
    elif scheme == "inv_sqrt_freq":
        w = 1.0 / np.sqrt(np.maximum(p, 1e-12))
    else:
        w = np.ones_like(p)
    w = w / max(float(w.mean()), 1e-12)
    return w.tolist()


def next_batch(iters: Dict[str, object], loaders: Dict[str, object], task_name: str):
    try:
        return next(iters[task_name])
    except StopIteration:
        iters[task_name] = iter(loaders[task_name])
        return next(iters[task_name])


def build_eval_indices_for_seed(
    buckets: Dict[int, List[int]],
    mode: str,
    per_class: int,
    seed: int,
    max_samples: int,
    min_per_class: int,
) -> Tuple[List[int], List[int]]:
    indices, counts = sample_files_for_eval(
        buckets,
        mode=mode,
        per_class=per_class,
        seed=seed,
        max_samples=max_samples,
        min_per_class=min_per_class,
    )
    return list(indices), counts


def build_value_token_map(tok, values: List[str]) -> Dict[str, List[int]]:
    return {v: tok.enc.encode(v, allowed_special={"<|endoftext|>"}) for v in values}


def append_tokens(x_text: torch.Tensor, tokens: List[int], max_len: int = 0) -> torch.Tensor:
    if not tokens:
        return x_text
    t = torch.tensor(tokens, device=x_text.device, dtype=torch.long).unsqueeze(0)
    x_text = torch.cat([x_text, t], dim=1)
    if max_len and max_len > 0 and x_text.size(1) > max_len:
        x_text = x_text[:, -max_len:]
    return x_text


def sample_one_step(logits_next: torch.Tensor, allowed_ids: List[int], temperature: float) -> Tuple[int, torch.Tensor, torch.Tensor]:
    idx = torch.tensor(allowed_ids, device=logits_next.device, dtype=torch.long)
    l = logits_next.index_select(0, idx) / max(float(temperature), 1e-6)
    probs = F.softmax(l, dim=-1)
    sampled_k = int(torch.multinomial(probs, num_samples=1).item())
    sampled_id = int(idx[sampled_k].item())
    logp = torch.log(probs[sampled_k] + 1e-12)
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    return sampled_id, logp, entropy


def greedy_one_step(logits_next: torch.Tensor, allowed_ids: List[int], temperature: float) -> int:
    idx = torch.tensor(allowed_ids, device=logits_next.device, dtype=torch.long)
    l = logits_next.index_select(0, idx) / max(float(temperature), 1e-6)
    j = int(torch.argmax(l).item())
    return int(idx[j].item())


def allowed_source_values(task_name: str) -> List[str]:
    return source_option_values(task_name)


def allowed_state_values(task_name: str, source: str) -> List[str]:
    if task_name == "TUEV":
        return ["na"]
    if task_name in {"HMC", "TUAB"}:
        if source == "background_like":
            return ["wake_like", "transition_like", "stable_sleep_like", "rem_like", "uncertain"]
        if source in {"cerebral_event", "noncerebral"}:
            return ["na", "uncertain"]
        return list(STATE_VALUES)
    return state_option_values(task_name)


def allowed_temporal_values(task_name: str, source: str, state: str) -> List[str]:
    if task_name == "TUEV":
        if source == "background_like":
            return ["stable_rhythm", "slow_drift", "none_or_uncertain"]
        if source in {"noncerebral", "cerebral_event"}:
            return list(TEMPORAL_VALUES)
        return temporal_option_values(task_name)

    if task_name in {"TUAB", "HMC"}:
        if source == "background_like":
            return ["stable_rhythm", "slow_drift", "state_transition", "none_or_uncertain"]
        if source == "noncerebral":
            return ["slow_drift", "broadband_irregular", "none_or_uncertain"]
        if source == "cerebral_event":
            return ["isolated_transient", "periodic_repeating", "slow_drift", "none_or_uncertain"]
        return temporal_option_values(task_name)

    return temporal_option_values(task_name)


def allowed_spatial_values(task_name: str, source: str, state: str, temporal: str) -> List[str]:
    if task_name == "TUEV":
        if source == "background_like":
            return ["na", "diffuse_mixed", "frontal_dominant"]
        if source == "noncerebral":
            return ["diffuse_mixed", "frontal_dominant", "na"]
        if source == "cerebral_event":
            return ["focal_local", "lateralized", "generalized", "frontal_dominant", "diffuse_mixed"]
        return spatial_option_values(task_name)

    if task_name in {"TUAB", "HMC"}:
        if source == "background_like":
            return ["na", "diffuse_mixed", "frontal_dominant", "generalized"]
        if source == "noncerebral":
            return ["frontal_dominant", "diffuse_mixed", "focal_local", "na"]
        if source == "cerebral_event":
            return ["focal_local", "lateralized", "generalized", "frontal_dominant", "diffuse_mixed"]
        return spatial_option_values(task_name)

    return spatial_option_values(task_name)


def allowed_morpho_values(task_name: str, source: str, state: str, temporal: str, spatial: str) -> List[str]:
    if task_name == "TUEV":
        if source == "background_like":
            return ["background_rhythm", "slow_wave_like", "uncertain", "na"]
        if source == "noncerebral":
            return ["drift_like", "noise_like", "uncertain", "na"]
        if source == "cerebral_event":
            return ["slow_wave_like", "spike_sharp_complex", "uncertain", "na"]
        return morpho_option_values(task_name)

    if task_name in {"TUAB", "HMC"}:
        if source == "background_like":
            return ["background_rhythm", "mixed_low_voltage", "slow_wave_like", "spindle_kcomplex_like", "uncertain"]
        if source == "noncerebral":
            return ["drift_like", "noise_like", "uncertain", "na"]
        if source == "cerebral_event":
            return ["spike_sharp_complex", "slow_wave_like", "uncertain", "na"]
        return morpho_option_values(task_name)

    return morpho_option_values(task_name)


def allowed_answer_values(task_name: str, pred_slots: Dict[str, str]) -> List[str]:
    return label_space_for_task(task_name)


def chain_validity_reward_multitask(
    task_name: str,
    pred_slots: Dict[str, str],
    reward_weight: float,
    invalidity_penalty: float,
) -> Tuple[float, bool, List[str]]:
    """Reward legal structured chains that imply at least one valid label."""
    required = ["source", "state", "temporal", "spatial", "morpho"]
    if any(k not in pred_slots for k in required):
        return -float(invalidity_penalty), False, []

    source = str(pred_slots.get("source", "uncertain"))
    state = str(pred_slots.get("state", "uncertain"))
    temporal = str(pred_slots.get("temporal", "none_or_uncertain"))
    spatial = str(pred_slots.get("spatial", "na"))
    morpho = str(pred_slots.get("morpho", "na"))

    legal = (
        source in allowed_source_values(task_name)
        and state in allowed_state_values(task_name, source)
        and temporal in allowed_temporal_values(task_name, source, state)
        and spatial in allowed_spatial_values(task_name, source, state, temporal)
        and morpho in allowed_morpho_values(task_name, source, state, temporal, spatial)
    )
    cand = list(dict.fromkeys(candidate_labels_from_factors(task_name, pred_slots))) if legal else []
    labels = set(label_space_for_task(task_name))
    valid = bool(legal and len(cand) > 0 and all(c in labels for c in cand))
    if not valid:
        return -float(invalidity_penalty), False, cand

    n_labels = len(labels)
    specificity = 1.0 - float(len(cand) - 1) / float(max(n_labels - 1, 1))
    score = 0.50 + 0.50 * max(0.0, min(1.0, specificity))
    return float(reward_weight) * score, True, cand


MAIN_METHOD_NAME = "answer_validity_consistency_discriminative"


def chain_answer_consistency_reward_multitask(
    task_name: str,
    pred_slots: Dict[str, str],
    y_pred: int,
    consistency_reward: float,
    inconsistency_penalty: float,
    singleton_bonus: float,
) -> float:
    labels = label_space_for_task(task_name)
    pred_label = labels[y_pred]
    cand = candidate_labels_from_factors(task_name, pred_slots)
    if pred_label not in cand:
        return -float(inconsistency_penalty)

    reward = float(consistency_reward)
    if len(cand) <= 1:
        reward += float(singleton_bonus)
    elif len(cand) == 2:
        reward += 0.5 * float(singleton_bonus)
    return reward


def chain_label_support_multitask(task_name: str, pred_slots: Dict[str, str], y: int) -> float:
    """Compatibility between a sampled chain and one label's canonical factors."""
    target = canonical_factors_for_label(task_name, int(y))
    weights = TASK_FACTOR_WEIGHTS[task_name]
    score = 0.0
    denom = 0.0
    for key in ["source", "state", "temporal", "spatial", "morpho"]:
        w = float(weights.get(key, 0.0))
        if w <= 0:
            continue
        denom += w
        pv = str(pred_slots.get(key, "uncertain"))
        tv = str(target.get(key, "uncertain"))
        if pv == tv:
            score += w
        elif pv in {"uncertain", "none_or_uncertain", "na"}:
            score += 0.10 * w
        else:
            score -= 0.15 * w
    return float(max(0.0, min(1.0, score / max(denom, 1e-6))))


def chain_margin_reward_multitask(task_name: str, pred_slots: Dict[str, str], y_true: int) -> Tuple[float, float, int]:
    """Reward chains that support the true label over the hardest wrong label."""
    labels = label_space_for_task(task_name)
    true_score = chain_label_support_multitask(task_name, pred_slots, y_true)
    best_wrong = -1.0
    best_wrong_idx = -1
    for yi in range(len(labels)):
        if yi == int(y_true):
            continue
        s = chain_label_support_multitask(task_name, pred_slots, yi)
        if s > best_wrong:
            best_wrong = s
            best_wrong_idx = yi
    margin = true_score - max(best_wrong, 0.0)
    return float(margin), float(true_score), int(best_wrong_idx)


def task_chain_margin_weight(task_name: str, base_weight: float, disable_seed_chain_margin: bool) -> float:
    if task_name == "SEED" and disable_seed_chain_margin:
        return 0.0
    return float(base_weight)


def counterfactual_sensitivity_reward_multitask(task_name: str, pred_slots: Dict[str, str], y_true: int) -> float:
    """Approximate whether important chain factors are causally useful.

    We cannot afford a full model counterfactual pass for every rollout here, so
    this uses the chain-label compatibility function as the mediator score:
    matched high-value factors should be necessary for preserving true-label
    support. Replacing them with uncertainty should reduce support.
    """
    base = chain_label_support_multitask(task_name, pred_slots, y_true)
    target = canonical_factors_for_label(task_name, int(y_true))
    drops: List[float] = []
    for key in ["source", "state", "temporal", "spatial", "morpho"]:
        if str(pred_slots.get(key, "uncertain")) != str(target.get(key, "uncertain")):
            continue
        cf_slots = dict(pred_slots)
        cf_slots[key] = "none_or_uncertain" if key == "temporal" else "uncertain"
        cf = chain_label_support_multitask(task_name, cf_slots, y_true)
        drops.append(max(0.0, base - cf))
    if not drops:
        return 0.0
    return float(max(0.0, min(1.0, sum(drops) / max(len(drops), 1))))


def build_prompt_ids(tok, task_name: str, text_max_len: int) -> torch.Tensor:
    prompt_ids = [tok.sep_id] + tok.enc.encode(prompt_str_shared(task_name), allowed_special={"<|endoftext|>"})
    if text_max_len and text_max_len > 0:
        prompt_ids = prompt_ids[-text_max_len:]
    return torch.tensor(prompt_ids, dtype=torch.long)


def sample_value_from_candidates(
    model: NeuroLM,
    X_eeg_tokens_1: torch.Tensor,
    x_eeg_emb_1: torch.Tensor,
    input_time_1: torch.Tensor,
    eeg_mask_1: torch.Tensor,
    x_text: torch.Tensor,
    candidate_values: List[str],
    value_token_map: Dict[str, List[int]],
    temperature: float,
    kind: str,
) -> Tuple[str, List[int], torch.Tensor, List[dict]]:
    active = [(v, value_token_map[v]) for v in candidate_values if v in value_token_map]
    emitted: List[int] = []
    action_recs: List[dict] = []

    while True:
        pos_in_value = len(emitted)
        next_allowed = sorted(list({seq[pos_in_value] for _, seq in active if len(seq) > pos_in_value}))
        gpt_mask = build_gpt_mask_multitask(X_eeg_tokens_1, x_text, eeg_mask_1, input_time_1)
        logits_pol, _, _ = model.GPT2(
            x_eeg=x_eeg_emb_1,
            x_text=x_text,
            y_text=None,
            eeg_time_idx=input_time_1,
            eeg_mask=eeg_mask_1,
            eeg_text_mask=gpt_mask,
        )
        lp_next = logits_pol[0, -1, :50257]
        tid, _, _ = sample_one_step(lp_next, next_allowed, temperature)

        pos = int(x_text.size(1))
        x_text = append_tokens(x_text, [tid], 0)
        action_recs.append({"pos": pos, "allowed": next_allowed, "temp": float(temperature), "kind": kind})
        emitted.append(tid)

        active = [(v, seq) for (v, seq) in active if len(seq) > pos_in_value and seq[pos_in_value] == tid]
        matched = [v for (v, seq) in active if len(seq) == len(emitted)]
        if len(matched) == 1:
            return matched[0], emitted, x_text, action_recs


def greedy_value_from_candidates(
    model: NeuroLM,
    X_eeg_tokens_1: torch.Tensor,
    x_eeg_emb_1: torch.Tensor,
    input_time_1: torch.Tensor,
    eeg_mask_1: torch.Tensor,
    x_text: torch.Tensor,
    candidate_values: List[str],
    value_token_map: Dict[str, List[int]],
    temperature: float,
) -> Tuple[str, List[int], torch.Tensor]:
    active = [(v, value_token_map[v]) for v in candidate_values if v in value_token_map]
    emitted: List[int] = []

    while True:
        pos_in_value = len(emitted)
        next_allowed = sorted(list({seq[pos_in_value] for _, seq in active if len(seq) > pos_in_value}))
        gpt_mask = build_gpt_mask_multitask(X_eeg_tokens_1, x_text, eeg_mask_1, input_time_1)
        logits_pol, _, _ = model.GPT2(
            x_eeg=x_eeg_emb_1,
            x_text=x_text,
            y_text=None,
            eeg_time_idx=input_time_1,
            eeg_mask=eeg_mask_1,
            eeg_text_mask=gpt_mask,
        )
        lp_next = logits_pol[0, -1, :50257]
        tid = greedy_one_step(lp_next, next_allowed, temperature)

        x_text = append_tokens(x_text, [tid], 0)
        emitted.append(tid)

        active = [(v, seq) for (v, seq) in active if len(seq) > pos_in_value and seq[pos_in_value] == tid]
        matched = [v for (v, seq) in active if len(seq) == len(emitted)]
        if len(matched) == 1:
            return matched[0], emitted, x_text


@torch.no_grad()
def score_value_options_normalized(
    model: NeuroLM,
    X_eeg_tokens_1: torch.Tensor,
    x_eeg_emb_1: torch.Tensor,
    input_time_1: torch.Tensor,
    eeg_mask_1: torch.Tensor,
    x_text: torch.Tensor,
    candidate_values: List[str],
    value_token_map: Dict[str, List[int]],
    temperature: float,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for value in candidate_values:
        ids = value_token_map.get(value, [])
        if not ids:
            scores[value] = float("-inf")
            continue
        ctx = x_text.clone()
        total = 0.0
        for tid in ids:
            gpt_mask = build_gpt_mask_multitask(X_eeg_tokens_1, ctx, eeg_mask_1, input_time_1)
            logits_pol, _, _ = model.GPT2(
                x_eeg=x_eeg_emb_1,
                x_text=ctx,
                y_text=None,
                eeg_time_idx=input_time_1,
                eeg_mask=eeg_mask_1,
                eeg_text_mask=gpt_mask,
            )
            logits_next = logits_pol[0, -1, :50257] / max(float(temperature), 1e-6)
            total += float(F.log_softmax(logits_next, dim=-1)[int(tid)].item())
            ctx = append_tokens(ctx, [int(tid)], 0)
        scores[value] = float(total / max(len(ids), 1))
    return scores


def tuab_abnormal_score_from_value_scores(scores: Dict[str, float]) -> Optional[float]:
    vals = [scores.get("normal", float("-inf")), scores.get("abnormal", float("-inf"))]
    if not np.all(np.isfinite(np.array(vals, dtype=np.float64))):
        return None
    return float(torch.softmax(torch.tensor(vals, dtype=torch.float32), dim=0)[1].item())


@torch.no_grad()
def sample_structured_rollout_no_grad_multitask_v2(
    model: NeuroLM,
    tok,
    source_token_map: Dict[str, List[int]],
    state_token_map: Dict[str, List[int]],
    temporal_token_map: Dict[str, List[int]],
    spatial_token_map: Dict[str, List[int]],
    morpho_token_map: Dict[str, List[int]],
    answer_token_maps: Dict[str, Dict[str, List[int]]],
    task_name: str,
    X_eeg_tokens_1: torch.Tensor,
    x_eeg_emb_1: torch.Tensor,
    input_time_1: torch.Tensor,
    eeg_mask_1: torch.Tensor,
    prompt_ids_1d: torch.Tensor,
    slot_temperature: float,
    ans_temperature: float,
) -> Tuple[torch.Tensor, List[dict], List[int], int, Dict[str, str], str]:
    device = X_eeg_tokens_1.device
    x_text = prompt_ids_1d.unsqueeze(0).to(device)
    action_recs: List[dict] = []
    format_pos: List[int] = []

    pref_task = tok.enc.encode(f"TASK={task_name}\n", allowed_special={"<|endoftext|>"})
    pref_source = tok.enc.encode("SOURCE=", allowed_special={"<|endoftext|>"})
    pref_state = tok.enc.encode("\nSTATE=", allowed_special={"<|endoftext|>"})
    pref_temporal = tok.enc.encode("\nTEMPORAL=", allowed_special={"<|endoftext|>"})
    pref_spatial = tok.enc.encode("\nSPATIAL=", allowed_special={"<|endoftext|>"})
    pref_morpho = tok.enc.encode("\nMORPHO=", allowed_special={"<|endoftext|>"})
    pref_ans = tok.enc.encode("\nANS=(", allowed_special={"<|endoftext|>"})
    suffix = tok.enc.encode(")\n<|endoftext|>", allowed_special={"<|endoftext|>"})

    def append_format(tokens: List[int]):
        nonlocal x_text
        for tid in tokens:
            pos = int(x_text.size(1))
            x_text = append_tokens(x_text, [tid], 0)
            format_pos.append(pos)

    append_format(pref_task)

    append_format(pref_source)
    source, _, x_text, acts = sample_value_from_candidates(
        model=model,
        X_eeg_tokens_1=X_eeg_tokens_1,
        x_eeg_emb_1=x_eeg_emb_1,
        input_time_1=input_time_1,
        eeg_mask_1=eeg_mask_1,
        x_text=x_text,
        candidate_values=allowed_source_values(task_name),
        value_token_map=source_token_map,
        temperature=slot_temperature,
        kind="slot",
    )
    action_recs.extend(acts)

    append_format(pref_state)
    state, _, x_text, acts = sample_value_from_candidates(
        model=model,
        X_eeg_tokens_1=X_eeg_tokens_1,
        x_eeg_emb_1=x_eeg_emb_1,
        input_time_1=input_time_1,
        eeg_mask_1=eeg_mask_1,
        x_text=x_text,
        candidate_values=allowed_state_values(task_name, source),
        value_token_map=state_token_map,
        temperature=slot_temperature,
        kind="slot",
    )
    action_recs.extend(acts)

    append_format(pref_temporal)
    temporal, _, x_text, acts = sample_value_from_candidates(
        model=model,
        X_eeg_tokens_1=X_eeg_tokens_1,
        x_eeg_emb_1=x_eeg_emb_1,
        input_time_1=input_time_1,
        eeg_mask_1=eeg_mask_1,
        x_text=x_text,
        candidate_values=allowed_temporal_values(task_name, source, state),
        value_token_map=temporal_token_map,
        temperature=slot_temperature,
        kind="slot",
    )
    action_recs.extend(acts)

    append_format(pref_spatial)
    spatial, _, x_text, acts = sample_value_from_candidates(
        model=model,
        X_eeg_tokens_1=X_eeg_tokens_1,
        x_eeg_emb_1=x_eeg_emb_1,
        input_time_1=input_time_1,
        eeg_mask_1=eeg_mask_1,
        x_text=x_text,
        candidate_values=allowed_spatial_values(task_name, source, state, temporal),
        value_token_map=spatial_token_map,
        temperature=slot_temperature,
        kind="slot",
    )
    action_recs.extend(acts)

    append_format(pref_morpho)
    morpho, _, x_text, acts = sample_value_from_candidates(
        model=model,
        X_eeg_tokens_1=X_eeg_tokens_1,
        x_eeg_emb_1=x_eeg_emb_1,
        input_time_1=input_time_1,
        eeg_mask_1=eeg_mask_1,
        x_text=x_text,
        candidate_values=allowed_morpho_values(task_name, source, state, temporal, spatial),
        value_token_map=morpho_token_map,
        temperature=slot_temperature,
        kind="slot",
    )
    action_recs.extend(acts)

    pred_slots = {
        "source": source,
        "state": state,
        "temporal": temporal,
        "spatial": spatial,
        "morpho": morpho,
    }

    append_format(pref_ans)
    answer_value, _, x_text, acts = sample_value_from_candidates(
        model=model,
        X_eeg_tokens_1=X_eeg_tokens_1,
        x_eeg_emb_1=x_eeg_emb_1,
        input_time_1=input_time_1,
        eeg_mask_1=eeg_mask_1,
        x_text=x_text,
        candidate_values=allowed_answer_values(task_name, pred_slots),
        value_token_map=answer_token_maps[task_name],
        temperature=ans_temperature,
        kind="ans",
    )
    action_recs.extend(acts)

    append_format(suffix)

    seq_ids_1d = x_text[0].detach().cpu()
    labels = label_space_for_task(task_name)
    y_pred = labels.index(answer_value) if answer_value in labels else 0
    txt = safe_decode(tok, x_text[0].tolist())
    tail = txt.split("DSL:\n", 1)[1] if "DSL:\n" in txt else txt
    return seq_ids_1d, action_recs, format_pos, y_pred, pred_slots, tail[-1000:]


def compute_rollout_losses_teacher_forcing_multitask(
    model: NeuroLM,
    ref_model: NeuroLM,
    X_eeg_tokens_1: torch.Tensor,
    x_eeg_emb_1: torch.Tensor,
    x_eeg_emb_ref_1: torch.Tensor,
    input_time_1: torch.Tensor,
    eeg_mask_1: torch.Tensor,
    seq_ids_1d: torch.Tensor,
    prompt_len: int,
    action_recs: List[dict],
    format_pos: List[int],
    format_coef: float,
    bc_coef: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x_eeg_emb_1.device
    seq = seq_ids_1d.to(device=device, dtype=torch.long).unsqueeze(0)
    L_eeg = int(x_eeg_emb_1.size(1))

    gpt_mask = build_gpt_mask_multitask(X_eeg_tokens_1, seq, eeg_mask_1, input_time_1)

    logits_pol, _, _ = model.GPT2(
        x_eeg=x_eeg_emb_1,
        y_eeg='nan',
        x_text=seq,
        y_text='nan',
        eeg_time_idx=input_time_1,
        eeg_mask=eeg_mask_1,
        eeg_text_mask=gpt_mask,
    )
    with torch.no_grad():
        logits_ref, _, _ = ref_model.GPT2(
            x_eeg=x_eeg_emb_ref_1,
            y_eeg='nan',
            x_text=seq,
            y_text='nan',
            eeg_time_idx=input_time_1,
            eeg_mask=eeg_mask_1,
            eeg_text_mask=gpt_mask,
        )

    logp_sum = torch.tensor(0.0, device=device)
    kl_sum = torch.tensor(0.0, device=device)
    ent_slot_sum = torch.tensor(0.0, device=device)

    def constrained_terms(pos_text: int, allowed: List[int], temp: float):
        if pos_text < 0 or pos_text >= seq.size(1):
            z = torch.tensor(0.0, device=device)
            return z, z, z

        tid = int(seq[0, pos_text].item())
        row = L_eeg + pos_text - 1
        if row < 0 or row >= logits_pol.size(1):
            z = torch.tensor(0.0, device=device)
            return z, z, z

        lp_row = logits_pol[0, row, :50257]
        lr_row = logits_ref[0, row, :50257]
        idx = torch.tensor(allowed, device=device, dtype=torch.long)
        lp_allowed = lp_row.index_select(0, idx) / max(float(temp), 1e-6)
        lr_allowed = lr_row.index_select(0, idx) / max(float(temp), 1e-6)
        p = torch.softmax(lp_allowed, dim=-1)
        q = torch.softmax(lr_allowed, dim=-1)
        j = allowed.index(tid) if tid in allowed else 0
        logp = torch.log(p[j] + 1e-12)
        kl = torch.sum(p * (torch.log(p + 1e-12) - torch.log(q + 1e-12)))
        ent = -torch.sum(p * torch.log(p + 1e-12))
        return logp, kl, ent

    for rec in action_recs:
        pos = int(rec["pos"])
        if pos <= 0:
            continue
        logp, kl, ent = constrained_terms(pos, rec["allowed"], float(rec["temp"]))
        logp_sum = logp_sum + logp
        kl_sum = kl_sum + kl
        if rec.get("kind", "slot") != "ans":
            ent_slot_sum = ent_slot_sum + ent

    reg_loss = torch.tensor(0.0, device=device)

    if format_coef > 0 and len(format_pos) > 0:
        nll = torch.tensor(0.0, device=device)
        cnt = 0
        for pos in format_pos:
            if pos <= 0 or pos >= seq.size(1):
                continue
            tid = int(seq[0, pos].item())
            if tid >= 50257:
                continue
            row = L_eeg + pos - 1
            if row < 0 or row >= logits_pol.size(1):
                continue
            lp_row = logits_pol[0, row, :50257]
            nll = nll - F.log_softmax(lp_row, dim=-1)[tid]
            cnt += 1
        if cnt > 0:
            reg_loss = reg_loss + format_coef * (nll / cnt)

    if bc_coef > 0:
        nll = torch.tensor(0.0, device=device)
        cnt = 0
        for pos in range(prompt_len, int(seq.size(1))):
            if pos <= 0 or pos >= seq.size(1):
                continue
            tid = int(seq[0, pos].item())
            if tid >= 50257:
                continue
            row = L_eeg + pos - 1
            if row < 0 or row >= logits_pol.size(1):
                continue
            lp_row = logits_pol[0, row, :50257]
            nll = nll - F.log_softmax(lp_row, dim=-1)[tid]
            cnt += 1
        if cnt > 0:
            reg_loss = reg_loss + bc_coef * (nll / cnt)

    return logp_sum, kl_sum, ent_slot_sum, reg_loss


@torch.no_grad()
def decode_structured_constrained_greedy_multitask_v2(
    model: NeuroLM,
    tok,
    source_token_map: Dict[str, List[int]],
    state_token_map: Dict[str, List[int]],
    temporal_token_map: Dict[str, List[int]],
    spatial_token_map: Dict[str, List[int]],
    morpho_token_map: Dict[str, List[int]],
    answer_token_maps: Dict[str, Dict[str, List[int]]],
    task_name: str,
    X_eeg_tokens_1: torch.Tensor,
    x_eeg_emb_1: torch.Tensor,
    input_time_1: torch.Tensor,
    eeg_mask_1: torch.Tensor,
    prompt_ids_1d: torch.Tensor,
    slot_temperature: float,
    ans_temperature: float,
) -> Tuple[int, Dict[str, str], str, Optional[float]]:
    device = X_eeg_tokens_1.device
    x_text = prompt_ids_1d.unsqueeze(0).to(device)

    pref_task = tok.enc.encode(f"TASK={task_name}\n", allowed_special={"<|endoftext|>"})
    pref_source = tok.enc.encode("SOURCE=", allowed_special={"<|endoftext|>"})
    pref_state = tok.enc.encode("\nSTATE=", allowed_special={"<|endoftext|>"})
    pref_temporal = tok.enc.encode("\nTEMPORAL=", allowed_special={"<|endoftext|>"})
    pref_spatial = tok.enc.encode("\nSPATIAL=", allowed_special={"<|endoftext|>"})
    pref_morpho = tok.enc.encode("\nMORPHO=", allowed_special={"<|endoftext|>"})
    pref_ans = tok.enc.encode("\nANS=(", allowed_special={"<|endoftext|>"})
    suffix = tok.enc.encode(")\n<|endoftext|>", allowed_special={"<|endoftext|>"})

    def append_format(tokens: List[int]):
        nonlocal x_text
        x_text = append_tokens(x_text, tokens, 0)

    append_format(pref_task)
    append_format(pref_source)
    source, _, x_text = greedy_value_from_candidates(
        model, X_eeg_tokens_1, x_eeg_emb_1, input_time_1, eeg_mask_1, x_text,
        candidate_values=allowed_source_values(task_name),
        value_token_map=source_token_map,
        temperature=slot_temperature,
    )

    append_format(pref_state)
    state, _, x_text = greedy_value_from_candidates(
        model, X_eeg_tokens_1, x_eeg_emb_1, input_time_1, eeg_mask_1, x_text,
        candidate_values=allowed_state_values(task_name, source),
        value_token_map=state_token_map,
        temperature=slot_temperature,
    )

    append_format(pref_temporal)
    temporal, _, x_text = greedy_value_from_candidates(
        model, X_eeg_tokens_1, x_eeg_emb_1, input_time_1, eeg_mask_1, x_text,
        candidate_values=allowed_temporal_values(task_name, source, state),
        value_token_map=temporal_token_map,
        temperature=slot_temperature,
    )

    append_format(pref_spatial)
    spatial, _, x_text = greedy_value_from_candidates(
        model, X_eeg_tokens_1, x_eeg_emb_1, input_time_1, eeg_mask_1, x_text,
        candidate_values=allowed_spatial_values(task_name, source, state, temporal),
        value_token_map=spatial_token_map,
        temperature=slot_temperature,
    )

    append_format(pref_morpho)
    morpho, _, x_text = greedy_value_from_candidates(
        model, X_eeg_tokens_1, x_eeg_emb_1, input_time_1, eeg_mask_1, x_text,
        candidate_values=allowed_morpho_values(task_name, source, state, temporal, spatial),
        value_token_map=morpho_token_map,
        temperature=slot_temperature,
    )

    pred_slots = {
        "source": source,
        "state": state,
        "temporal": temporal,
        "spatial": spatial,
        "morpho": morpho,
    }

    append_format(pref_ans)
    score_abnormal = None
    if task_name == "TUAB":
        score_abnormal = tuab_abnormal_score_from_value_scores(
            score_value_options_normalized(
                model=model,
                X_eeg_tokens_1=X_eeg_tokens_1,
                x_eeg_emb_1=x_eeg_emb_1,
                input_time_1=input_time_1,
                eeg_mask_1=eeg_mask_1,
                x_text=x_text,
                candidate_values=TUAB_LABELS,
                value_token_map=answer_token_maps[task_name],
                temperature=ans_temperature,
            )
        )
    answer_value, _, x_text = greedy_value_from_candidates(
        model, X_eeg_tokens_1, x_eeg_emb_1, input_time_1, eeg_mask_1, x_text,
        candidate_values=allowed_answer_values(task_name, pred_slots),
        value_token_map=answer_token_maps[task_name],
        temperature=ans_temperature,
    )
    append_format(suffix)

    labels = label_space_for_task(task_name)
    y_pred = labels.index(answer_value) if answer_value in labels else 0
    txt = safe_decode(tok, x_text[0].tolist())
    tail = txt.split("DSL:\n", 1)[1] if "DSL:\n" in txt else txt
    return y_pred, pred_slots, tail[-1000:], score_abnormal


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
def eval_multitask_loader_grpo_v2(
    model: NeuroLM,
    tok,
    source_token_map: Dict[str, List[int]],
    state_token_map: Dict[str, List[int]],
    temporal_token_map: Dict[str, List[int]],
    spatial_token_map: Dict[str, List[int]],
    morpho_token_map: Dict[str, List[int]],
    answer_token_maps: Dict[str, Dict[str, List[int]]],
    task_name: str,
    loader,
    device: torch.device,
    text_max_len: int,
    temperature: float,
    print_samples: int = 0,
) -> Dict[str, object]:
    model.eval()
    prompt_ids_t = build_prompt_ids(tok, task_name, text_max_len).to(device)

    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    tuab_scores: List[float] = []
    inconsistent = 0
    format_valid_count = 0
    kcs_count = 0
    car_count = 0
    specificity_sum = 0.0
    shown = 0

    for batch in loader:
        X_eeg, _t, Y, input_chans, input_time, eeg_mask, _gm = batch
        X_eeg = X_eeg.to(device)
        Y = Y.to(device)
        input_chans = input_chans.to(device)
        input_time = input_time.to(device)
        eeg_mask = eeg_mask.to(device)

        x_eeg_emb = encode_eeg_only(model, X_eeg, input_chans, input_time, eeg_mask)

        B = X_eeg.size(0)
        labels = label_space_for_task(task_name)
        for i in range(B):
            y_pred, pred_slots, tail, score_abnormal = decode_structured_constrained_greedy_multitask_v2(
                model=model,
                tok=tok,
                source_token_map=source_token_map,
                state_token_map=state_token_map,
                temporal_token_map=temporal_token_map,
                spatial_token_map=spatial_token_map,
                morpho_token_map=morpho_token_map,
                answer_token_maps=answer_token_maps,
                task_name=task_name,
                X_eeg_tokens_1=X_eeg[i:i+1],
                x_eeg_emb_1=x_eeg_emb[i:i+1],
                input_time_1=input_time[i:i+1],
                eeg_mask_1=eeg_mask[i:i+1],
                prompt_ids_1d=prompt_ids_t,
                slot_temperature=temperature,
                ans_temperature=temperature,
            )
            y_true = int(Y[i].item())
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)
            if task_name == "TUAB" and score_abnormal is not None:
                tuab_scores.append(float(score_abnormal))
            pred_label = labels[y_pred]
            format_ok, kcs_ok, car_bad, cand = chain_eval_flags(task_name, pred_label, pred_slots)
            format_valid_count += int(format_ok)
            kcs_count += int(kcs_ok)
            car_count += int(car_bad)
            specificity_sum += chain_specificity_score(task_name, cand, format_ok)
            if car_bad:
                inconsistent += 1
            if shown < print_samples:
                shown += 1
                print(f"----- sample gen tail ({task_name}, multitask GRPO v2) -----")
                print(tail)

    y_true_np = np.array(y_true_all, dtype=np.int64)
    y_pred_np = np.array(y_pred_all, dtype=np.int64)
    label_ids = list(range(len(label_space_for_task(task_name))))
    rec = recall_score(y_true_np, y_pred_np, average=None, labels=label_ids, zero_division=0)
    cm = confusion_matrix(y_true_np, y_pred_np, labels=label_ids).tolist()
    out = {
        "n_eval": int(len(y_true_np)),
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_np, y_pred_np)),
        "f1_weighted": float(f1_score(y_true_np, y_pred_np, average="weighted")),
        "format_validity": float(format_valid_count / max(len(y_true_np), 1)),
        "knowledge_constraint_satisfaction": float(kcs_count / max(len(y_true_np), 1)),
        "chain_answer_contradiction_rate": float(car_count / max(len(y_true_np), 1)),
        "specificity": float(specificity_sum / max(len(y_true_np), 1)),
        "chain_inconsistency_rate": float(inconsistent / max(len(y_true_np), 1)),
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


def maybe_run_eval_suite(
    tag: str,
    model: NeuroLM,
    tok,
    source_token_map: Dict[str, List[int]],
    state_token_map: Dict[str, List[int]],
    temporal_token_map: Dict[str, List[int]],
    spatial_token_map: Dict[str, List[int]],
    morpho_token_map: Dict[str, List[int]],
    answer_token_maps: Dict[str, Dict[str, List[int]]],
    specs: Dict[str, object],
    buckets: Dict[str, object],
    seed_datasets: Optional[Dict[str, object]],
    args,
    device: torch.device,
    eval_bs: int,
    split: str,
    tasks: Optional[List[str]] = None,
) -> Dict[str, Dict[str, object]]:
    results = {}
    split_mode = args.val_eval_mode if split == "val" else args.test_eval_mode
    split_per_class = args.val_eval_per_class if split == "val" else args.test_eval_per_class
    split_max_samples = args.val_eval_max_samples if split == "val" else args.test_eval_max_samples
    split_min_per_class = args.val_eval_min_per_class if split == "val" else args.test_eval_min_per_class
    seed_base = args.seed if split == "val" else args.seed + 1000
    print_samples = args.print_eval_samples if split == "val" else 0

    eval_tasks = tasks if tasks is not None else ["TUAB", "TUEV", "HMC", "SEED"]
    for task_name in eval_tasks:
        if task_name == "SEED":
            if seed_datasets is None:
                raise ValueError("seed_datasets must be provided when evaluating SEED")
            ds_seed = seed_datasets[split]
            indices_sel, counts_sel = build_eval_indices_for_seed(
                buckets=buckets[task_name],
                mode=split_mode,
                per_class=split_per_class,
                seed=seed_base + TASK_NAME_TO_OFFSET[task_name],
                max_samples=split_max_samples,
                min_per_class=split_min_per_class,
            )
            loader = build_eval_loader_for_indices(ds_seed, indices_sel, eval_bs, args.num_workers)
            selected_n = len(indices_sel)
        else:
            split_dir = specs[task_name].val_dir if split == "val" else specs[task_name].test_dir
            if split == "test" and (not os.path.isdir(split_dir) or sum(len(v) for v in buckets[task_name].values()) <= 0):
                print(f"[{tag}-collection] skip {task_name}: no {split} split found under {split_dir}")
                continue
            loader, files_sel, counts_sel = build_eval_loader(
                task_name=task_name,
                loader_cls=specs[task_name].loader_cls,
                split_dir=split_dir,
                buckets=buckets[task_name],
                mode=split_mode,
                per_class=split_per_class,
                seed=seed_base + TASK_NAME_TO_OFFSET[task_name],
                eeg_max_len=args.eeg_max_len,
                text_max_len=args.text_max_len,
                batch_size=eval_bs,
                num_workers=args.num_workers,
                max_samples=split_max_samples,
                min_per_class=split_min_per_class,
            )
            selected_n = len(files_sel)
        print(
            f"[{tag}-collection] task={task_name} mode={split_mode} "
            f"selected={selected_n} per_class={counts_sel} "
            f"max_samples={split_max_samples if split_mode == 'proportional' else 0} "
            f"min_per_class={split_min_per_class if split_mode == 'proportional' else 0}"
        )
        res = eval_multitask_loader_grpo_v2(
            model=model,
            tok=tok,
            source_token_map=source_token_map,
            state_token_map=state_token_map,
            temporal_token_map=temporal_token_map,
            spatial_token_map=spatial_token_map,
            morpho_token_map=morpho_token_map,
            answer_token_maps=answer_token_maps,
            task_name=task_name,
            loader=loader,
            device=device,
            text_max_len=args.text_max_len,
            temperature=args.eval_temperature,
            print_samples=print_samples,
        )
        print_eval(f"{task_name}({split},main-structured-grpo)", res)
        results[task_name] = res
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_ckpt", type=str, required=True)
    ap.add_argument("--tuab_root", type=str, required=True)
    ap.add_argument("--tuev_root", type=str, required=True)
    ap.add_argument("--hmc_root", type=str, required=True)
    ap.add_argument("--seed_root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--eeg_max_len", type=int, default=1024)
    ap.add_argument("--text_max_len", type=int, default=768)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--balanced_sampler", action="store_true")
    ap.add_argument("--task_mix", type=str, default="1,1,1,1", help="TUAB,TUEV,HMC,SEED sampling ratio")

    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--grpo_steps", type=int, default=1000)
    ap.add_argument("--grpo_lr", type=float, default=2e-6)
    ap.add_argument("--slot_temperature", type=float, default=0.9)
    ap.add_argument("--ans_temperature", type=float, default=0.8)
    ap.add_argument("--eval_temperature", type=float, default=0.7)
    ap.add_argument("--kl_coef", type=float, default=0.12)
    ap.add_argument("--entropy_coef", type=float, default=0.003)
    ap.add_argument("--format_coef", type=float, default=0.03)
    ap.add_argument("--bc_coef", type=float, default=0.01)
    ap.add_argument("--adv_normalize", action="store_true")

    ap.add_argument("--reward_scheme", type=str, default="inv_sqrt_freq", choices=["uniform", "inv_freq", "inv_sqrt_freq"])
    ap.add_argument("--correct_reward_scale", type=float, default=1.0, help="Base scale for the answer reward.")
    ap.add_argument("--wrong_penalty", type=float, default=0.55)
    ap.add_argument("--answer_reward_weight", type=float, default=1.0)
    ap.add_argument("--validity_reward_weight", type=float, default=0.06)
    ap.add_argument("--invalidity_penalty", type=float, default=0.06)
    ap.add_argument("--chain_margin_weight", type=float, default=0.08)
    ap.add_argument("--counterfactual_weight", type=float, default=0.08)
    ap.add_argument("--disable_seed_chain_margin", action="store_true", help="Disable canonical-factor margin for SEED; SEED emotion factors are weak.")
    ap.add_argument("--consistency_reward", type=float, default=0.06)
    ap.add_argument("--inconsistency_penalty", type=float, default=0.06)
    ap.add_argument("--singleton_bonus", type=float, default=0.01)

    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--print_eval_samples", type=int, default=3)
    ap.set_defaults(skip_baseline_eval=True)
    ap.add_argument("--skip_baseline_eval", dest="skip_baseline_eval", action="store_true")
    ap.add_argument("--run_baseline_eval", dest="skip_baseline_eval", action="store_false")
    ap.add_argument("--val_eval_mode", type=str, default="balanced", choices=["balanced", "full", "proportional"])
    ap.add_argument("--val_eval_per_class", type=int, default=500)
    ap.add_argument("--val_eval_max_samples", type=int, default=0)
    ap.add_argument("--val_eval_min_per_class", type=int, default=0)
    ap.add_argument("--test_eval_mode", type=str, default="proportional", choices=["balanced", "full", "proportional"])
    ap.add_argument("--test_eval_per_class", type=int, default=500)
    ap.add_argument("--test_eval_max_samples", type=int, default=12000)
    ap.add_argument("--test_eval_min_per_class", type=int, default=200)
    ap.add_argument("--test_checkpoint_mode", type=str, default="task_best", choices=["task_best", "joint_best"])
    ap.add_argument("--joint_weights", type=str, default="0.25,0.25,0.25,0.25")

    ap.add_argument("--save_dir", type=str, default="runs_main_structured_grpo")
    ap.add_argument("--log_dir", type=str, default="logs")
    ap.add_argument("--log_prefix", type=str, default="main_structured_grpo")
    args = ap.parse_args()

    set_seed(args.seed)
    if setup_run_logging is not None:
        setup_run_logging(args.log_dir, prefix=args.log_prefix)

    device = torch.device(args.device)
    tok = build_tokmap()
    joint_weights = parse_joint_weights(args.joint_weights)

    specs = {
        "TUAB": build_task_spec("TUAB", args.tuab_root),
        "TUEV": build_task_spec("TUEV", args.tuev_root),
        "HMC": build_task_spec("HMC", args.hmc_root),
        "SEED": build_task_spec("SEED", args.seed_root),
    }

    data = {}
    sampler_weights = {}
    class_weights = {}
    for task_name, spec in specs.items():
        if task_name == "SEED":
            ds_seed_train_tmp = build_seed_dataset(spec.seed_h5_path, "train", args.eeg_max_len, args.text_max_len)
            ds_seed_val_tmp = build_seed_dataset(spec.seed_h5_path, "val", args.eeg_max_len, args.text_max_len)
            ds_seed_test_tmp = build_seed_dataset(spec.seed_h5_path, "test", args.eeg_max_len, args.text_max_len)
            seed_train_buckets_tmp, seed_train_counts_tmp = build_seed_index_buckets(ds_seed_train_tmp)
            seed_val_buckets_tmp, seed_val_counts_tmp = build_seed_index_buckets(ds_seed_val_tmp)
            seed_test_buckets_tmp, seed_test_counts_tmp = build_seed_index_buckets(ds_seed_test_tmp)
            sampler_weights[task_name] = seed_sampler_weights_from_buckets(ds_seed_train_tmp, seed_train_buckets_tmp)
            class_weights[task_name] = make_class_weights(seed_train_counts_tmp, args.reward_scheme)
            data[task_name] = {
                "train_ds": ds_seed_train_tmp,
                "val_ds": ds_seed_val_tmp,
                "test_ds": ds_seed_test_tmp,
                "train_buckets": seed_train_buckets_tmp,
                "val_buckets": seed_val_buckets_tmp,
                "test_buckets": seed_test_buckets_tmp,
                "train_class_counts": seed_train_counts_tmp,
                "val_class_counts": seed_val_counts_tmp,
                "test_class_counts": seed_test_counts_tmp,
            }
            print(
                f"[{task_name}] train={len(ds_seed_train_tmp)} val={len(ds_seed_val_tmp)} test={len(ds_seed_test_tmp)} "
                f"train_class_counts={seed_train_counts_tmp}"
            )
            print(f"[reward:{task_name}] scheme={args.reward_scheme} class_weights(normed)={[round(x, 4) for x in class_weights[task_name]]}")
            continue
        train_files = list_pkls(spec.train_dir)
        val_files = list_pkls(spec.val_dir)
        test_files = list_pkls(spec.test_dir) if os.path.isdir(spec.test_dir) else []
        counts, weights = scan_labels_for_sampler(task_name, spec.train_dir, train_files, len(spec.labels))
        sampler_weights[task_name] = weights
        class_weights[task_name] = make_class_weights(counts, args.reward_scheme)
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
        print(f"[reward:{task_name}] scheme={args.reward_scheme} class_weights(normed)={[round(x, 4) for x in class_weights[task_name]]}")

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
        "SEED": data["SEED"]["train_ds"],
    }

    val_buckets = {}
    test_buckets = {}
    for task_name, spec in specs.items():
        if task_name == "SEED":
            val_buckets[task_name] = data[task_name]["val_buckets"]
            test_buckets[task_name] = data[task_name]["test_buckets"]
            print(
                f"[{task_name}] val_class_counts={data[task_name]['val_class_counts']} "
                f"test_class_counts={data[task_name]['test_class_counts']}"
            )
            continue
        val_buckets[task_name], val_counts = build_label_buckets(
            task_name,
            spec.val_dir,
            data[task_name]["val_files"],
            len(spec.labels),
        )
        test_buckets[task_name], test_counts = build_label_buckets(
            task_name,
            spec.test_dir,
            data[task_name]["test_files"],
            len(spec.labels),
        )
        print(f"[{task_name}] val_class_counts={val_counts} test_class_counts={test_counts}")

    train_loaders = {
        k: maybe_make_loader(
            ds_train[k],
            args.batch_size,
            args.num_workers,
            sampler_weights[k],
            args.balanced_sampler,
            True,
            True,
        )
        for k in specs
    }
    eval_bs = max(1, int(args.batch_size * 3 // 2))

    base_ckpt = torch.load(args.sft_ckpt, map_location="cpu")
    model_args = base_ckpt.get("model_args", {})
    model = NeuroLM(GPTConfig(**model_args), init_from="scratch")
    ref_model = NeuroLM(GPTConfig(**model_args), init_from="scratch")

    state_dict = base_ckpt.get("model", base_ckpt.get("state_dict", {}))
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    ref_model.load_state_dict(model.state_dict(), strict=False)
    print(f"number of parameters: {model.get_num_params()/1e6:.2f}M")
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    print(f"[init] loaded Structured SFT checkpoint for main structured GRPO: {args.sft_ckpt}")
    model.to(device)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    source_token_map = build_value_token_map(tok, list(SOURCE_VALUES))
    state_token_map = build_value_token_map(tok, list(STATE_VALUES))
    temporal_token_map = build_value_token_map(tok, list(TEMPORAL_VALUES))
    spatial_token_map = build_value_token_map(tok, list(SPATIAL_VALUES))
    morpho_token_map = build_value_token_map(tok, list(MORPHO_VALUES))
    answer_token_maps = {
        "TUAB": build_value_token_map(tok, TUAB_LABELS),
        "TUEV": build_value_token_map(tok, TUEV_LABELS),
        "HMC": build_value_token_map(tok, HMC_LABELS),
        "SEED": build_value_token_map(tok, SEED_LABELS),
    }
    prompt_ids = {task_name: build_prompt_ids(tok, task_name, args.text_max_len).to(device) for task_name in specs}

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.grpo_lr)
    train_iters = {k: iter(train_loaders[k]) for k in specs}
    schedule = build_mix_schedule(args.task_mix)

    os.makedirs(args.save_dir, exist_ok=True)
    last_path = os.path.join(args.save_dir, "multitask_main_structured_grpo_last.pt")
    best_tuab_path = os.path.join(args.save_dir, "multitask_main_structured_grpo_tuab_best.pt")
    best_tuev_path = os.path.join(args.save_dir, "multitask_main_structured_grpo_tuev_best.pt")
    best_hmc_path = os.path.join(args.save_dir, "multitask_main_structured_grpo_hmc_best.pt")
    best_seed_path = os.path.join(args.save_dir, "multitask_main_structured_grpo_seed_best.pt")
    best_joint_path = os.path.join(args.save_dir, "multitask_main_structured_grpo_joint_best.pt")

    best_tuab = -1.0
    best_tuev = -1.0
    best_hmc = -1.0
    best_seed = -1.0
    best_joint = -1.0

    print(
        f"[Main structured GRPO] start steps={args.grpo_steps} lr={args.grpo_lr} task_mix={schedule} "
        f"group_size={args.group_size} kl={args.kl_coef} entropy={args.entropy_coef} "
        f"weights(answer/validity/consistency/margin/cf)="
        f"{args.answer_reward_weight}/{args.validity_reward_weight}/{args.consistency_reward}/"
        f"{args.chain_margin_weight}/{args.counterfactual_weight} "
        f"components={MAIN_METHOD_NAME}"
    )
    print(
        f"[eval-plan] val mode={args.val_eval_mode} per_class={args.val_eval_per_class} | "
        f"test mode={args.test_eval_mode} per_class={args.test_eval_per_class} "
        f"max_samples={args.test_eval_max_samples if args.test_eval_mode == 'proportional' else 0} "
        f"min_per_class={args.test_eval_min_per_class if args.test_eval_mode == 'proportional' else 0} | "
        f"test_checkpoint_mode={args.test_checkpoint_mode} | "
        f"joint_weights={joint_weights}"
    )

    if not args.skip_baseline_eval:
        baseline_results = maybe_run_eval_suite(
            tag="baseline-val",
            model=model,
            tok=tok,
            source_token_map=source_token_map,
            state_token_map=state_token_map,
            temporal_token_map=temporal_token_map,
            spatial_token_map=spatial_token_map,
            morpho_token_map=morpho_token_map,
            answer_token_maps=answer_token_maps,
            specs=specs,
            buckets=val_buckets,
            seed_datasets={"val": data["SEED"]["val_ds"], "test": data["SEED"]["test_ds"]},
            args=args,
            device=device,
            eval_bs=eval_bs,
            split="val",
        )
        best_tuab = float(baseline_results["TUAB"]["balanced_accuracy"])
        best_tuev = float(baseline_results["TUEV"]["balanced_accuracy"])
        best_hmc = float(baseline_results["HMC"]["balanced_accuracy"])
        best_seed = float(baseline_results["SEED"]["balanced_accuracy"])
        best_joint = float(joint_score(baseline_results["TUAB"], baseline_results["TUEV"], baseline_results["HMC"], baseline_results["SEED"], joint_weights))
        print(f"[baseline joint(val)] score={best_joint:.4f}")
        save_ckpt(best_tuab_path, {"model_args": model_args, "model": model.state_dict()})
        save_ckpt(best_tuev_path, {"model_args": model_args, "model": model.state_dict()})
        save_ckpt(best_hmc_path, {"model_args": model_args, "model": model.state_dict()})
        save_ckpt(best_seed_path, {"model_args": model_args, "model": model.state_dict()})
        save_ckpt(best_joint_path, {"model_args": model_args, "model": model.state_dict()})
    else:
        print("[baseline] skipped initial baseline validation; training starts immediately.")

    step = 0
    while step < args.grpo_steps:
        for task_name in schedule:
            if step >= args.grpo_steps:
                break
            step += 1

            batch = next_batch(train_iters, train_loaders, task_name)
            X_eeg, _t, Y, input_chans, input_time, eeg_mask, _gm = batch
            X_eeg = X_eeg.to(device)
            Y = Y.to(device)
            input_chans = input_chans.to(device)
            input_time = input_time.to(device)
            eeg_mask = eeg_mask.to(device)
            B = X_eeg.size(0)

            model.eval()
            with torch.no_grad():
                x_eeg_emb = encode_eeg_only(model, X_eeg, input_chans, input_time, eeg_mask)
                x_eeg_emb_ref = encode_eeg_only(ref_model, X_eeg, input_chans, input_time, eeg_mask)

                rollouts = []
                rewards_mat = torch.zeros((B, args.group_size), device=device, dtype=torch.float32)
                r_ans_mat = torch.zeros_like(rewards_mat)
                r_validity_mat = torch.zeros_like(rewards_mat)
                r_consistency_mat = torch.zeros_like(rewards_mat)
                r_discriminative_mat = torch.zeros_like(rewards_mat)
                r_margin_mat = torch.zeros_like(rewards_mat)
                r_counterfactual_mat = torch.zeros_like(rewards_mat)
                validity_ok_mat = torch.zeros_like(rewards_mat)
                chain_support_mat = torch.zeros_like(rewards_mat)

                for b in range(B):
                    for g in range(args.group_size):
                        seq_ids, action_recs, format_pos, y_pred, pred_slots, _tail = sample_structured_rollout_no_grad_multitask_v2(
                            model=model,
                            tok=tok,
                            source_token_map=source_token_map,
                            state_token_map=state_token_map,
                            temporal_token_map=temporal_token_map,
                            spatial_token_map=spatial_token_map,
                            morpho_token_map=morpho_token_map,
                            answer_token_maps=answer_token_maps,
                            task_name=task_name,
                            X_eeg_tokens_1=X_eeg[b:b+1],
                            x_eeg_emb_1=x_eeg_emb[b:b+1],
                            input_time_1=input_time[b:b+1],
                            eeg_mask_1=eeg_mask[b:b+1],
                            prompt_ids_1d=prompt_ids[task_name],
                            slot_temperature=args.slot_temperature,
                            ans_temperature=args.ans_temperature,
                        )

                        y_true = int(Y[b].item())
                        w_cls = float(class_weights[task_name][y_true])
                        r_validity, validity_ok, _cand = chain_validity_reward_multitask(
                            task_name=task_name,
                            pred_slots=pred_slots,
                            reward_weight=args.validity_reward_weight,
                            invalidity_penalty=args.invalidity_penalty,
                        )
                        answer_base = (
                            args.correct_reward_scale * w_cls
                            if y_pred == y_true
                            else (-args.wrong_penalty * w_cls)
                        )
                        r_ans = args.answer_reward_weight * float(answer_base)
                        r_consistency = chain_answer_consistency_reward_multitask(
                            task_name=task_name,
                            pred_slots=pred_slots,
                            y_pred=y_pred,
                            consistency_reward=args.consistency_reward,
                            inconsistency_penalty=args.inconsistency_penalty,
                            singleton_bonus=args.singleton_bonus,
                        )
                        chain_margin, chain_support, _hard_negative = chain_margin_reward_multitask(
                            task_name=task_name,
                            pred_slots=pred_slots,
                            y_true=y_true,
                        )
                        margin_weight = task_chain_margin_weight(
                            task_name=task_name,
                            base_weight=args.chain_margin_weight,
                            disable_seed_chain_margin=args.disable_seed_chain_margin,
                        )
                        r_margin = float(margin_weight) * max(0.0, float(chain_margin))
                        r_counterfactual = args.counterfactual_weight * counterfactual_sensitivity_reward_multitask(
                            task_name=task_name,
                            pred_slots=pred_slots,
                            y_true=y_true,
                        )
                        r_discriminative = float(r_margin + r_counterfactual)
                        r_total = float(r_ans + r_validity + r_consistency + r_discriminative)

                        rewards_mat[b, g] = r_total
                        r_ans_mat[b, g] = float(r_ans)
                        r_validity_mat[b, g] = float(r_validity)
                        r_consistency_mat[b, g] = float(r_consistency)
                        r_discriminative_mat[b, g] = float(r_discriminative)
                        r_margin_mat[b, g] = float(r_margin)
                        r_counterfactual_mat[b, g] = float(r_counterfactual)
                        validity_ok_mat[b, g] = float(1.0 if validity_ok else 0.0)
                        chain_support_mat[b, g] = float(chain_support)

                        rollouts.append({
                            "b": b,
                            "seq_ids": seq_ids,
                            "action_recs": action_recs,
                            "format_pos": format_pos,
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "pred_slots": pred_slots,
                        })

            model.train()
            x_eeg_emb = encode_eeg_only(model, X_eeg, input_chans, input_time, eeg_mask)
            adv = rewards_mat - rewards_mat.mean(dim=1, keepdim=True)
            if args.adv_normalize:
                adv = adv / (adv.std(dim=1, keepdim=True) + 1e-8)

            idx = 0
            for b in range(B):
                for g in range(args.group_size):
                    rollouts[idx]["adv"] = float(adv[b, g].item())
                    idx += 1

            opt.zero_grad(set_to_none=True)

            loss_pg_acc = 0.0
            loss_kl_acc = 0.0
            loss_ent_acc = 0.0
            loss_reg_acc = 0.0

            for ro in rollouts:
                b = ro["b"]
                adv_i = ro["adv"]
                logp_sum, kl_sum, ent_slot_sum, reg_loss = compute_rollout_losses_teacher_forcing_multitask(
                    model=model,
                    ref_model=ref_model,
                    X_eeg_tokens_1=X_eeg[b:b+1],
                    x_eeg_emb_1=x_eeg_emb[b:b+1],
                    x_eeg_emb_ref_1=x_eeg_emb_ref[b:b+1],
                    input_time_1=input_time[b:b+1],
                    eeg_mask_1=eeg_mask[b:b+1],
                    seq_ids_1d=ro["seq_ids"],
                    prompt_len=int(prompt_ids[task_name].size(0)),
                    action_recs=ro["action_recs"],
                    format_pos=ro["format_pos"],
                    format_coef=args.format_coef,
                    bc_coef=args.bc_coef,
                )

                adv_t = torch.tensor(adv_i, device=device, dtype=torch.float32)
                loss_pg = -(adv_t * logp_sum)
                loss_kl = args.kl_coef * kl_sum
                loss_ent = -args.entropy_coef * ent_slot_sum if args.entropy_coef > 0 else torch.tensor(0.0, device=device)
                loss = loss_pg + loss_kl + loss_ent + reg_loss
                loss = loss / max(1.0, float(B * args.group_size))
                loss.backward()

                loss_pg_acc += float(loss_pg.item())
                loss_kl_acc += float(loss_kl.item())
                loss_ent_acc += float(loss_ent.item())
                loss_reg_acc += float(reg_loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 50 == 0:
                denom = max(1.0, float(B * args.group_size))
                print(
                    f"[step {step}] task={task_name} "
                    f"pg={loss_pg_acc/denom:.4f} "
                    f"kl={loss_kl_acc/denom:.4f} "
                    f"ent={loss_ent_acc/denom:.4f} "
                    f"reg={loss_reg_acc/denom:.4f} "
                    f"reward_mean={rewards_mat.mean().item():.4f} "
                    f"r_ans={r_ans_mat.mean().item():.4f} "
                    f"r_validity={r_validity_mat.mean().item():.4f} "
                    f"r_consistency={r_consistency_mat.mean().item():.4f} "
                    f"r_disc={r_discriminative_mat.mean().item():.4f} "
                    f"r_margin={r_margin_mat.mean().item():.4f} "
                    f"r_cf={r_counterfactual_mat.mean().item():.4f} "
                    f"validity_ok={validity_ok_mat.mean().item():.4f} "
                    f"chain_support={chain_support_mat.mean().item():.4f}"
                )

            if (step % args.eval_every == 0) or (step == args.grpo_steps):
                save_ckpt(last_path, {"model_args": model_args, "model": model.state_dict()})

                val_results = maybe_run_eval_suite(
                    tag=f"val-step{step}",
                    model=model,
                    tok=tok,
                    source_token_map=source_token_map,
                    state_token_map=state_token_map,
                    temporal_token_map=temporal_token_map,
                    spatial_token_map=spatial_token_map,
                    morpho_token_map=morpho_token_map,
                    answer_token_maps=answer_token_maps,
                    specs=specs,
                    buckets=val_buckets,
                    seed_datasets={"val": data["SEED"]["val_ds"], "test": data["SEED"]["test_ds"]},
                    args=args,
                    device=device,
                    eval_bs=eval_bs,
                    split="val",
                )
                jscore = float(joint_score(val_results["TUAB"], val_results["TUEV"], val_results["HMC"], val_results["SEED"], joint_weights))
                print(f"[joint(val)] score={jscore:.4f}")

                if float(val_results["TUAB"]["balanced_accuracy"]) > best_tuab:
                    best_tuab = float(val_results["TUAB"]["balanced_accuracy"])
                    save_ckpt(best_tuab_path, {"model_args": model_args, "model": model.state_dict()})
                    print(f"[save] best TUAB -> {best_tuab_path}")
                if float(val_results["TUEV"]["balanced_accuracy"]) > best_tuev:
                    best_tuev = float(val_results["TUEV"]["balanced_accuracy"])
                    save_ckpt(best_tuev_path, {"model_args": model_args, "model": model.state_dict()})
                    print(f"[save] best TUEV -> {best_tuev_path}")
                if float(val_results["HMC"]["balanced_accuracy"]) > best_hmc:
                    best_hmc = float(val_results["HMC"]["balanced_accuracy"])
                    save_ckpt(best_hmc_path, {"model_args": model_args, "model": model.state_dict()})
                    print(f"[save] best HMC -> {best_hmc_path}")
                if float(val_results["SEED"]["balanced_accuracy"]) > best_seed:
                    best_seed = float(val_results["SEED"]["balanced_accuracy"])
                    save_ckpt(best_seed_path, {"model_args": model_args, "model": model.state_dict()})
                    print(f"[save] best SEED -> {best_seed_path}")
                if jscore > best_joint:
                    best_joint = float(jscore)
                    save_ckpt(best_joint_path, {"model_args": model_args, "model": model.state_dict()})
                    print(f"[save] best JOINT -> {best_joint_path}")

    if args.test_checkpoint_mode == "joint_best":
        if os.path.isfile(best_joint_path):
            load_ckpt_weights(best_joint_path, model, device)
            print(f"[test] loaded joint-best checkpoint: {best_joint_path}")
        else:
            print("[test] joint-best checkpoint not found, using current model weights")
        test_results = maybe_run_eval_suite(
            tag="test-joint-best",
            model=model,
            tok=tok,
            source_token_map=source_token_map,
            state_token_map=state_token_map,
            temporal_token_map=temporal_token_map,
            spatial_token_map=spatial_token_map,
            morpho_token_map=morpho_token_map,
            answer_token_maps=answer_token_maps,
            specs=specs,
            buckets=test_buckets,
            seed_datasets={"val": data["SEED"]["val_ds"], "test": data["SEED"]["test_ds"]},
            args=args,
            device=device,
            eval_bs=eval_bs,
            split="test",
        )
        if all(k in test_results for k in ["TUAB", "TUEV", "HMC", "SEED"]):
            joint_best_test = joint_score(
                test_results["TUAB"],
                test_results["TUEV"],
                test_results["HMC"],
                test_results["SEED"],
                joint_weights,
            )
            print(f"[joint-best joint(test)] score={joint_best_test:.4f}")
    else:
        task_best_paths = {
            "TUAB": best_tuab_path,
            "TUEV": best_tuev_path,
            "HMC": best_hmc_path,
            "SEED": best_seed_path,
        }
        test_results = {}
        for test_task, ckpt_path in task_best_paths.items():
            if os.path.isfile(ckpt_path):
                load_ckpt_weights(ckpt_path, model, device)
                print(f"[test:{test_task}] loaded task-best checkpoint: {ckpt_path}")
            elif os.path.isfile(best_joint_path):
                load_ckpt_weights(best_joint_path, model, device)
                print(f"[test:{test_task}] task-best missing, loaded joint-best checkpoint: {best_joint_path}")
            else:
                print(f"[test:{test_task}] no best checkpoint found, using current model weights")

            res_one = maybe_run_eval_suite(
                tag=f"test-{test_task}-best",
                model=model,
                tok=tok,
                source_token_map=source_token_map,
                state_token_map=state_token_map,
                temporal_token_map=temporal_token_map,
                spatial_token_map=spatial_token_map,
                morpho_token_map=morpho_token_map,
                answer_token_maps=answer_token_maps,
                specs=specs,
                buckets=test_buckets,
                seed_datasets={"val": data["SEED"]["val_ds"], "test": data["SEED"]["test_ds"]},
                args=args,
                device=device,
                eval_bs=eval_bs,
                split="test",
                tasks=[test_task],
            )
            test_results.update(res_one)

        if all(k in test_results for k in ["TUAB", "TUEV", "HMC", "SEED"]):
            task_best_joint = joint_score(
                test_results["TUAB"],
                test_results["TUEV"],
                test_results["HMC"],
                test_results["SEED"],
                joint_weights,
            )
            print(f"[task-best joint(test)] score={task_best_joint:.4f}")


if __name__ == "__main__":
    main()



