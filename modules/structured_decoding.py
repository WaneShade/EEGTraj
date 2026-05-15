# -*- coding: utf-8 -*-
"""Structured rollout, constrained value decoding, and GRPO loss utilities."""

from modules.grpo_core import (
    allowed_answer_values,
    allowed_morpho_values,
    allowed_source_values,
    allowed_spatial_values,
    allowed_state_values,
    allowed_temporal_values,
    append_tokens,
    build_prompt_ids,
    build_value_token_map,
    compute_rollout_losses_teacher_forcing_multitask,
    decode_structured_constrained_greedy_multitask_v2,
    greedy_one_step,
    greedy_value_from_candidates,
    sample_one_step,
    sample_structured_rollout_no_grad_multitask_v2,
    sample_value_from_candidates,
    score_value_options_normalized,
    tuab_abnormal_score_from_value_scores,
)


__all__ = [
    "allowed_answer_values",
    "allowed_morpho_values",
    "allowed_source_values",
    "allowed_spatial_values",
    "allowed_state_values",
    "allowed_temporal_values",
    "append_tokens",
    "build_prompt_ids",
    "build_value_token_map",
    "compute_rollout_losses_teacher_forcing_multitask",
    "decode_structured_constrained_greedy_multitask_v2",
    "greedy_one_step",
    "greedy_value_from_candidates",
    "sample_one_step",
    "sample_structured_rollout_no_grad_multitask_v2",
    "sample_value_from_candidates",
    "score_value_options_normalized",
    "tuab_abnormal_score_from_value_scores",
]
