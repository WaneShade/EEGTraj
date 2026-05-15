# -*- coding: utf-8 -*-
"""Reward components used by the main structured GRPO method."""

from modules.grpo_core import (
    TASK_FACTOR_WEIGHTS,
    chain_answer_consistency_reward_multitask,
    chain_label_support_multitask,
    chain_margin_reward_multitask,
    chain_validity_reward_multitask,
    counterfactual_sensitivity_reward_multitask,
    make_class_weights,
    task_chain_margin_weight,
)


__all__ = [
    "TASK_FACTOR_WEIGHTS",
    "chain_answer_consistency_reward_multitask",
    "chain_label_support_multitask",
    "chain_margin_reward_multitask",
    "chain_validity_reward_multitask",
    "counterfactual_sensitivity_reward_multitask",
    "make_class_weights",
    "task_chain_margin_weight",
]
