# Multitask Chain Project Clean

This is a lightweight paper-code style layout for the four-task structured
SFT and GRPO pipeline on TUAB, TUEV, HMC, and SEED.

The project keeps the NeuroLM-style organization: two training entry points,
model code under `model/`, preprocessing scripts under `prepare/`, and a small
set of method files for data, chain decoding, rewards, and evaluation.

## Main Commands

Structured SFT:

```bash
python multitask_chain_project_clean/train_sft.py \
  --ckpt NeuroLM/NeuroLM-B.pt \
  --tuab_root data/TUAB/processed \
  --tuev_root data/TUEV/processed \
  --hmc_root data/HMC \
  --seed_root data/SEED
```

Main structured GRPO:

```bash
python multitask_chain_project_clean/train_grpo.py \
  --sft_ckpt runs_main_structured_sft/multitask_chain_sft_joint_best.pt \
  --tuab_root data/TUAB/processed \
  --tuev_root data/TUEV/processed \
  --hmc_root data/HMC \
  --seed_root data/SEED
```

## File Layout

- `train_sft.py`: command-line entry for structured SFT.
- `train_grpo.py`: command-line entry for main structured GRPO.
- `engine_sft.py`: SFT training, validation, checkpointing, and final test loop.
- `engine_grpo.py`: GRPO rollout training, reward aggregation, validation, and final test loop.
- `dataset.py`: base NeuroLM EEG dataset utilities.
- `downstream_dataset.py`: TUAB, TUEV, HMC, and SEED downstream dataset loaders.
- `task_data.py`: task labels, split resolution, samplers, eval subsets, and checkpoint helpers.
- `chain.py`: shared-chain DSL, prompts, slot ontology, teacher slots, and label mapping.
- `decoding.py`: constrained slot/answer decoding and GRPO rollout utilities.
- `rewards.py`: answer, validity, consistency, and discriminative reward components.
- `evaluation.py`: shared SFT/GRPO evaluation helpers and joint score.
- `utils.py`: shared NeuroLM tokenization, EEG encoding, masks, and signal utilities.
- `run_logger.py`: run log redirection and timestamped log file setup.
- `model/`: NeuroLM model implementation.
- `prepare/`: dataset preprocessing scripts.
- `metrics/`: lightweight metric helpers for classification reports.

This folder is intentionally flatter than a packaged Python library, matching
the style of the upstream NeuroLM/LaBraM research code while separating the
current project's method-specific parts.
