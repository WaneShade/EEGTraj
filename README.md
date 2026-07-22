# Multitask Chain Project

This is a lightweight paper-code style layout for the four-task structured
SFT and GRPO pipeline on TUAB, TUEV, HMC, and SEED.

The project keeps the NeuroLM-style organization: two training entry points,
model code under `model/`, preprocessing scripts under `prepare/`, and a small
set of method files for data, chain decoding, rewards, and evaluation.

## Main Commands

Structured SFT:

```bash
python train_sft.py \
  --ckpt NeuroLM/NeuroLM-B.pt \
  --tuab_root data/TUAB/processed \
  --tuev_root data/TUEV/processed \
  --hmc_root data/HMC \
  --seed_root data/SEED
```

Main structured GRPO:

```bash
python train_grpo.py \
  --sft_ckpt runs_main_structured_sft/checkpoint.pt \
  --tuab_root data/TUAB/processed \
  --tuev_root data/TUEV/processed \
  --hmc_root data/HMC \
  --seed_root data/SEED
```

