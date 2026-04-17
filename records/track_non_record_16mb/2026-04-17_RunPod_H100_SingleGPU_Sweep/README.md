# RunPod H100 Single-GPU Sweep

This folder captures an initial non-record participation pass run on a single `NVIDIA H100 80GB HBM3` RunPod pod.

The goal of this sweep is practical orientation rather than leaderboard chasing:

- verify the published `sp1024` dataset path on RunPod
- reproduce a baseline on 1xH100 using the upstream `train_gpt.py`
- test a small set of low-risk hyperparameter changes drawn from earlier public records
- keep the artifact under the 16MB cap and document exact commands, logs, and outcomes

## Trial Set

Planned trials:

1. `baseline_sp1024_single_h100`
2. `lower_lr_sp1024_single_h100`
3. `seq4096_lowerlr_single_h100`

## Environment

- Pod provider: RunPod
- GPU: `1xH100 80GB HBM3`
- Repo: `meett07/meett07-parameter-golf`
- Dataset: `data/datasets/fineweb10B_sp1024`
- Tokenizer: `data/tokenizers/fineweb_1024_bpe.model`

## Status

This README is filled in incrementally as the pod runs complete. Final metrics, logs, and the exact command lines are added after each trial finishes.
