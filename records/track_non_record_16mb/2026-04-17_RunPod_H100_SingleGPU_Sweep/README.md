# RunPod H100 Single-GPU Sweep

This folder captures an initial non-record participation pass run on a single `NVIDIA H100 80GB HBM3` RunPod pod.

The goal of this sweep was practical orientation rather than leaderboard chasing:

- verify the published `sp1024` dataset path on RunPod
- reproduce a baseline on `1xH100`
- test a small set of low-risk hyperparameter changes drawn from earlier public records
- keep the artifact under the 16MB cap and document exact commands, logs, and outcomes

## Outcome

The best run in this sweep was `lower_lr_sp1024_single_h100`, which improved the post-quant score from `1.33586092` to `1.33140622 val_bpb` while staying comfortably under the 16MB artifact cap.

This is not a record attempt. It is a clean, reproducible single-GPU reference point for future H100 sweeps on the user's RunPod pod.

## Compatibility Note

The stock attention path in the repo assumed a PyTorch build that accepts `enable_gqa=` in `torch.nn.functional.scaled_dot_product_attention`. The RunPod pod had `PyTorch 2.4.1+cu124`, where that keyword is not available.

To make the repo run on the pod without changing the model shape, the code snapshot in this folder adds a small compatibility fallback:

- detect whether `enable_gqa` is supported
- manually expand KV heads when it is not
- preserve the same GQA model definition and hyperparameter surface

## Environment

- Pod provider: RunPod
- GPU: `1xH100 80GB HBM3`
- Python: `3.11.10`
- PyTorch: `2.4.1+cu124`
- Repo: `meett07/meett07-parameter-golf`
- Dataset: `data/datasets/fineweb10B_sp1024`
- Tokenizer: `data/tokenizers/fineweb_1024_bpe.model`
- Wallclock cap: `600s`

## Command

The full sweep was launched with:

```bash
bash records/track_non_record_16mb/2026-04-17_RunPod_H100_SingleGPU_Sweep/run_trials.sh
```

That script runs the following trials sequentially:

1. `baseline_sp1024_single_h100`
2. `lower_lr_sp1024_single_h100`
3. `seq4096_lowerlr_single_h100`

## Results

| Trial | Key overrides | Stop step | Pre-quant val_bpb | Post-quant val_bpb | Total bytes | Result |
|---|---|---:|---:|---:|---:|---|
| `baseline_sp1024_single_h100` | stock `sp1024` config | 1288 | 1.3346 | 1.33586092 | 13,419,241 | baseline |
| `lower_lr_sp1024_single_h100` | `MATRIX_LR=0.02`, `SCALAR_LR=0.02`, `TIED_EMBED_LR=0.03` | 1306 | 1.3286 | **1.33140622** | 11,913,909 | best |
| `seq4096_lowerlr_single_h100` | lower LR + `TRAIN_SEQ_LEN=4096`, `TRAIN_BATCH_TOKENS=393216`, Muon warmup changes | 1231 | 1.3456 | 1.35902715 | 10,165,286 | worse |

## Takeaways

- Reducing the learning rates improved both pre-quant and post-quant validation on the single-H100 10-minute budget.
- The longer-context `4096` sequence experiment fit easily in memory, but it completed fewer optimizer steps inside the same wallclock cap and regressed.
- The winning run also compressed materially better than baseline: `11,865,688` model bytes vs `13,371,020`.

## Best Run

`lower_lr_sp1024_single_h100`

- `step_stop`: `1306`
- `pre_quant_val_loss`: `2.2434`
- `pre_quant_val_bpb`: `1.3286`
- `post_quant_val_loss`: `2.24802363`
- `post_quant_val_bpb`: `1.33140622`
- `bytes_model_int8_zlib`: `11865688`
- `bytes_total`: `11913909`
- `peak_memory_allocated`: `13065 MiB`

## Included Files

- `run_trials.sh` — exact sweep launcher used on the pod
- `train_gpt.py` — code snapshot that ran on RunPod, including the SDPA/GQA compatibility fix
- `results.tsv` — compact results table for all three trials
- `baseline_sp1024_single_h100.log` — compact run log for trial 1
- `lower_lr_sp1024_single_h100.log` — compact run log for the best trial
- `seq4096_lowerlr_single_h100.log` — compact run log for trial 3
- `submission.json` — metadata keyed to the best run in this sweep
