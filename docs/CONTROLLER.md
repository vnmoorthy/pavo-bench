# Released Meta-Controller Specification

This file records the exact preprocessing, network, action construction, mask,
and training settings implemented by `experiments/exp3_train_ppo.py`. It is the
artifact-level specification needed to reproduce the released checkpoint.

## Input vector

The policy consumes 12 `float32` values in this fixed order:

| Position | Category | Released encoding | Source/default |
|---:|---|---|---|
| 1 | Acoustic | `snr / 50` | `snr_db`, falling back to `snr`, then `20.0` |
| 2 | Acoustic | `0.4` | fixed speaking-rate default (`4.0 / 10.0`) |
| 3 | Acoustic | `0.5` | fixed pitch-variance default |
| 4 | Acoustic | `clip(snr, 0, 50) / 50` | WADA-SNR proxy using the same SNR source |
| 5 | Hardware | `cpu_util` | default `0.5` |
| 6 | Hardware | `0.8` | fixed available-RAM fraction |
| 7 | Hardware | `battery` | default `0.8` |
| 8 | Hardware | `0.3` | fixed GPU-utilization value |
| 9 | Network | `rtt_ms / 200` | default `50.0` ms |
| 10 | Network | `0.5` | fixed bandwidth proxy |
| 11 | Context | `complexity / 5` | annotated complexity, default `3` |
| 12 | Context | `ctx_tokens / 2000` | `ctx_tokens`, falling back to `context_tokens`, then `200` |

The final paper separately describes the online feature definitions for a
deployment with live acoustic, hardware, and network telemetry. The table above
is deliberately narrower: it documents the exact released training encoding,
including its constant defaults and duplicated SNR source.

## Network and parameter count

Policy network:

```text
12 → Linear(256) → ReLU → Linear(256) → ReLU → Linear(48)
```

Value network used during PPO training:

```text
12 → Linear(256) → ReLU → Linear(1)
```

The policy contains 81,456 trainable parameters and the value network contains
3,585, for a total of **85,041**. The controller emits 48 categorical logits.
Training samples an action; inference uses the highest-probability feasible
action.

## Analytic action profiles

For action index `k` in `0..47`, the released training environment constructs:

```text
latency_factor = 0.5 + 2.0 * k / 48
energy_factor  = 0.3 + 1.5 * k / 48
quality_factor = 1.0 - 0.3 * k / 48
max_complexity = min(1 + floor(5 * k / 48), 5)
```

The feasibility mask removes an action when the record complexity exceeds
`max_complexity`. When SNR is below 10 dB, it also removes actions with
`quality_factor < 0.8`. If no action remains, index 47 is enabled as the
fallback. The implementation adds `-1e9` to masked logits before constructing
the categorical distribution.

No k-means calibration artifact or concrete mapping from the 48 indices to
ASR/LLM/TTS model, precision, and placement tuples was found in the release
workspace. The checkpoints and source reproduce logits and analytic factors,
but do not by themselves resolve an index into deployable stage selections.

## Reward implemented by the release

The environment uses equal objective weights:

```text
w_latency = w_energy = w_memory = w_quality = 0.25
```

with references `L_ref = 1153 ms`, `E_ref = 6.82`, and `M_ref = 1.0`.
Factual records (complexity 1–2) routed to a profile with
`quality_factor < 0.85` incur a `0.5` violation penalty and multiply quality by
`0.7`. Changing the action relative to the corresponding previous-batch action
incurs a fixed `0.02` penalty.

## PPO training configuration

| Setting | Released value |
|---|---|
| Training data | 40,000 synthetic records; uniform sampling with replacement |
| Target extent | 100,000 records; loop processes 100,352 (196 × 512) |
| Batch size | 512 |
| PPO epochs | 4 full-batch update epochs per collection |
| Optimizer | Adam, learning rate `3e-4`; PyTorch default betas/epsilon |
| LR schedule | Cosine annealing, `T_max = 195`, `eta_min = 0` |
| PPO clipping | `0.2` |
| Discount | `gamma = 0.99` |
| GAE | `lambda = 0.95` |
| KL coefficient | `0.01` |
| Entropy coefficient | `0.01` |
| Value contribution | `0.5 × value_loss`, where value loss already includes `0.5 × MSE` |
| Advantage normalization | Per batch, denominator `std + 1e-8` |
| Gradient clipping | Global norm `0.5` |
| Action-change penalty | Fixed `0.02` |
| Constraint penalty | `0.5` |
| Randomness | NumPy seed 42; PyTorch RNG is not explicitly seeded |
| Episode flags | All collected `done` values are zero |
| Checkpoint rule | Best collected-batch mean reward plus final checkpoint |
| Reported hardware/time | NVIDIA H100 SXM5; 106.398 seconds in the committed log |

The detailed step-by-step training trajectory is in
`experiments/outputs/training_log.json`. Both pickle-free inference-weight
checkpoints are in `experiments/outputs/`.
