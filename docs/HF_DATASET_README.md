<!-- Copy the entire content of this file into the README on
     https://huggingface.co/datasets/vnmoorthy/pavo-bench (Edit pencil, top right).
     Preserve the YAML frontmatter; HF requires it for dataset configs. -->

---
license: cc-by-4.0
task_categories:
  - automatic-speech-recognition
  - text-generation
  - text-to-speech
language:
  - en
tags:
  - pavo
  - benchmark
  - asr
  - llm
  - tts
  - pipeline-routing
  - voice-assistant
  - latency
  - quality
  - cost
  - energy
pretty_name: PAVO-Bench
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: train
        path: tier3_50k_train.jsonl
      - split: test
        path: tier3_50k_test.jsonl
---

# PAVO-Bench: 50K-Turn Benchmark for ASR-LLM-TTS Pipeline Routing

**Code:** [github.com/vnmoorthy/pavo-bench](https://github.com/vnmoorthy/pavo-bench) · **Paper:** [TMLR 2026 (accepted)](https://openreview.net/forum?id=zrneoIxlFx) · **Authors:** NarasingaMoorthy VeiluKanthaPerumal (UPenn), Mohammed Imthathullah (Google)

```bash
pip install git+https://github.com/vnmoorthy/pavo-bench.git
```

## Headline results (vs fixed-cloud baseline, 50,000 voice turns)

| Metric | Result | Significance |
|---|---|---|
| P95 end-to-end latency (H100, LibriSpeech) | **−10.3%** (−167 ms) | — |
| Median latency | **−34%** | |
| Energy per turn | **−71%** | |
| Coherence-failure rate | **7.1% → 0.9%** (7.9× reduction) | hard-constraint masking, +110 ms median cost |
| Meta-controller size | 85,041 parameters | — |
| Meta-controller training | 106 seconds on H100 | — |

The `p = 2×10⁻⁶` result applies to the paired test of **mean** end-to-end latency (2,277 vs 2,671 ms over five bootstrap replications), not the descriptive P95 comparison; the paired Wilcoxon test on the same five replications gives `p = 0.0625`.

The empirical contribution is a two-regime coupling structure (sharp factual-accuracy cliff + gradual semantic degradation) characterized over **n = 5,430 measurements** across two hardware platforms (H100, Apple M3) and three LLM families (Llama 3.1 8B, Mistral 7B, Gemma2 2B).

## Description

PAVO-Bench evaluates **ASR-LLM-TTS voice pipeline routing** decisions. It provides 50,000 turns of benchmark data designed to measure how well different pipeline configurations balance **latency**, **quality**, **cost**, and **energy** when routing spoken-language queries through cascaded ASR, LLM, and TTS components.

The benchmark is organized into three tiers plus component-level ablation. The 50K routing evaluation is simulated using measured H100/M3 latencies and published benchmark values for unavailable configurations; the repository also includes direct-inference H100 results.

## Dataset Files

The Hugging Face `default` config exposes the 40K/10K JSONL files as standard `train` and `test` splits. The heterogeneous result-summary JSON files below remain directly downloadable artifacts rather than dataset splits, which keeps the Dataset Viewer schema valid.

### Primary dataset

| File | Split | Rows |
|------|-------|-----:|
| `tier3_50k_train.jsonl` | train | 40,000 |
| `tier3_50k_test.jsonl` | test | 10,000 |

### Tier 1 — Unit-Level Validation

| File | Description |
|------|-------------|
| `tier1_statistical_results.json` | Statistical reproducibility across 5 trials × 1,000 turns each (seeds 42, 123, 456, 789, 1024). |
| `tier1_coupling_results.json` | Coupling-cliff calibration — LLM quality degradation vs ASR word-error rate (WER 0–20%). |
| `tier1_llm_latency_results.json` | Latency profile for `llama3.1:8b` across short / medium / long generation contexts. |

### Tier 2 — Integration-Level Evaluation

| File | Description |
|------|-------------|
| `tier2_e2e_results.json` | End-to-end pipeline measurements (cloud_premium, ondevice_fast, hybrid_balanced, pavo_adaptive) on 200 LibriSpeech samples. |
| `tier2_cross_dataset_results.json` | Cross-dataset ASR (LibriSpeech + FLEURS) for whisper-large-v3 and whisper-tiny. |
| `tier2_noise_robustness_results.json` | ASR robustness at SNR 5–30 dB plus clean baseline. |

### Tier 3 — Scale Evaluation

| File | Description |
|------|-------------|
| `tier3_50k_summary.json` | Summary statistics for the 50K-turn dataset (40K train / 10K test split, complexity 1–5). |
| `tier3_scaling_results.json` | Per-model latency benchmarks for simple / medium / complex queries. |

### Component Analysis

| File | Description |
|------|-------------|
| `component_ablation_results.json` | PAVO-Full vs PAVO-NoCoupling, Always-Cloud, Always-OnDevice, etc. |

## Usage

```python
from huggingface_hub import hf_hub_download
import json

path = hf_hub_download(
    repo_id="vnmoorthy/pavo-bench",
    filename="tier3_50k_summary.json",
    repo_type="dataset",
)
print(json.load(open(path)))
```

Or via the pip package:

```python
from pavo_bench import load_dataset, PretrainedPAVORouter, benchmark_router
turns = load_dataset(split="test")
pavo  = PretrainedPAVORouter.from_released()
print(benchmark_router(pavo, turns))
```

Or directly with `datasets`:

```python
from datasets import load_dataset

dataset = load_dataset("vnmoorthy/pavo-bench")
print(dataset["train"].num_rows, dataset["test"].num_rows)  # 40000 10000
```

## Citation

```bibtex
@article{veilukanthaperumal2026pavo,
  title   = {PAVO: Pipeline-Aware Voice Orchestration with Demand-Conditioned Inference Routing},
  author  = {VeiluKanthaPerumal, NarasingaMoorthy and Imthathullah, Mohammed},
  journal = {Transactions on Machine Learning Research},
  year    = {2026}
}
```

## License

Dataset, results, coupling matrices, and model weights: CC-BY 4.0. Code is MIT licensed in the linked GitHub repository.
