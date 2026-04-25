# Result-file schemas

Each committed JSON in this repo is one experimental artifact. This file
documents the shape of every one so consumers don't have to grep.

## Tier 1 — components

### `tier1_coupling_results.json`
Coupling cliff calibration. Per-WER-level mean / std quality scores.

```jsonc
{
  "experiment": "Coupling Constraint Validation",
  "model": "llama3.1:8b",
  "n_queries_per_wer": 200,
  "wer_levels_tested": [0, 1, 2, 3, 5, 8, 10, 15, 20],
  "results": {
    "wer_<pct>": {
      "wer_pct": <int>,
      "mean_quality": <float, 0..1>,
      "std_quality":  <float>,
      "n_queries":    <int>,
      "quality_scores": [<float>, ...],
      "degradation_from_clean": <float>,
      "degradation_pct":        <float>
    }
  }
}
```

### `tier1_llm_latency_results.json`
LLM latency profile across short/medium/long contexts.

```jsonc
{
  "<model>_<short|medium|long>": {
    "model": "llama3.1:8b",
    "context": "short|medium|long",
    "num_predict": <int>,
    "n_measurements": <int>,
    "total_latency_ms":     {"mean":..., "std":..., "median":..., "p95":..., "p99":..., "min":..., "max":...},
    "time_to_first_token_ms": {...},
    "tokens_per_second":      {...},
    "output_tokens":          {...}
  }
}
```

### `tier1_statistical_results.json`
Statistical reproducibility across 5 trials of 1,000 turns each.

## Tier 2 — integration

### `tier2_e2e_results.json`
End-to-end pipeline measurements for four configurations.

```jsonc
{
  "cloud_premium":   {"config": {...}, "n_samples": 200, "e2e_latency_ms": {...},
                      "asr_latency_ms": {...}, "llm_latency_ms": {...},
                      "sample_asr_outputs": [...], "sample_llm_responses": [...]},
  "ondevice_fast":   {...},
  "hybrid_balanced": {...},
  "pavo_adaptive":   {"description":..., "n_samples":..., "e2e_latency_ms":...,
                      "config_distribution": {...}, "config_distribution_pct": {...}}
}
```

### `tier2_cross_dataset_results.json`
ASR cross-dataset comparison on LibriSpeech and FLEURS.

### `tier2_noise_robustness_results.json`
ASR WER under white-noise injection at SNR 5–30 dB.

## Tier 3 — scale

### `tier3_50k_train.jsonl` / `tier3_50k_test.jsonl`
The PAVO-Bench dataset. One JSON object per line:

```jsonc
{
  "index":           <int>,           // turn ID
  "complexity":      <int 1..5>,      // turn complexity
  "snr_db":          <float>,         // ambient SNR
  "noise_type":      <str>,           // "babble" | "traffic" | ...
  "cpu_util":        <float 0..1>,    // device CPU
  "battery":         <float 0..1>,    // device battery
  "rtt_ms":          <float>,         // network RTT
  "ctx_tokens":      <int>,           // dialogue context depth
  "user_input":      <str>,
  "reference_response": <str>,
  "source":          <str>,
  "audio_idx":       <int|null>
}
```

### `tier3_50k_summary.json`
Aggregate statistics for the dataset.

### `tier3_scaling_results.json`
Per-model latency benchmarks for simple / medium / complex queries.

## Component analysis

### `component_ablation_results.json`
PAVO-Full vs ablated configurations on the 50K test split.

```jsonc
{
  "<config_name>": {
    "latency_ms":  {"mean": <float>, "std": <float>},
    "quality":     {"mean": <float>, "std": <float>},
    "cost_usd":    {"mean": <float>, "std": <float>},
    "energy_mj":   {"mean": <float>, "std": <float>},
    "coupling_violations": {"mean": <int>, "per_1000": <float>},
    "infeasible_pct": <float>
  }
}
```
