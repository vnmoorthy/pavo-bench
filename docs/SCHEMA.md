# Camera-Ready Data and Result Schemas

This document describes the paper-aligned files identified in
`docs/ARTIFACT_STATUS.md`. Historical files retain their original schemas and
are not remapped to camera-ready claims.

## Benchmark records

`tier3_50k_train.jsonl` and `tier3_50k_test.jsonl` contain one JSON object per
line:

```jsonc
{
  "index": 32453,
  "complexity": 1,
  "snr_db": 20.58,
  "noise_type": "white",
  "cpu_util": 0.84,
  "battery": 0.89,
  "rtt_ms": 94.27,
  "ctx_tokens": 250,
  "source": "synthetic",
  "audio_idx": 113,
  "user_input": "...",
  "reference_response": "..."
}
```

The train and test files contain 40,000 and 10,000 rows. The unmodified bytes
contain 1,743 literal `[TIMEOUT]` reference responses and two generation
phases. `data/DATASET_AUDIT.json` records exact hashes, split integrity, and
realized distributions. `tier3_50k_summary.json` is generator-supplied resume
metadata; its `error_count` describes the resumed 20K phase, not all 50K rows.

## Statistical simulation

`tier1_statistical_results.json` contains five paired catalog-simulation
replications. Metric objects contain `values`, `mean`, `std`, and 95% confidence
limits. `comparisons.pavo_vs_cloud_latency` contains the paired t-test and
Wilcoxon values. The p-value here applies to mean latency, not the descriptive
P95 result.

## Mixed measured/simulated E2E summary

`tier2_e2e_results.json` contains `cloud_premium`, `ondevice_fast`,
`hybrid_balanced`, and `pavo_adaptive`. Fixed configurations contain latency
summaries and stage summaries. The adaptive record adds routing counts and
percentages. See `docs/RESULT_PROVENANCE.md`: the adaptive row is heuristic
aggregate sampling, not released-controller checkpoint replay.

## Three-model H100 coupling aggregate

`experiments/outputs_new/coupling_3models.json` is keyed by LLM family and then
nominal injected-WER level. Each leaf contains:

```jsonc
{
  "exact_match": 0.95,
  "quality_score": 0.876,
  "n": 200
}
```

There are 27 aggregate cells: 3 models × 9 WER levels × `n=200`, reporting
5,400 calls. Individual records and a byte-matching final generator are not
present.

## Preliminary M3 factual-QA pilot

`experiments/outputs_new/coupling_m3_factual_qa.json` is keyed by three model
configurations and eight nominal WER labels. Each condition summarizes the same
30 questions:

```jsonc
{
  "wer_pct": 2,
  "accuracy": 0.6333333333333333,
  "correct": 19,
  "total": 30
}
```

The `threshold` field is the first tested WER label where accuracy fell below
0.70. The recovered generator is
`experiments/original_runs/coupling_m3_factual_qa.py`. Its minimum-one-token
corruption makes the WER labels nominal for these short questions; see the
provenance document.

## Real ASR coupling

`experiments/outputs_new/real_asr_coupling.json` is keyed by ASR–LLM pair:

```jsonc
{
  "asr_word_accuracy_pct": 99.57,
  "bertscore_deberta": 0.527,
  "n_samples": 100
}
```

`asr_word_accuracy_pct` is word accuracy on a 0–100 scale; it is not WER.

## Other paper-aligned summaries

- `tier1_llm_latency_results.json`: per-context latency, TTFT, throughput, and
  output-token summaries.
- `tier2_cross_dataset_results.json`: LibriSpeech and FLEURS ASR summaries.
- `tier2_noise_robustness_results.json`: direct ASR noise/SNR summaries plus a
  synthetic-text-corruption LLM section. Its `error_rate` is an API-failure
  indicator, not a coherence-failure rate.
- `tier3_scaling_results.json`: model latency by query-complexity band.
- `experiments/outputs_new/ablation_deberta.json`: final DeBERTa aggregate.
- `data/supervised_baseline_results.json`: classifier/oracle comparison.
- `experiments/outputs_new/e2e_librispeech.json`: additional direct-inference
  summary.
- `experiments/outputs/training_log.json`: PPO trajectory and runtime metadata.

## Model files

`experiments/outputs/meta_controller.safetensors` and
`experiments/outputs/meta_controller_best.safetensors` contain policy and value
network weights. Each has 85,041 scalar parameters. Checksums are recorded in
`experiments/outputs/CHECKSUMS.txt` and `ARTIFACTS.sha256`.
