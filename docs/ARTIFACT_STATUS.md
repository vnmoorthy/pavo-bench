# Artifact Status: Camera-Ready vs Historical

The repository preserves earlier public files and Git history. Presence in the
tree therefore does not automatically make a file camera-ready evidence. This
catalog is the authoritative status map.

## Camera-ready artifact set

| Category | Authoritative files |
|---|---|
| Benchmark data | `tier3_50k_train.jsonl`, `tier3_50k_test.jsonl`, `tier3_50k_summary.json`, `data/DATASET_AUDIT.json` |
| Controller | `experiments/outputs/meta_controller*.safetensors`, `experiments/outputs/training_log.json`, `experiments/exp3_train_ppo.py`, `docs/CONTROLLER.md` |
| Coupling | `experiments/outputs_new/coupling_3models.json`, `experiments/outputs_new/coupling_m3_factual_qa.json`, `experiments/outputs_new/real_asr_coupling.json` |
| Direct/mixed E2E | `tier2_e2e_results.json`, `experiments/outputs_new/e2e_librispeech.json` |
| Other summaries | `tier1_llm_latency_results.json`, `tier1_statistical_results.json`, `tier2_cross_dataset_results.json`, `tier2_noise_robustness_results.json`, `tier3_scaling_results.json`, `experiments/outputs_new/ablation_deberta.json`, `data/supervised_baseline_results.json` |
| Provenance | `docs/RESULT_PROVENANCE.md`, `experiments/original_runs/` |

The exact scope and limitations of every claimed mapping are in
`docs/RESULT_PROVENANCE.md`. In particular, not every paper-only aggregate has
a standalone raw trace in this repository.

## Historical or superseded files

| Path | Status and reason |
|---|---|
| `tier1_coupling_results.json` | Historical n=10 pilot; its own conclusion says no critical threshold was found. Do not use for the final coupling figure. |
| `experiments/outputs/coupling_results_200.json` | Historical two-model run; superseded by the three-model H100 aggregate. |
| `experiments/outputs/ablation_results_real.json` | Historical A100-era heuristic ablation. |
| `experiments/outputs/ablation_bertscore.json` | Historical A100-era output; not the final DeBERTa summary. |
| `component_ablation_results.json` | Historical simulator-scale summary; not evidence for the manuscript's final 34%/71% aggregate claims. |
| `tier3_50k_checkpoint.jsonl` | Interrupted 67-record checkpoint using an earlier schema. |
| `experiments/exp4_fix.py`, `experiments/run_remaining.sh` | Historical development utilities. |
| `experiments/*.log` | Historical run logs; not claim-bearing result artifacts. |
| `experiments/outputs/meta_controller*.pt` | Legacy pickle checkpoints retained for compatibility. Prefer safetensors. |

No historical file is silently deleted or rewritten by the camera-ready sync.
New consumers should start from the camera-ready artifact set above.
