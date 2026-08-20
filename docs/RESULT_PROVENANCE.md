# Result Provenance and Scope

This map avoids the stale claim-to-file mappings that appeared in earlier
review bundles. It distinguishes values directly recoverable from an included
artifact from aggregate values reported only in the camera-ready manuscript.

## Directly supported by included files

| Manuscript result | Included artifact | Machine-readable evidence |
|---|---|---|
| 10.3% P95 E2E reduction | `tier2_e2e_results.json` | Cloud P95 `1619.78` ms; PAVO P95 `1453.37` ms; fixed configurations use measured ASR/LLM plus fixed simulated TTS, while the PAVO row is heuristic aggregate sampling, not a controller-checkpoint replay |
| Catalog-simulation mean-latency paired test | `tier1_statistical_results.json` | Five 1,000-turn synthetic trials; PAVO mean `2276.9561`; cloud mean `2670.6966`; paired t-test `p=0.000002`; Wilcoxon `p=0.0625` |
| H100 coupling aggregate | `experiments/outputs_new/coupling_3models.json` | 27 aggregate cells reporting 3 models × 9 WER levels × `n=200`, or 5,400 calls; individual call records and the final generator are not included |
| Preliminary Apple M3 factual-QA pilot | `experiments/outputs_new/coupling_m3_factual_qa.json` | 3 configurations × 8 nominal WER labels, each evaluated on the same 30-question set; recovered generator in `experiments/original_runs/coupling_m3_factual_qa.py` |
| 50K benchmark split | `tier3_50k_{train,test}.jsonl` | 40,000 train and 10,000 test records |
| Released-dataset integrity facts | `data/DATASET_AUDIT.json` | Exact hashes, 1,743 timeout references, realized distributions, and two-phase generation characteristics |
| 85,041 controller parameters | `experiments/outputs/*.safetensors` | Tensor shapes sum to 85,041; architecture in `docs/CONTROLLER.md` |
| PPO trajectory and runtime metadata | `experiments/outputs/training_log.json` | 196 logged batches, H100 metadata, committed wall-clock duration |
| Real ASR-pair measurements | `experiments/outputs_new/real_asr_coupling.json` | Six ASR–LLM pairs; field is `asr_word_accuracy_pct` |
| Supervised routing baselines | `data/supervised_baseline_results.json` | LR, RF, XGBoost, MLP, and oracle summary |
| Cross-dataset/noise/scaling summaries | corresponding `tier2_*` and `tier3_scaling_results.json` files | Values as serialized in each artifact |

The `p=2×10^-6` result concerns the mean-latency paired test. It is not a
significance test of the descriptive P95 difference.

The retained source `experiments/original_runs/tier1_statistical_significance.py`
generates synthetic demand vectors and exhaustively chooses the lowest-cost
entry from a hard-coded component catalog. It does not bootstrap raw measured
E2E samples and does not load the PPO checkpoint. The p-value is reproducible
for that catalog-based simulation, not evidence from direct voice-pipeline
traces.
The original saved JSON was truncated mid-write. The included compact repaired
copy retains the five per-seed metric values and the PAVO-vs-cloud latency
comparison only; the unmodified entry point would emit additional comparisons.
Recomputing from those five paired values gives `t=-43.61518896`, an unrounded
two-sided `p=1.652265e-6` (serialized as `0.000002`), and Wilcoxon `p=0.0625`.

The exact original entry point for the E2E summary is retained at
`experiments/original_runs/tier2_end_to_end.py`. It directly executes ASR and LLM for
the three fixed configurations, adds fixed TTS latency, and constructs the
`pavo_adaptive` row from seeded synthetic demand variables, a hand-written
routing rule, and samples from aggregate latency statistics. Accordingly, the
included P95 arithmetic is directly recoverable, but the PAVO row must not be
described as inference from the released PPO checkpoint.

The M3 pilot's WER labels are nominal. Its minimum-one-token corruption rule
produces the same corrupted prompt for every nonzero labeled level on the short
questions used. The saved aggregate therefore supports the displayed counts,
but does not independently establish realized 2%-increment WER. It also does
not contain the paired outcomes or calculation needed to reproduce the
manuscript's Fisher-exact `p=0.038`. Finally, the paper's `n=5,430` combines
5,400 H100 calls with 30 distinct M3 questions; it should not be interpreted as
5,430 homogeneous raw records in these files.

The two ASR result files use different, undocumented sample/protocol slices.
`real_asr_coupling.json` reports word accuracy of 99.57% and 98.82% on 100
samples (approximately 0.43% and 1.18% word error), whereas
`tier2_cross_dataset_results.json` reports 5.77% and 18.54% WER on 200
LibriSpeech samples for the same Whisper model families. No raw trace or
generator for the former is present, so the camera-ready artifact set does not treat the two as
interchangeable corroboration.

The `routing_recommendation` fields in the cross-dataset and noise JSONs are
generator-defined labels derived from the reported WER values; they are not
actions emitted by the released controller. Likewise,
`tier3_scaling_results.json` is a compact aggregate for which no byte-matching
generator was found, so it is preserved as a result summary rather than claimed
as independently regenerated here. `ablation_deberta.json` is likewise retained
as the final paper-aligned aggregate, but no matching raw trace or generator was
found.

In the noise summary, ASR is directly evaluated for the noise conditions, but
`llm_quality_under_noise` is produced by synthetically corrupting reference text
at the measured WER rather than passing the recorded ASR hypotheses to the LLM.
Its `error_rate` counts API responses containing `[ERROR]`; a value of 0.0 means
no recorded API-call failures, not zero semantic or coherence failures.

The released PPO source and safetensors define 48 analytic action-factor logits.
No released k-means artifact or mapping from those indices to concrete
ASR/LLM/TTS model, precision, and placement tuples was found. The checkpoints
therefore reproduce network weights and logits, but not a deployment action
resolver. The implemented entropy coefficient is fixed at `0.01`.

## Paper-only aggregate statements

No separate raw trace or standalone machine-readable source for the following
aggregate statements was present in the release workspace audited for the
camera-ready supplement:

- 34% median-latency reduction and 71% energy reduction on the 50K simulation.
- 7.1% to 0.9% coherence-failure reduction and the associated +110 ms cost.
- The M3 Fisher-exact `p=0.038`.
- Inter-annotator agreement `κ=0.81`.
- The full acoustic-feature ablation aggregates.

These values remain reported in the camera-ready manuscript. This repository does
not substitute unrelated legacy JSON files as evidence for them and does not
claim independent machine-readable reproduction of those paper-only aggregates.
The 34%/71% and 7.9× ratios are arithmetically consistent with the manuscript's
displayed table values; the underlying routing traces and annotation labels are
not present.

## Superseded artifacts retained for history

Git history and existing public links are preserved, so the following earlier
development/review artifacts remain in the repository. They are explicitly
excluded from the camera-ready evidence map:

- `tier1_coupling_results.json`: 10 queries per WER; concludes that no critical
  threshold was found.
- `experiments/outputs/coupling_results_200.json`: older two-model coupling run,
  superseded by `experiments/outputs_new/coupling_3models.json`.
- `experiments/outputs/ablation_results_real.json` and
  `experiments/outputs/ablation_bertscore.json`: A100-era ablation outputs,
  superseded for the final presentation by
  `experiments/outputs_new/ablation_deberta.json`.
- `tier3_50k_checkpoint.jsonl`: interrupted 67-record generation checkpoint
  with the pre-release schema.
- `component_ablation_results.json`: simulator-scale component summary that
  does not source the final 34%/71% headline values.
- `experiments/exp4_fix.py`, `experiments/run_remaining.sh`, and the committed
  experiment logs: historical development utilities and logs.

See `docs/ARTIFACT_STATUS.md` before mapping a manuscript claim to a file.
