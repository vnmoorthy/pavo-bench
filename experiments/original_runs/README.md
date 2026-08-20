# Original-run source: read before use

These files are retained unmodified for provenance. They are environment-bound
records of original experiment entry points, not supported one-command
reproduction scripts. Do not run them as-is on a shared or production system:
they use local model services, assume machine-specific paths, and may install
packages or require large model downloads. Their additional runtime dependencies
are intentionally not covered by the repository's default `requirements.txt`.

## `tier2_end_to_end.py`

This is the source for the method serialized in
`tier2_e2e_results.json`:

- ASR and LLM stages for three fixed configurations are directly executed on
  up to 200 audio files.
- TTS is not executed; fixed 200/350 ms values are added.
- The `pavo_adaptive` row does not load the released controller checkpoint. It
  draws seeded synthetic SNR, battery, and RTT values, applies a hand-written
  routing rule, and samples latency from the fixed configurations' aggregate
  means and standard deviations.

The result file is retained because it is the source of the manuscript's P95
comparison, but it should be interpreted as mixed measurement and simulation.

## `tier1_statistical_significance.py`

This is the original method/entry point underlying
`tier1_statistical_results.json`. It generates five sets of 1,000 synthetic
demand vectors and exhaustively selects the minimum-cost configuration from a
hard-coded component catalog. It does not load raw E2E measurements or the PPO
checkpoint. The original saved output was truncated; the included compact JSON
is a repaired completion retaining the PAVO-vs-cloud latency comparison, so this
script is not a byte-for-byte generator for that compact file.

## `tier3_pavo_bench_50k.py` and `tier3_50k_resume.py`

These are the two environment-bound dataset generators. The released bytes show
that indices 0–29,999 and 30,000–49,999 came from materially different phases.
All released records have `source="synthetic"`; no audio is bundled. The first
phase contains all 1,743 timeout references, while the resumed 20K phase reports
zero generation errors. Exact realized statistics are in
`data/DATASET_AUDIT.json`.

## `coupling_m3_factual_qa.py`

This is the recovered source for the preliminary M3 factual-QA summary. Its WER
labels are nominal. Because the code enforces at least one selected token and
the questions are short, all nonzero labeled levels select the same token per
question; five selected words of length two or less are left unchanged. See
`docs/RESULT_PROVENANCE.md` for the resulting scope limitation.
