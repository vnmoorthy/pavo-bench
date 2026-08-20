# PAVO-Bench v1.0.1 — Camera-Ready Artifact Clarification

This patch release aligns the public repository with the TMLR camera-ready
artifact audit. It changes documentation and runtime safety/correctness; it does
not change model weights, benchmark rows, paper result JSONs, or manuscript
headline values.

## Changes

- Adds the exact released controller feature, architecture, action-mask, and PPO
  specification.
- Adds a claim-to-artifact provenance map and labels retained historical files
  as superseded instead of silently deleting them.
- Adds the recovered preliminary Apple M3 factual-QA aggregate and original-run
  source, with its nominal-WER limitation documented.
- Adds an independently recomputed audit of the unmodified 40K/10K dataset.
- Makes the runtime prefer pickle-free safetensors and expose the checkpoint's
  raw 48-action logits.
- Makes unresolved checkpoint routing fail clearly: the release does not
  include a defensible 48-index-to-deployment-tuple resolver, so v1.0.1 does
  not fabricate a three-profile collapse.
- Corrects the final coupling figure source, removes the P95/significance
  conflation, and updates H100/A100 wording.
- Adds dependencies, checksums, tests, and gating CI for artifact integrity and
  the previously degenerate 100%-hybrid quickstart path.

## Public supplement asset

Attach the de-anonymized public bundle to the `v1.0.1` GitHub release:

- File: `PAVO_TMLR_camera_ready_supplementary_deanonymized.zip`
- Planned URL after publication: `https://github.com/vnmoorthy/pavo-bench/releases/download/v1.0.1/PAVO_TMLR_camera_ready_supplementary_deanonymized.zip`
- SHA-256: `3cb19f33967213c285941872f6dfd776170ac7a7feecacc6a28cfbeb08ac581a`

Do **not** publish the separate anonymous TMLR-upload ZIP as a public release
asset. See `docs/PUBLIC_SUPPLEMENT.md`.

## Evidence scope

The repository directly supports the artifacts mapped in
`docs/RESULT_PROVENANCE.md`. Several manuscript-level aggregate statements do
not have standalone raw traces in the release workspace; v1.0.1 documents that
fact rather than substituting unrelated historical outputs.

The existing v1.0.0 tag remains an immutable record of the original submission
artifact. Publish these notes under a new v1.0.1 tag; do not move v1.0.0.
