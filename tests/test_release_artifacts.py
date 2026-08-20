import hashlib
import json
from pathlib import Path

import pytest

from pavo_bench import PretrainedPAVORouter, load_dataset, load_pretrained

REPO = Path(__file__).resolve().parents[1]


def _json(path: str):
    with (REPO / path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_released_dataset_hashes_and_counts():
    audit = _json("data/DATASET_AUDIT.json")
    for filename, expected in audit["file_sha256"].items():
        assert _sha256(REPO / filename) == expected

    assert sum(1 for _ in (REPO / "tier3_50k_train.jsonl").open()) == 40_000
    assert sum(1 for _ in (REPO / "tier3_50k_test.jsonl").open()) == 10_000
    assert audit["record_counts"]["total"] == 50_000
    assert audit["reference_response_audit"]["literal_timeout_total"] == 1_743


def test_camera_ready_summary_arithmetic_and_statistics_semantics():
    e2e = _json("tier2_e2e_results.json")
    cloud_p95 = e2e["cloud_premium"]["e2e_latency_ms"]["p95"]
    pavo_p95 = e2e["pavo_adaptive"]["e2e_latency_ms"]["p95"]
    assert 100 * (cloud_p95 - pavo_p95) / cloud_p95 == pytest.approx(10.2736, abs=0.001)

    stats = _json("tier1_statistical_results.json")["comparisons"][
        "pavo_vs_cloud_latency"
    ]
    assert stats["p_value_ttest"] == 0.000002
    assert stats["p_value_wilcoxon"] == 0.0625


def test_coupling_aggregate_shapes_and_real_asr_field_name():
    h100 = _json("experiments/outputs_new/coupling_3models.json")
    assert set(h100) == {"llama3.1:8b", "gemma2:2b", "mistral:7b"}
    assert sum(cell["n"] for model in h100.values() for cell in model.values()) == 5_400

    m3 = _json("experiments/outputs_new/coupling_m3_factual_qa.json")
    assert len(m3) == 3
    for model in m3.values():
        cells = [value for key, value in model.items() if key.startswith("wer_")]
        assert len(cells) == 8
        assert {cell["total"] for cell in cells} == {30}

    real_asr = _json("experiments/outputs_new/real_asr_coupling.json")
    assert all("asr_word_accuracy_pct" in row for row in real_asr.values())
    assert all("asr_wer_pct" not in row for row in real_asr.values())


def test_safetensors_checkpoint_loads_all_85041_parameters():
    model, info = load_pretrained(repo_root=REPO, allow_download=False)
    assert info["format"] == "safetensors"
    assert info["checkpoint_path"].endswith("meta_controller_best.safetensors")
    assert model.count_params() == 85_041
    assert info["n_params_loaded"] == 85_041


def test_unresolved_checkpoint_actions_fail_closed():
    turn = load_dataset(split="test", repo_root=REPO, limit=1)[0]
    controller = PretrainedPAVORouter.from_released(repo_root=str(REPO))
    assert tuple(controller.action_logits(turn).shape) == (48,)
    with pytest.raises(RuntimeError, match="does not define a mapping"):
        controller.route(turn)


def test_historical_artifacts_are_explicitly_cataloged():
    catalog = (REPO / "docs/ARTIFACT_STATUS.md").read_text(encoding="utf-8")
    for path in (
        "tier1_coupling_results.json",
        "experiments/outputs/coupling_results_200.json",
        "experiments/outputs/ablation_results_real.json",
        "experiments/outputs/ablation_bertscore.json",
        "component_ablation_results.json",
        "tier3_50k_checkpoint.jsonl",
    ):
        assert f"`{path}`" in catalog
