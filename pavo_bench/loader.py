"""Load the released MetaController checkpoint.

Prefers the pickle-free safetensors checkpoints that ship in this repository
and on the PAVO-Bench Hugging Face dataset. Legacy PyTorch checkpoints remain a
local fallback for compatibility.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from .model import MetaController

_DEFAULT_CANDIDATES = (
    "experiments/outputs/meta_controller_best.safetensors",
    "experiments/outputs/meta_controller.safetensors",
    "experiments/outputs/meta_controller_best.pt",
    "experiments/outputs/meta_controller.pt",
)

_HF_CANDIDATES = (
    "models/meta_controller_best.safetensors",
    "models/meta_controller.safetensors",
    "experiments/outputs/meta_controller_best.safetensors",
    "experiments/outputs/meta_controller.safetensors",
)


def _find_checkpoint(repo_root: Path | None, allow_download: bool = True) -> Path:
    search: list[Path] = []
    if repo_root is not None:
        for rel in _DEFAULT_CANDIDATES:
            search.append(Path(repo_root) / rel)
    if env := os.environ.get("PAVO_BENCH_ROOT"):
        for rel in _DEFAULT_CANDIDATES:
            search.append(Path(env) / rel)
    for rel in _DEFAULT_CANDIDATES:
        search.append(Path.cwd() / rel)
        search.append(Path.cwd().parent / rel)

    for p in search:
        if p.exists():
            return p

    if allow_download:
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
        except ImportError as exc:
            raise FileNotFoundError(
                "Could not find a released checkpoint locally and "
                "huggingface_hub is not installed. Pass repo_root= or set "
                "PAVO_BENCH_ROOT to a repository checkout."
            ) from exc

        errors = []
        for filename in _HF_CANDIDATES:
            try:
                return Path(hf_hub_download(
                    repo_id="vnmoorthy/pavo-bench",
                    filename=filename,
                    repo_type="dataset",
                ))
            except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as exc:
                errors.append(f"{filename}: {exc}")
        raise FileNotFoundError(
            "Could not download a released safetensors checkpoint from "
            "vnmoorthy/pavo-bench. Tried: " + "; ".join(errors)
        )

    raise FileNotFoundError(
        "Could not find a released checkpoint. Pass repo_root=, set "
        "PAVO_BENCH_ROOT, or allow the Hugging Face download fallback."
    )


def load_pretrained(
    repo_root: str | Path | None = None,
    device: str = "cpu",
    allow_download: bool = True,
) -> tuple[MetaController, dict]:
    """Load the released meta-controller.

    Returns:
        (model, info) where `info` is a dict with keys that may include
        'architecture', 'n_params', 'training_steps', etc. — whatever the
        released checkpoint bundles.
    """
    ckpt_path = _find_checkpoint(
        Path(repo_root) if repo_root else None,
        allow_download=allow_download,
    )

    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        blob = load_file(str(ckpt_path), device=device)
        checkpoint_format = "safetensors"
    else:
        blob = torch.load(ckpt_path, map_location=device, weights_only=True)
        checkpoint_format = "pytorch-legacy"

    if isinstance(blob, dict) and "model_state_dict" in blob:
        arch = blob.get("architecture", {}) or {}
        model = MetaController(
            state_dim=int(arch.get("state_dim", 12)),
            hidden=int(arch.get("hidden", 256)),
            n_profiles=int(arch.get("n_profiles", 48)),
        )
        model.load_state_dict(blob["model_state_dict"])
        info = {k: v for k, v in blob.items() if k != "model_state_dict"}
    else:
        model = MetaController()
        model.load_state_dict(blob)
        info = {"source": str(ckpt_path), "format": checkpoint_format}

    model.to(device).eval()
    info["checkpoint_path"] = str(ckpt_path)
    info.setdefault("format", checkpoint_format)
    info["n_params_loaded"] = model.count_params()
    return model, info
