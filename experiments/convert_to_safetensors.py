#!/usr/bin/env python3
"""
Convert pickle-format PyTorch checkpoints to safetensors.

Pickle .pt files can execute arbitrary code at load time. Safetensors cannot.
Run this once locally, then re-upload the .safetensors files to HuggingFace
in place of the .pt files.

Usage:
    pip install torch safetensors
    python convert_to_safetensors.py

Verifies SHA256 sums against CHECKSUMS.txt before converting.
"""

import hashlib
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
OUTPUTS = os.path.join(PROJECT_ROOT, "experiments", "outputs")
OUTPUTS_REL = "experiments/outputs"  # path used in CHECKSUMS.txt (project-relative)
CHECKPOINTS = ["meta_controller.pt", "meta_controller_best.pt"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_expected_sums(checksums_path):
    sums = {}
    if not os.path.exists(checksums_path):
        return sums
    with open(checksums_path) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                sums[os.path.basename(parts[1])] = parts[0]
    return sums


def _unwrap_state_dict(state):
    """Handle common training-checkpoint shapes like {"model": state_dict, ...}."""
    import torch
    if isinstance(state, dict) and not all(isinstance(v, torch.Tensor) for v in state.values()):
        for key in ("model", "state_dict", "model_state_dict"):
            inner = state.get(key)
            if isinstance(inner, dict) and all(isinstance(v, torch.Tensor) for v in inner.values()):
                dropped = sorted(k for k in state.keys() if k != key)
                if dropped:
                    print(f"  unwrapped '{key}'; dropping non-weight keys: {dropped}")
                    print("  (safetensors holds inference weights only — cannot resume training from this file)")
                else:
                    print(f"  unwrapped nested checkpoint: top-level['{key}']")
                return inner
    return state


def main():
    import torch
    from safetensors.torch import save_file

    checksums_path = os.path.join(OUTPUTS, "CHECKSUMS.txt")
    expected = load_expected_sums(checksums_path)
    new_sums = []

    for ckpt in CHECKPOINTS:
        src = os.path.join(OUTPUTS, ckpt)
        if not os.path.exists(src):
            print(f"SKIP {ckpt}: not found at {src}")
            continue

        actual = sha256(src)
        want = expected.get(ckpt)
        if want and want != actual:
            print(f"FAIL {ckpt}: sha256 mismatch")
            print(f"  expected {want}")
            print(f"  got      {actual}")
            sys.exit(1)
        if want:
            print(f"OK   {ckpt}: sha256 verified")

        state = torch.load(src, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            print(f"FAIL {ckpt}: not a state_dict (got {type(state).__name__})")
            print("  Re-save with torch.save(model.state_dict(), path) before conversion.")
            sys.exit(1)

        state = _unwrap_state_dict(state)
        non_tensor = [k for k, v in state.items() if not isinstance(v, torch.Tensor)]
        if non_tensor:
            print(f"FAIL {ckpt}: non-tensor values at keys: {non_tensor[:5]}")
            print("  Strip optimizer/scheduler/step state before conversion.")
            sys.exit(1)

        dst = os.path.join(OUTPUTS, ckpt.replace(".pt", ".safetensors"))
        save_file(state, dst)
        new_sums.append((sha256(dst), os.path.basename(dst)))
        print(f"WROTE {dst} ({len(state)} tensors)")

    if new_sums:
        existing = ""
        if os.path.exists(checksums_path):
            with open(checksums_path) as fh:
                existing = fh.read()
        new_basenames = {name for _, name in new_sums}
        with open(checksums_path, "w") as f:
            for line in existing.splitlines():
                tail = line.rsplit(None, 1)[-1] if line.strip() else ""
                if os.path.basename(tail) in new_basenames:
                    continue
                f.write(line + "\n")
            for digest, name in new_sums:
                f.write(f"{digest}  {OUTPUTS_REL}/{name}\n")
        print(f"\nUpdated {checksums_path} with {len(new_sums)} safetensors entries.")

    print("Done. Upload the .safetensors files to HuggingFace and point the dataset README at them.")


if __name__ == "__main__":
    main()
