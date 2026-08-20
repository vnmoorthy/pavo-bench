#!/usr/bin/env python3
"""
Experiment 5: Upload all artifacts to HuggingFace.
Uploads: 50K dataset, model weights, training logs, all experiment results.
"""

import os


def upload_all(results_dir, token):
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = "vnmoorthy/pavo-bench"

    parent_dir = os.path.dirname(results_dir)
    outputs_dir = os.path.join(results_dir, "outputs")

    print(f"  Uploading to {repo_id}...")

    # 1. Upload the canonical 50K dataset files used by the HF train/test config.
    for fname in ["tier3_50k_train.jsonl", "tier3_50k_test.jsonl", "tier3_50k_summary.json"]:
        fpath = os.path.join(parent_dir, fname)
        if os.path.exists(fpath):
            print(f"    Uploading {fname}...")
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
        else:
            print(f"    WARNING: {fname} not found at {fpath}")

    # 2. Upload original experiment results
    for fname in os.listdir(parent_dir):
        if fname.endswith(".json") and not fname.startswith("tier3_50k"):
            fpath = os.path.join(parent_dir, fname)
            print(f"    Uploading {fname}...")
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f"results/{fname}",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )

    # 3. Upload experiment outputs under the same paths used by the repository.
    if os.path.exists(outputs_dir):
        for fname in os.listdir(outputs_dir):
            fpath = os.path.join(outputs_dir, fname)
            if os.path.isfile(fpath):
                print(f"    Uploading outputs/{fname}...")
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=f"experiments/outputs/{fname}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                )

    outputs_new_dir = os.path.join(results_dir, "outputs_new")
    if os.path.exists(outputs_new_dir):
        for fname in os.listdir(outputs_new_dir):
            fpath = os.path.join(outputs_new_dir, fname)
            if os.path.isfile(fpath) and fname.endswith(".json"):
                print(f"    Uploading outputs_new/{fname}...")
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=f"experiments/outputs_new/{fname}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                )

    # 4. Upload model weights specifically to a models/ directory.
    # Prefer .safetensors when present; fall back to legacy .pt for backward compat.
    model_files = [
        "meta_controller.safetensors",
        "meta_controller_best.safetensors",
        "meta_controller.pt",
        "meta_controller_best.pt",
    ]
    for model_file in model_files:
        model_path = os.path.join(outputs_dir, model_file)
        if os.path.exists(model_path):
            print(f"    Uploading model: {model_file}...")
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=f"models/{model_file}",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )

    print("  Upload complete!")
    print(f"  View at: https://huggingface.co/datasets/{repo_id}")


def _resolve_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import HfFolder
        cached = HfFolder.get_token()
        if cached:
            return cached
    except ImportError:
        pass
    import getpass
    token = getpass.getpass("HuggingFace token (input hidden): ").strip()
    if not token:
        raise SystemExit("No HuggingFace token provided. Set HF_TOKEN or run `hf auth login`.")
    return token


if __name__ == "__main__":
    token = _resolve_token()
    results_dir = os.path.dirname(os.path.abspath(__file__))
    upload_all(results_dir, token)
