#!/usr/bin/env python3
"""
Coupling Threshold Experiment — Factual QA with Exact Match Accuracy
Measures how upstream WER degrades LLM factual accuracy.
"""

import json
import os
import random
import re
import sys
import time
import requests

# ─── Step 1: Factual QA pairs ───────────────────────────────────────────────

QA_PAIRS = [
    ("What is the capital of France?", "Paris"),
    ("What is 15 multiplied by 8?", "120"),
    ("What year did World War 2 end?", "1945"),
    ("What is the chemical symbol for gold?", "Au"),
    ("How many days are in a leap year?", "366"),
    ("What planet is closest to the sun?", "Mercury"),
    ("What is the square root of 144?", "12"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("What year was the iPhone first released?", "2007"),
    ("What is the speed of light in km/s approximately?", "300000"),
    ("What is the atomic number of carbon?", "6"),
    ("How many bones are in the adult human body?", "206"),
    ("What is the currency of Japan?", "Yen"),
    ("What gas do plants absorb from the atmosphere?", "carbon dioxide"),
    ("What is the longest river in the world?", "Nile"),
    ("How many strings does a standard guitar have?", "6"),
    ("What is the freezing point of water in Fahrenheit?", "32"),
    ("What continent is Egypt in?", "Africa"),
    ("What is the powerhouse of the cell?", "mitochondria"),
    ("How many hours are in 3 days?", "72"),
    ("What language is spoken in Brazil?", "Portuguese"),
    ("What is the tallest mountain on Earth?", "Everest"),
    ("How many planets are in our solar system?", "8"),
    ("What color is the sky on a clear day?", "blue"),
    ("What is 100 divided by 4?", "25"),
    ("What year did the Berlin Wall fall?", "1989"),
    ("What is the smallest prime number?", "2"),
]

# ─── Step 2: Word corruption function ───────────────────────────────────────

def inject_wer(text, target_wer, seed=42):
    random.seed(seed)
    words = text.split()
    n_corrupt = max(1, int(len(words) * target_wer))
    if target_wer == 0.0:
        return text
    indices = random.sample(range(len(words)), min(n_corrupt, len(words)))
    chars = 'abcdefghijklmnopqrstuvwxyz'
    for idx in indices:
        word = words[idx]
        if len(word) > 2:
            corrupt = ''.join(random.choices(chars, k=len(word)))
            words[idx] = corrupt
    return ' '.join(words)

# ─── Ollama helper ──────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
SYSTEM_PROMPT = "Answer in one word or number only. Do not explain."

def ollama_generate(model, prompt, num_predict=30, timeout=120):
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {"num_predict": num_predict},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()

def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return models
    except Exception:
        return None

# ─── Main experiment ────────────────────────────────────────────────────────

def main():
    print("COUPLING THRESHOLD EXPERIMENT (Factual QA, Exact Match)")
    print("=" * 55)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check ollama
    models = check_ollama()
    if models is None:
        print("ERROR: Cannot connect to ollama at localhost:11434")
        print("Start ollama with: ollama serve")
        sys.exit(1)
    print(f"Ollama models available: {models}")
    print()

    WER_LEVELS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]

    LLM_CONFIGS = [
        ("llama3.1:8b", "llama3.1:8b", 30),
        ("gemma2:2b (80tok)", "gemma2:2b", 30),
        ("gemma2:2b (15tok)", "gemma2:2b", 15),
    ]

    all_results = {}

    for config_name, model, num_predict in LLM_CONFIGS:
        print(f"Config: {config_name}")

        # Warm up
        try:
            ollama_generate(model, "Hello", num_predict=5, timeout=60)
        except Exception as e:
            print(f"  Warmup failed: {e}, skipping.")
            continue

        config_results = {}
        threshold = None

        for wer_level in WER_LEVELS:
            correct = 0
            details = []

            for qi, (question, ref_answer) in enumerate(QA_PAIRS):
                corrupted_q = inject_wer(question, wer_level, seed=42 + qi)

                try:
                    response = ollama_generate(model, corrupted_q, num_predict=num_predict)
                except Exception as e:
                    response = f"[ERROR: {e}]"

                # Exact match: reference answer appears as substring (case-insensitive)
                is_correct = ref_answer.lower() in response.lower()
                if is_correct:
                    correct += 1

                details.append({
                    "question": question,
                    "corrupted": corrupted_q,
                    "reference": ref_answer,
                    "response": response,
                    "correct": is_correct,
                })

            accuracy = correct / len(QA_PAIRS)
            wer_pct = int(wer_level * 100)
            print(f"  WER={wer_pct}%:   accuracy = {accuracy:.2f} ({correct}/30 correct)")

            config_results[f"wer_{wer_pct}"] = {
                "wer_pct": wer_pct,
                "accuracy": accuracy,
                "correct": correct,
                "total": len(QA_PAIRS),
                "details": details,
            }

            if threshold is None and accuracy < 0.70:
                threshold = wer_pct

        if threshold is not None:
            print(f"  THRESHOLD θ = {threshold}% (first WER where accuracy < 0.70)")
        else:
            print(f"  THRESHOLD θ = >15% (accuracy never dropped below 0.70)")

        config_results["threshold"] = threshold if threshold is not None else ">15"
        all_results[config_name] = config_results
        print()

    # ─── Step 5: Print summary table ────────────────────────────────────────

    print()
    print("COUPLING THRESHOLD RESULTS (Factual QA, Exact Match)")
    print("=" * 55)
    print()
    print("SUMMARY TABLE (LaTeX-ready)")
    print(f"{'LLM Config':<25} | {'θ (WER %)'}")
    print("-" * 25 + "-+-" + "-" * 10)
    for config_name in [c[0] for c in LLM_CONFIGS]:
        if config_name in all_results:
            t = all_results[config_name]["threshold"]
            t_str = f"{t}%" if isinstance(t, int) else t
            print(f"{config_name:<25} | {t_str}")
    print()

    # LaTeX table
    print("% LaTeX table:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Coupling threshold $\\theta_j(c_j)$: min upstream WER before factual accuracy $< 0.70$}")
    print("\\begin{tabular}{lr}")
    print("\\toprule")
    print("LLM Config & $\\theta$ (WER \\%) \\\\")
    print("\\midrule")
    for config_name in [c[0] for c in LLM_CONFIGS]:
        if config_name in all_results:
            t = all_results[config_name]["threshold"]
            t_str = f"{t}\\%" if isinstance(t, int) else f"$>{t[1:]}$"
            print(f"{config_name} & {t_str} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()

    # Full accuracy table
    print("% Full accuracy-vs-WER table:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Factual QA accuracy vs.\\ upstream WER degradation}")
    print("\\begin{tabular}{l" + "r" * len(WER_LEVELS) + "}")
    print("\\toprule")
    header = "Config & " + " & ".join([f"{int(w*100)}\\%" for w in WER_LEVELS]) + " \\\\"
    print(header)
    print("\\midrule")
    for config_name in [c[0] for c in LLM_CONFIGS]:
        if config_name in all_results:
            cr = all_results[config_name]
            vals = []
            for wer_level in WER_LEVELS:
                wer_pct = int(wer_level * 100)
                key = f"wer_{wer_pct}"
                if key in cr:
                    acc = cr[key]["accuracy"]
                    vals.append(f"{acc:.2f}")
                else:
                    vals.append("---")
            print(f"{config_name} & " + " & ".join(vals) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # Save results
    # Strip details for JSON (too large), keep summary
    save_results = {}
    for config_name, cr in all_results.items():
        save_results[config_name] = {}
        for key, val in cr.items():
            if isinstance(val, dict) and "details" in val:
                save_results[config_name][key] = {
                    k: v for k, v in val.items() if k != "details"
                }
            else:
                save_results[config_name][key] = val

    with open("coupling_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nAll results saved to coupling_results.json")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
