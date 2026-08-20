#!/usr/bin/env python3
"""
TIER 1: Statistical Significance
- Re-run routing simulation 5× with different seeds
- Compute mean ± std for all metrics
- Run paired t-tests and Wilcoxon signed-rank tests
- Compute 95% confidence intervals
- Output: tier1_statistical_results.json
"""
import json
import os
import time
import numpy as np
from pathlib import Path

print("=" * 70)
print("TIER 1: Statistical Significance Analysis")
print("=" * 70)

os.system("pip install -q --break-system-packages scipy tqdm 2>/dev/null")

from scipy import stats
from tqdm import tqdm

RESULTS_DIR = Path("/home/ubuntu/pavo-gpu-experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# PAVO routing simulation (same as Block 2 but parameterized)
# ─────────────────────────────────────────────────────────────

# Pipeline component catalog
ASR_OPTIONS = {
    "Whisper_Large_FP16": {"wer": 5.1, "latency_ms": 320, "energy_mj": 180, "cost_usd": 0.006, "location": "cloud"},
    "Whisper_Tiny_INT8": {"wer": 12.8, "latency_ms": 45, "energy_mj": 15, "cost_usd": 0.0, "location": "ondevice"},
    "Parakeet_FP16": {"wer": 1.9, "latency_ms": 280, "energy_mj": 160, "cost_usd": 0.005, "location": "cloud"},
}

LLM_OPTIONS = {
    "GPT4o_mini": {"quality": 0.95, "latency_15": 850, "latency_80": 3200, "energy_mj": 0, "cost_usd": 0.015, "location": "cloud"},
    "Llama3_8B_FP16": {"quality": 0.88, "latency_15": 420, "latency_80": 1600, "energy_mj": 320, "cost_usd": 0.0, "location": "cloud"},
    "Gemma2_2B_INT4": {"quality": 0.72, "latency_15": 120, "latency_80": 450, "energy_mj": 85, "cost_usd": 0.0, "location": "ondevice"},
}

TTS_OPTIONS = {
    "Cloud_TTS": {"latency_ms": 200, "energy_mj": 5, "cost_usd": 0.004, "quality": 0.95, "location": "cloud"},
    "Kokoro": {"latency_ms": 350, "energy_mj": 120, "cost_usd": 0.0, "quality": 0.88, "location": "ondevice"},
}

ONDEVICE_QUANT = {"Gemma2_2B_INT4"}

def generate_demand_vector(seed):
    """Generate a random demand vector."""
    rng = np.random.RandomState(seed)
    snr = rng.uniform(5, 40)
    cpu_util = rng.uniform(0.1, 0.95)
    battery = rng.uniform(0.05, 1.0)
    rtt_ms = rng.lognormal(mean=3.5, sigma=0.8)
    rtt_ms = min(rtt_ms, 500)
    ctx_tokens = rng.choice([50, 100, 200, 300, 500, 800])
    complexity = rng.choice([1, 2, 3, 4, 5], p=[0.25, 0.30, 0.25, 0.15, 0.05])
    return {
        "snr": snr, "cpu_util": cpu_util, "battery": battery,
        "rtt_ms": rtt_ms, "ctx_tokens": ctx_tokens, "complexity": complexity,
    }

def compute_cost(asr_name, llm_name, tts_name, demand):
    """Feature-dependent cost function with coupling constraint."""
    asr = ASR_OPTIONS[asr_name]
    llm = LLM_OPTIONS[llm_name]
    tts = TTS_OPTIONS[tts_name]

    snr = demand["snr"]
    cpu_util = demand["cpu_util"]
    battery = demand["battery"]
    rtt_ms = demand["rtt_ms"]
    ctx_tokens = demand["ctx_tokens"]

    # SNR degrades ASR WER
    if snr < 15:
        snr_mult = 1.0 + (15 - snr) / 15 * 2.0
    elif snr < 25:
        snr_mult = 1.0 + (25 - snr) / 25 * 0.5
    else:
        snr_mult = 1.0
    effective_wer = min(asr["wer"] * snr_mult, 60.0)

    # Coupling constraint
    if effective_wer > 2.0 and llm_name in ONDEVICE_QUANT:
        return float("inf"), {}, {}

    # Latency
    use_short = ctx_tokens < 300
    llm_lat = llm["latency_15"] if use_short else llm["latency_80"]
    if llm["location"] == "ondevice":
        llm_lat *= (1.0 + cpu_util * 3.0)
    if llm["location"] == "cloud":
        llm_lat += rtt_ms
    tts_lat = tts["latency_ms"]
    if tts["location"] == "cloud":
        tts_lat += rtt_ms
    asr_lat = asr["latency_ms"]
    if asr["location"] == "cloud":
        asr_lat += rtt_ms
    total_lat = asr_lat + llm_lat + tts_lat

    # Quality
    wer_penalty = effective_wer / 100.0
    quality = llm["quality"] * tts["quality"] * (1.0 - 0.5 * wer_penalty)

    # Energy
    total_energy = asr["energy_mj"] + llm["energy_mj"] + tts["energy_mj"]

    # Cost
    total_cost = asr["cost_usd"] + llm["cost_usd"] + tts["cost_usd"]

    # Normalize
    L_hat = min(total_lat / 5000.0, 1.0)
    E_hat = min(total_energy / 500.0, 1.0)
    Q = quality

    # Adaptive weights
    e_w = 0.15 + 0.20 * (1.0 - battery)
    l_w = 0.35
    m_w = 0.10
    q_w = 1.0 - l_w - e_w - m_w

    cost = l_w * L_hat + e_w * E_hat + m_w * total_cost - q_w * Q

    metrics = {
        "latency_ms": total_lat,
        "quality": quality,
        "energy_mj": total_energy,
        "cost_usd": total_cost,
        "effective_wer": effective_wer,
    }

    config = {"asr": asr_name, "llm": llm_name, "tts": tts_name}

    return cost, metrics, config

def pavo_route(demand):
    """PAVO optimal routing."""
    best_cost = float("inf")
    best_metrics = None
    best_config = None
    for asr_name in ASR_OPTIONS:
        for llm_name in LLM_OPTIONS:
            for tts_name in TTS_OPTIONS:
                cost, metrics, config = compute_cost(asr_name, llm_name, tts_name, demand)
                if cost < best_cost:
                    best_cost = cost
                    best_metrics = metrics
                    best_config = config
    return best_cost, best_metrics, best_config

def always_cloud_route(demand):
    return compute_cost("Whisper_Large_FP16", "GPT4o_mini", "Cloud_TTS", demand)

def always_ondevice_route(demand):
    return compute_cost("Parakeet_FP16", "Gemma2_2B_INT4", "Kokoro", demand)

def random_route(demand, rng):
    asr = rng.choice(list(ASR_OPTIONS.keys()))
    llm = rng.choice(list(LLM_OPTIONS.keys()))
    tts = rng.choice(list(TTS_OPTIONS.keys()))
    return compute_cost(asr, llm, tts, demand)

# ─────────────────────────────────────────────────────────────
# Run 5× with different seeds
# ─────────────────────────────────────────────────────────────
N_TURNS = 1000
N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 1024]

print(f"\nRunning {N_SEEDS} trials × {N_TURNS} turns each...")

all_runs = {
    "pavo": {"latency": [], "quality": [], "cost": []},
    "cloud": {"latency": [], "quality": [], "cost": []},
    "ondevice": {"latency": [], "quality": [], "cost": []},
    "random": {"latency": [], "quality": [], "cost": []},
}

for trial, seed in enumerate(SEEDS):
    print(f"\n  Trial {trial+1}/{N_SEEDS} (seed={seed})")

    trial_metrics = {
        "pavo": {"latency": [], "quality": [], "cost": []},
        "cloud": {"latency": [], "quality": [], "cost": []},
        "ondevice": {"latency": [], "quality": [], "cost": []},
        "random": {"latency": [], "quality": [], "cost": []},
    }
    rng = np.random.RandomState(seed)

    for t in range(N_TURNS):
        demand = generate_demand_vector(seed * 10000 + t)

        # PAVO
        _, m, _ = pavo_route(demand)
        if m:
            trial_metrics["pavo"]["latency"].append(m["latency_ms"])
            trial_metrics["pavo"]["quality"].append(m["quality"])
            trial_metrics["pavo"]["cost"].append(m["cost_usd"])

        # Cloud
        _, m, _ = always_cloud_route(demand)
        if m:
            trial_metrics["cloud"]["latency"].append(m["latency_ms"])
            trial_metrics["cloud"]["quality"].append(m["quality"])
            trial_metrics["cloud"]["cost"].append(m["cost_usd"])

        # Ondevice
        _, m, _ = always_ondevice_route(demand)
        if m:
            trial_metrics["ondevice"]["latency"].append(m["latency_ms"])
            trial_metrics["ondevice"]["quality"].append(m["quality"])
            trial_metrics["ondevice"]["cost"].append(m["cost_usd"])

        # Random
        _, m, _ = random_route(demand, rng)
        if m:
            trial_metrics["random"]["latency"].append(m["latency_ms"])
            trial_metrics["random"]["quality"].append(m["quality"])
            trial_metrics["random"]["cost"].append(m["cost_usd"])

    # Store per-trial means
    for policy in all_runs:
        for metric in ["latency", "quality", "cost"]:
            vals = trial_metrics[policy][metric]
            if vals:
                all_runs[policy][metric].append(np.mean(vals))

    print(f"    PAVO latency: {np.mean(trial_metrics['pavo']['latency']):.0f}ms")
    print(f"    Cloud latency: {np.mean(trial_metrics['cloud']['latency']):.0f}ms")

# ─────────────────────────────────────────────────────────────
# Statistical tests
# ─────────────────────────────────────────────────────────────
print("\n\nStatistical Analysis:")
print("=" * 50)

results = {"trials": N_SEEDS, "turns_per_trial": N_TURNS, "seeds": SEEDS}

for policy in all_runs:
    for metric in ["latency", "quality", "cost"]:
        vals = all_runs[policy][metric]
        key = f"{policy}_{metric}"
        results[key] = {
            "values": [round(v, 4) for v in vals],
            "mean": round(np.mean(vals), 4),
            "std": round(np.std(vals), 4),
            "ci_95_lower": round(np.mean(vals) - 1.96 * np.std(vals) / np.sqrt(len(vals)), 4),
            "ci_95_upper": round(np.mean(vals) + 1.96 * np.std(vals) / np.sqrt(len(vals)), 4),
        }

# Paired comparisons: PAVO vs each baseline
comparisons = {}
for baseline in ["cloud", "ondevice", "random"]:
    for metric in ["latency", "quality", "cost"]:
        pavo_vals = all_runs["pavo"][metric]
        base_vals = all_runs[baseline][metric]

        if len(pavo_vals) >= 3 and len(base_vals) >= 3:
            # Paired t-test
            t_stat, p_value_t = stats.ttest_rel(pavo_vals, base_vals)
            # Wilcoxon signed-rank test
            try:
                w_stat, p_value_w = stats.wilcoxon(pavo_vals, base_vals)
            except:
                w_stat, p_value_w = 0, 1.0

            key = f"pavo_vs_{baseline}_{metric}"
            comparisons[key] = {
                "pavo_mean": round(np.mean(pavo_vals), 4),
                "baseline_mean": round(np.mean(base_vals), 4),
                "difference": round(np.mean(pavo_vals) - np.mean(base_vals), 4),
                "t_statistic": round(t_stat, 4),
                "p_value_ttest": round(p_value_t, 6),
                "w_statistic": round(float(w_stat), 4),
                "p_value_wilcoxon": round(float(p_value_w), 6),
                "significant_at_005": p_value_t < 0.05,
                "significant_at_001": p_value_t < 0.01,
            }

            sig = "***" if p_value_t < 0.001 else "**" if p_value_t < 0.01 else "*" if p_value_t < 0.05 else "n.s."
            print(f"  PAVO vs {baseline} ({metric}): p={p_value_t:.6f} {sig}")

results["comparisons"] = comparisons

# ─────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────
output_path = RESULTS_DIR / "tier1_statistical_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Tier 1 Statistical Significance complete! Results saved to {output_path}")
