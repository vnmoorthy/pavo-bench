#!/usr/bin/env python3
"""
TIER 3.2: PAVO-Bench 50K — RESUME from checkpoint_30000
Continues generating remaining 20,000 samples.
"""

import json
import os
import random
import requests
import time
from tqdm import tqdm

BENCH_DIR = os.path.expanduser("~/pavo-gpu-experiments/pavo_bench_50k")
RESULTS_DIR = os.path.expanduser("~/pavo-gpu-experiments/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("TIER 3: PAVO-Bench 50K — RESUMING from checkpoint_30000")
print("=" * 70)

# ────────────────────────────────────────────────────────────────────────────
# 1. Load existing checkpoint
# ────────────────────────────────────────────────────────────────────────────
print("\n[1/3] Loading checkpoint_30000...")
with open(os.path.join(BENCH_DIR, "checkpoint_30000.json")) as f:
    existing_data = json.load(f)

print(f"✓ Loaded {len(existing_data)} existing samples")
print(f"  Keys: {list(existing_data[0].keys())}")

# ────────────────────────────────────────────────────────────────────────────
# 2. Generate remaining 20,000 samples
# ────────────────────────────────────────────────────────────────────────────
TOTAL = 50000
remaining = TOTAL - len(existing_data)
print(f"\n[2/3] Generating {remaining} remaining samples...")

# Same complexity distribution
complexity_dist = {1: 0.25, 2: 0.30, 3: 0.25, 4: 0.15, 5: 0.05}

# Same query templates as original
query_templates = {
    1: [
        "What time is it?",
        "Set a timer for {} minutes",
        "Turn off the lights",
        "What's the weather?",
        "Call {}",
        "Play some music",
        "What's today's date?",
        "Set an alarm for {} AM",
        "Turn up the volume",
        "Stop the music",
    ],
    2: [
        "Schedule a meeting for tomorrow at {}pm",
        "Send a message to {} about the project",
        "What's the news today?",
        "Play my favorite playlist",
        "Remind me to buy groceries at {}pm",
        "How far is {} from here?",
        "What's the traffic like?",
        "Order something from Amazon",
        "Add milk to my shopping list",
        "What's {}'s phone number?",
    ],
    3: [
        "Summarize my emails from the past week",
        "What's the cheapest flight to {} next month?",
        "Translate 'hello how are you' to Spanish",
        "How many calories are in a banana?",
        "Create a shopping list for dinner",
        "What are the reviews for that restaurant?",
        "Compare prices of laptops across stores",
        "Find a restaurant near me open now",
        "What's the recipe for pasta carbonara?",
        "How do I fix a leaky faucet?",
    ],
    4: [
        "Analyze my calendar and suggest the best time for a meeting next week",
        "Compare the features of iPhone 15 and Samsung Galaxy S24",
        "Write a summary of the book 'Atomic Habits'",
        "What are the pros and cons of solar panels?",
        "Plan a 3-day trip to {} with budget $2000",
        "Help me draft an email to my boss about the project update",
        "What are the best laptops under $1000?",
        "Explain the difference between machine learning and deep learning",
        "Create a workout plan for weight loss",
        "What should I invest in given the current market?",
    ],
    5: [
        "Generate a business plan for a voice AI startup",
        "Explain quantum computing and its applications in cryptography",
        "Design a system architecture for a real-time voice translation service",
        "Compare different ML frameworks for NLP tasks",
        "Draft a research proposal on routing policies for edge computing",
        "Analyze the market trends in AI for the next 5 years",
        "Create a comprehensive marketing strategy for a tech product",
        "Explain the implications of AI regulation on healthcare",
        "Design a machine learning pipeline for speech recognition",
        "Write a technical specification for a voice assistant router",
    ]
}

names = ["John", "Sarah", "Mike", "Emily", "Alex", "Lisa", "David", "Rachel", "Tom", "Amy"]
cities = ["New York", "London", "Tokyo", "Paris", "Sydney", "Dubai", "Berlin", "Toronto", "Seoul", "Mumbai"]
noise_types = ["white", "babble", "traffic", "music", "clean"]

def make_query(complexity):
    template = random.choice(query_templates[complexity])
    return template.format(random.choice(names + cities + [str(random.randint(1, 10))]))

new_data = []
errors = 0
start_time = time.time()
BATCH_SIZE = 100

for i in tqdm(range(remaining), desc="LLM responses"):
    idx = len(existing_data) + i
    complexity = random.choices(list(complexity_dist.keys()), weights=list(complexity_dist.values()))[0]
    query = make_query(complexity)

    # Generate demand vector
    snr = float(random.uniform(5, 40))
    cpu_util = float(random.uniform(0.1, 0.9))
    battery = float(random.uniform(0.1, 1.0))
    rtt_ms = float(abs(random.gauss(50, 30)))
    ctx_tokens = int(150 + complexity * 100)
    noise_type = random.choice(noise_types)

    # Call ollama for reference response
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.1:8b',
                'prompt': f"User: {query}\nAssistant:",
                'temperature': 0,
                'seed': 42 + idx,
                'stream': False
            },
            timeout=60
        )
        if response.status_code == 200:
            ref_response = response.json().get('response', '').strip()
        else:
            ref_response = f"[HTTP_{response.status_code}]"
            errors += 1
    except requests.exceptions.Timeout:
        ref_response = "[TIMEOUT]"
        errors += 1
    except Exception as e:
        ref_response = f"[ERROR: {str(e)[:50]}]"
        errors += 1

    item = {
        "index": int(idx),
        "complexity": int(complexity),
        "snr_db": snr,
        "noise_type": noise_type,
        "cpu_util": cpu_util,
        "battery": battery,
        "rtt_ms": rtt_ms,
        "ctx_tokens": ctx_tokens,
        "source": "synthetic",
        "audio_idx": int(idx % 4620),
        "user_input": query,
        "reference_response": ref_response
    }
    new_data.append(item)

    # Save checkpoint every 5000
    if (i + 1) % 5000 == 0:
        checkpoint_count = len(existing_data) + len(new_data)
        checkpoint_path = os.path.join(BENCH_DIR, f"checkpoint_{checkpoint_count}.json")
        with open(checkpoint_path, "w") as f:
            json.dump(existing_data + new_data, f)
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        eta = (remaining - i - 1) / rate
        print(f"\n  Checkpoint saved: {checkpoint_count} samples | "
              f"Rate: {rate:.2f} samples/sec | "
              f"Errors: {errors} | "
              f"ETA: {eta/3600:.1f} hours")

elapsed = time.time() - start_time
print(f"\n✓ Generated {len(new_data)} new responses in {elapsed/3600:.1f} hours")
print(f"  Errors: {errors}")

# ────────────────────────────────────────────────────────────────────────────
# 3. Combine and create final dataset
# ────────────────────────────────────────────────────────────────────────────
print("\n[3/3] Creating final 50K dataset...")

all_data = existing_data + new_data
random.shuffle(all_data)

split_idx = int(len(all_data) * 0.8)
train_data = all_data[:split_idx]
test_data = all_data[split_idx:]

# Ensure all values are JSON-serializable (convert numpy/bool types)
def sanitize(item):
    sanitized = {}
    for k, v in item.items():
        if isinstance(v, bool):
            sanitized[k] = bool(v)
        elif isinstance(v, (int,)):
            sanitized[k] = int(v)
        elif isinstance(v, (float,)):
            sanitized[k] = float(v)
        else:
            sanitized[k] = v
    return sanitized

with open(os.path.join(RESULTS_DIR, "tier3_50k_train.jsonl"), "w") as f:
    for item in train_data:
        f.write(json.dumps(sanitize(item)) + "\n")

with open(os.path.join(RESULTS_DIR, "tier3_50k_test.jsonl"), "w") as f:
    for item in test_data:
        f.write(json.dumps(sanitize(item)) + "\n")

summary = {
    "dataset": "PAVO-Bench 50K",
    "total_samples": len(all_data),
    "train_samples": len(train_data),
    "test_samples": len(test_data),
    "generation_time_hours": float(elapsed / 3600),
    "resumed_from": 30000,
    "new_samples_generated": len(new_data),
    "error_count": int(errors),
    "error_rate": float(errors / max(len(new_data), 1)),
    "complexity_distribution": {str(k): float(v) for k, v in complexity_dist.items()},
}

with open(os.path.join(RESULTS_DIR, "tier3_50k_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*70}")
print(f"✓ PAVO-Bench 50K COMPLETE!")
print(f"  Total: {len(all_data)} samples")
print(f"  Train: {len(train_data)} → tier3_50k_train.jsonl")
print(f"  Test: {len(test_data)} → tier3_50k_test.jsonl")
print(f"  Resumed from: 30,000 | New: {len(new_data)}")
print(f"  Time for new samples: {elapsed/3600:.1f} hours")
print(f"  Errors: {errors}")
print(f"{'='*70}")
