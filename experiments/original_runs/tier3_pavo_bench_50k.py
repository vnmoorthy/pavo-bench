#!/usr/bin/env python3
"""
TIER 3: PAVO-Bench 50K Dataset
- Scale the dataset from 5K to 50K turns
- Diverse speakers, noise conditions, complexity levels
- Generate reference responses for all turns
- Output: train_50k.jsonl, test_50k.jsonl, metadata_50k.json
"""
import json
import os
import time
import numpy as np
import requests
from pathlib import Path
from tqdm import tqdm

print("=" * 70)
print("TIER 3: PAVO-Bench 50K Dataset Generation")
print("=" * 70)

os.system("pip install -q --break-system-packages soundfile datasets tqdm 2>/dev/null")

import soundfile as sf

RESULTS_DIR = Path("/home/ubuntu/pavo-gpu-experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BENCH_DIR = Path("/home/ubuntu/pavo-gpu-experiments/pavo_bench_50k")
BENCH_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"

# ─────────────────────────────────────────────────────────────
# Step 1: Load source datasets
# ─────────────────────────────────────────────────────────────
print("\n[1/4] Loading source audio datasets...")
from datasets import load_dataset

# LibriSpeech
print("  Loading LibriSpeech...")
try:
    ls_ds = load_dataset("librispeech_asr", "clean", split="test", trust_remote_code=True)
    ls_train = load_dataset("librispeech_asr", "clean", split="train.clean.100", trust_remote_code=True)
    print(f"    LibriSpeech test: {len(ls_ds)}, train: {len(ls_train)}")
except Exception as e:
    print(f"    LibriSpeech error: {e}")
    ls_ds = None
    ls_train = None

# Common Voice (if available)
print("  Loading Common Voice...")
try:
    cv_ds = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="test", trust_remote_code=True)
    print(f"    Common Voice: {len(cv_ds)}")
except Exception as e:
    print(f"    Common Voice error: {e}")
    cv_ds = None

# ─────────────────────────────────────────────────────────────
# Step 2: Generate diverse demand vectors
# ─────────────────────────────────────────────────────────────
print("\n[2/4] Generating 50K diverse demand vectors...")

COMPLEXITY_DIST = {1: 0.20, 2: 0.25, 3: 0.25, 4: 0.20, 5: 0.10}
NOISE_TYPES = ["clean", "white", "pink", "babble", "traffic"]
SNR_RANGE = (5, 40)

N_TOTAL = 50000
rng = np.random.RandomState(42)

metadata = []
for i in tqdm(range(N_TOTAL), desc="  Generating metadata"):
    complexity = rng.choice(list(COMPLEXITY_DIST.keys()), p=list(COMPLEXITY_DIST.values()))
    snr = rng.uniform(*SNR_RANGE)
    noise_type = rng.choice(NOISE_TYPES)
    cpu_util = rng.uniform(0.1, 0.95)
    battery = rng.uniform(0.05, 1.0)
    rtt_ms = min(rng.lognormal(3.5, 0.8), 500)
    ctx_tokens = rng.choice([50, 100, 200, 300, 500, 800])

    # Select source audio
    if ls_ds and ls_train:
        total_ls = len(ls_ds) + len(ls_train)
        source_idx = i % total_ls
        if source_idx < len(ls_ds):
            source = "librispeech_test"
            audio_idx = source_idx
        else:
            source = "librispeech_train"
            audio_idx = source_idx - len(ls_ds)
    else:
        source = "synthetic"
        audio_idx = i

    metadata.append({
        "index": i,
        "complexity": int(complexity),
        "snr_db": round(snr, 1),
        "noise_type": noise_type,
        "cpu_util": round(cpu_util, 3),
        "battery": round(battery, 3),
        "rtt_ms": round(rtt_ms, 1),
        "ctx_tokens": int(ctx_tokens),
        "source": source,
        "audio_idx": audio_idx,
    })

# ─────────────────────────────────────────────────────────────
# Step 3: Generate user queries based on complexity
# ─────────────────────────────────────────────────────────────
print("\n[3/4] Generating user queries and reference responses...")

QUERY_TEMPLATES = {
    1: [  # Simple
        "What time is it?", "Set an alarm for {h} o'clock.",
        "What's the weather today?", "Turn {on_off} the {device}.",
        "Play some {genre} music.", "Call {name}.",
        "How far is the nearest {place}?", "What's {num1} plus {num2}?",
    ],
    2: [  # Medium
        "What's the traffic like on my way to {place}?",
        "Remind me to {task} at {h} {ampm}.",
        "Search for {cuisine} restaurants nearby with good reviews.",
        "What are the top news headlines today?",
        "Read me my unread messages.",
    ],
    3: [  # Complex
        "Plan a {duration}-day trip to {city} with a budget of ${budget}.",
        "Compare {product1} and {product2} for {use_case}.",
        "Summarize the key points from my last meeting about {topic}.",
        "What are the pros and cons of {option1} versus {option2}?",
    ],
    4: [  # Very complex
        "Help me debug this error: '{error_msg}'. I'm using {language} with {framework}.",
        "Write a professional email to {recipient} about {subject}, mentioning {details}.",
        "Analyze the sentiment of recent reviews for {product} and suggest improvements.",
    ],
    5: [  # Expert
        "Design a microservices architecture for a {system_type} that handles {load} requests per second.",
        "Explain the tradeoffs between {tech1} and {tech2} for building {application_type} in production.",
    ],
}

FILL_VALUES = {
    "h": lambda rng: str(rng.randint(1, 13)),
    "on_off": lambda rng: rng.choice(["on", "off"]),
    "device": lambda rng: rng.choice(["lights", "fan", "TV", "AC", "heater"]),
    "genre": lambda rng: rng.choice(["jazz", "rock", "classical", "pop", "hip-hop"]),
    "name": lambda rng: rng.choice(["John", "Sarah", "Mom", "Dad", "Alex"]),
    "place": lambda rng: rng.choice(["gas station", "hospital", "airport", "restaurant"]),
    "num1": lambda rng: str(rng.randint(1, 100)),
    "num2": lambda rng: str(rng.randint(1, 100)),
    "cuisine": lambda rng: rng.choice(["Italian", "Chinese", "Indian", "Mexican", "Japanese"]),
    "task": lambda rng: rng.choice(["buy groceries", "call the dentist", "submit the report"]),
    "ampm": lambda rng: rng.choice(["AM", "PM"]),
    "duration": lambda rng: str(rng.randint(3, 10)),
    "city": lambda rng: rng.choice(["Tokyo", "Paris", "Rome", "Barcelona", "New York"]),
    "budget": lambda rng: str(rng.choice([500, 1000, 2000, 5000])),
    "product1": lambda rng: rng.choice(["iPhone 15", "MacBook Pro", "Tesla Model 3"]),
    "product2": lambda rng: rng.choice(["Samsung S24", "ThinkPad X1", "BMW i4"]),
    "use_case": lambda rng: rng.choice(["daily use", "professional work", "long trips"]),
    "topic": lambda rng: rng.choice(["project timeline", "budget allocation", "hiring"]),
    "option1": lambda rng: rng.choice(["remote work", "cloud hosting", "React"]),
    "option2": lambda rng: rng.choice(["office work", "on-premise", "Vue"]),
    "error_msg": lambda rng: rng.choice(["IndexError", "NullPointerException", "CORS error"]),
    "language": lambda rng: rng.choice(["Python", "JavaScript", "Java"]),
    "framework": lambda rng: rng.choice(["FastAPI", "React", "Spring Boot"]),
    "recipient": lambda rng: rng.choice(["the team", "the client", "management"]),
    "subject": lambda rng: rng.choice(["project delay", "new feature", "quarterly review"]),
    "details": lambda rng: rng.choice(["the deadline change", "budget constraints", "new requirements"]),
    "product": lambda rng: rng.choice(["our mobile app", "the new feature", "customer support"]),
    "system_type": lambda rng: rng.choice(["e-commerce platform", "social media app", "IoT dashboard"]),
    "load": lambda rng: str(rng.choice([1000, 5000, 10000, 50000])),
    "tech1": lambda rng: rng.choice(["Kubernetes", "PostgreSQL", "gRPC"]),
    "tech2": lambda rng: rng.choice(["Docker Swarm", "MongoDB", "REST"]),
    "application_type": lambda rng: rng.choice(["real-time analytics", "recommendation system", "chat application"]),
}

def fill_template(template, rng):
    result = template
    import re
    placeholders = re.findall(r'\{(\w+)\}', template)
    for ph in placeholders:
        if ph in FILL_VALUES:
            result = result.replace("{" + ph + "}", FILL_VALUES[ph](rng), 1)
    return result

# Generate queries
print("  Generating queries...")
for i, item in enumerate(tqdm(metadata, desc="  Queries")):
    complexity = item["complexity"]
    templates = QUERY_TEMPLATES[complexity]
    template = rng.choice(templates)
    item["user_input"] = fill_template(template, rng)

# Generate reference responses via ollama (batch)
print("\n  Generating reference responses via ollama...")
BATCH_SIZE = 100
n_done = 0
n_errors = 0

for i in tqdm(range(0, N_TOTAL, BATCH_SIZE), desc="  LLM responses"):
    batch = metadata[i:i+BATCH_SIZE]
    for item in batch:
        prompt = f"User: {item['user_input']}\nAssistant: Provide a helpful, concise response."
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": "llama3.1:8b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "seed": 42, "num_predict": 150}
            }, timeout=30)
            item["reference_response"] = resp.json()["response"].strip()
            n_done += 1
        except:
            item["reference_response"] = "[TIMEOUT]"
            n_errors += 1

    # Progress checkpoint every 5000
    if (i + BATCH_SIZE) % 5000 == 0:
        print(f"    Progress: {i+BATCH_SIZE}/{N_TOTAL} ({n_errors} errors)")
        # Save checkpoint
        checkpoint_path = BENCH_DIR / f"checkpoint_{i+BATCH_SIZE}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(metadata[:i+BATCH_SIZE], f)

# ─────────────────────────────────────────────────────────────
# Step 4: Split and save
# ─────────────────────────────────────────────────────────────
print("\n[4/4] Saving dataset...")

# 80/20 train/test split
rng.shuffle(metadata)
split_idx = int(N_TOTAL * 0.8)
train_data = metadata[:split_idx]
test_data = metadata[split_idx:]

train_path = BENCH_DIR / "train_50k.jsonl"
test_path = BENCH_DIR / "test_50k.jsonl"
meta_path = BENCH_DIR / "metadata_50k.json"

with open(train_path, "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open(test_path, "w") as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")

dataset_meta = {
    "name": "PAVO-Bench 50K",
    "version": "2.0",
    "total_samples": N_TOTAL,
    "train_samples": len(train_data),
    "test_samples": len(test_data),
    "n_errors": n_errors,
    "complexity_distribution": COMPLEXITY_DIST,
    "noise_types": NOISE_TYPES,
    "snr_range": list(SNR_RANGE),
    "sources": ["librispeech_test", "librispeech_train"],
}

with open(meta_path, "w") as f:
    json.dump(dataset_meta, f, indent=2)

print(f"\n✓ PAVO-Bench 50K complete!")
print(f"  Train: {len(train_data)} samples → {train_path}")
print(f"  Test: {len(test_data)} samples → {test_path}")
print(f"  Errors: {n_errors}/{N_TOTAL}")

# Copy to results dir too
output_path = RESULTS_DIR / "tier3_bench50k_stats.json"
with open(output_path, "w") as f:
    json.dump(dataset_meta, f, indent=2)
