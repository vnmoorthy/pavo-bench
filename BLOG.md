# The Coupling Cliff: Why You Can't Tune Voice Pipelines One Stage at a Time

*A teaching-first walkthrough of the finding behind [PAVO-Bench](https://github.com/vnmoorthy/pavo-bench).*

> **Camera-ready artifact note (August 2026):** The compact experiment below
> is an exploratory teaching example, not the final H100 calibration or a
> byte-for-byte reproduction recipe. Use the
> [result-provenance map](docs/RESULT_PROVENANCE.md) and
> [artifact-status catalog](docs/ARTIFACT_STATUS.md) when connecting paper
> claims to released files.

---

Real-time voice assistants are a pipeline: **speech → ASR → LLM → TTS → speech**.

The usual way to make one faster is to tune each stage in isolation. Ship a smaller Whisper, pick a quantized Llama, swap the TTS for something streaming. Three independent wins, ship it.

That approach leaves a lot on the table. The small runnable example below
illustrates the intuition on a laptop; the final aggregate evidence and its
limitations are documented separately in the camera-ready artifact set.

The short version: **the stages aren't independent.** The quality you can get out of the LLM is bounded, sharply, by the word-error rate of the ASR that feeds it. Miss the bound and your LLM doesn't degrade gracefully — it falls off a cliff.

We call that the *coupling cliff*. Once you know it exists, you stop tuning stage-by-stage and start routing.

## The experiment

Take one LLM — let's say Llama 3.1 8B — and feed it a fixed set of ten questions. For each question, intentionally corrupt the wording to simulate what a bad ASR would have produced, and measure the quality of the LLM's answer against the clean reference. Sweep the corruption rate from 0% to 20% in nine steps.

Here's the core loop, 40 lines of Python. You can paste it into a notebook:

```python
import random, re
from typing import Callable

QUERIES = [
    "What time is it in Tokyo?",
    "Summarize this document in one sentence.",
    "Remind me to call mom at six.",
    "Translate 'good morning' into French.",
    "How long is the flight from JFK to LAX?",
    "What's the weather like tomorrow?",
    "Set an alarm for seven.",
    "Play some jazz music.",
    "Define 'photosynthesis' in one sentence.",
    "What's 48 divided by 6?",
]

def inject_wer(text: str, pct: float, rng: random.Random) -> str:
    toks = re.findall(r"\S+|\s+", text)
    for _ in range(int(len(toks) * pct / 100.0)):
        i = rng.randrange(len(toks))
        op = rng.choice(["sub", "del", "ins"])
        if op == "sub" and toks[i].strip():
            toks[i] = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(len(toks[i])))
        elif op == "del":
            toks.pop(i)
        else:
            toks.insert(i, " " + "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(4)) + " ")
    return "".join(toks)

def quality(pred: str, ref: str) -> float:
    ref_toks = set(re.findall(r"[a-z0-9']+", ref.lower()))
    out_toks = set(re.findall(r"[a-z0-9']+", pred.lower()))
    return len(ref_toks & out_toks) / max(len(ref_toks), 1)

def sweep(llm_fn: Callable[[str], str]):
    rng = random.Random(42)
    for wer in [0, 1, 2, 3, 5, 8, 10, 15, 20]:
        scores = [quality(llm_fn(inject_wer(q, wer, rng) if wer else q), q) for q in QUERIES]
        print(f"WER {wer:>2d}%   mean quality = {sum(scores)/len(scores):.3f}")
```

Swap in your LLM and run it. Or — if you don't have a GPU handy — `pip install pavo-bench` and call `reproduce_coupling_cliff(llm_fn)` with the same shape of input; the library's already set up.

The committed camera-ready figure below is generated from the separate
three-model H100 aggregate, not from the ten-question teaching loop above:

![Coupling cliff](figures/coupling_cliff.png)

The larger H100 calibration uses 200 queries per WER level for each of three LLM families. Quality is stable through 10% injected WER and then degrades at 15–20%. A smaller preliminary M3 factual-QA calibration showed sensitivity around 2%; PAVO therefore uses 2% as a conservative safety threshold, not as the measured H100 cliff location.

We reran the H100 calibration with Llama 3.1 8B, Mistral 7B, and Gemma2 2B. The exact degradation profile varies by model, but all three show the same stable-then-degrading two-regime structure.

## Why this breaks stage-by-stage optimization

If you're a systems engineer optimizing a voice pipeline, your instinct is:

> "Our P95 latency is dominated by the ASR. Let me swap the large Whisper for tiny. I'll take a WER hit but the LLM can handle it."

On paper this looks fine. You measure Whisper-large: 4.2% WER. Whisper-tiny: 6.1% WER. That's a 50% speedup for a 2-point WER regression. Great trade.

In practice, an ASR change can move a request across the downstream model's calibrated safety boundary. The resulting LLM-quality loss does not appear when ASR and LLM are benchmarked independently.

This is what "inter-stage coupling" means, and it's what the PAVO paper formalizes: the LLM's quality function is **conditionally defined** on the ASR's output distribution. Optimizing either in isolation will systematically mislead you.

## What you do about it

Once you accept the cliff, two things follow.

**First**, you can't pick "the right" ASR-LLM pair once at deployment time. The right pair depends on the turn. A quiet, high-SNR turn with a simple request can use a tiny ASR and a small LLM; a noisy turn with a complex request needs the big ASR *precisely to keep the LLM off the cliff.* You route per turn.

**Second**, the router needs to *know* about the cliff. A naive optimizer that sees only latency and cost will happily pick the small ASR for the noisy turn and watch the LLM collapse. You have to give it the coupling constraint explicitly.

In PAVO we train an 85,041-parameter policy-plus-value MLP. Its released input
is a 12-dimensional state and its output is a distribution over 48 analytic
action factors, with infeasible actions removed by hard logit masking.
Multi-objective PPO training completes in 106 seconds on an H100. The release
does not contain a concrete mapping from those 48 indices to deployable
ASR/LLM/TTS tuples, so the public wrapper exposes logits and requires a
deployment-specific resolver.

The camera-ready manuscript reports the following results against a fixed-cloud
baseline:

- **−10.3% P95 tail compression** (−167 ms on H100 / 200 LibriSpeech samples)
- **−34% median latency** (50K-turn benchmark)
- **−71% energy per turn** (50K-turn benchmark)
- **7.1% → 0.9% coherence-failure rate** (7.9× reduction via hard-constraint masking, +110 ms median latency cost)
- Quality parity on non-coupling-violating turns

The `p = 2×10⁻⁶` test concerns **mean** latency over five paired
catalog-simulation replications, not the descriptive P95 result; the paired
Wilcoxon test on those five replications gives `p = 0.0625`. Standalone raw
traces were not found for the manuscript's 34%/71%, coherence-failure, kappa,
or full acoustic-ablation aggregates; the provenance map states the exact
evidence scope.

## Reproduce it

The unmodified 50,000-row synthetic split, released result summaries,
controller checkpoints, and code are on GitHub. The dataset contains 1,743
literal `[TIMEOUT]` reference responses and two generation phases, both
recorded in `data/DATASET_AUDIT.json`. Code uses the MIT License; data, results,
and model weights use CC-BY 4.0:

```bash
pip install git+https://github.com/vnmoorthy/pavo-bench.git
git clone https://github.com/vnmoorthy/pavo-bench  # code, data, and experiments
```

Or, if you just want to see the result on a free-tier Colab in two minutes:

**[Open quickstart in Colab](https://colab.research.google.com/github/vnmoorthy/pavo-bench/blob/main/notebooks/quickstart.ipynb)**

The things I'd love feedback on:

1. The cliff's shape on model pairs I didn't test. File a reproduction report with your results.
2. The PPO reward design — we used a soft penalty for coupling violations; a constrained-RL formulation might be cleaner.
3. The benchmark generator. Complexity labels are heuristic; a learned labeler might change the numbers.

The paper was accepted at TMLR in 2026. If you build on PAVO-Bench, the `CITATION.cff` in the repo has a copy-paste citation.

---

*Written by NarasingaMoorthy VeiluKanthaPerumal (University of Pennsylvania) and Mohammed Imthathullah (Google). Questions or reproduction results: open an issue on the repo, or DM me.*
