"""benchmark_router — aggregate a router's per-turn choices into a metrics dict."""
from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from . import _profile_costs as pc
from .dataset import PAVOBenchTurn
from .routers import VALID_PROFILES, BaseRouter


@dataclass
class BenchmarkResult:
    """Aggregate metrics for a router over a PAVO-Bench split."""

    router: str
    n_turns: int
    latency_ms_mean: float
    latency_ms_std:  float
    latency_ms_p50:  float
    latency_ms_p95:  float
    quality_mean:    float
    cost_usd_mean:   float
    energy_mj_mean:  float
    coupling_violations: int
    infeasible_pct: float
    profile_distribution: dict

    def as_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult({self.router}, n={self.n_turns}, "
            f"P95 latency={self.latency_ms_p95:.0f} ms, "
            f"quality={self.quality_mean:.3f}, "
            f"energy={self.energy_mj_mean:.1f} mJ, "
            f"profiles={self.profile_distribution})"
        )


def _sample_latency(profile: str, rng: random.Random) -> float:
    """Sample a per-turn latency from the profile's committed mean/std."""
    prior = pc.LATENCY_MS[profile]
    # Truncated Gaussian, clamped at 0.
    x = rng.gauss(prior["mean"], prior["std"])
    return max(x, 0.0)


def benchmark_router(
    router: BaseRouter,
    turns: Iterable[PAVOBenchTurn],
    *,
    seed: int = 0,
) -> BenchmarkResult:
    """Evaluate a router over a PAVO-Bench split.

    The convenience simulator samples per-turn latency from aggregate priors
    and looks up quality, cost, and energy per public profile. It is useful for
    API smoke tests and comparisons within this simulator, but it is not a
    replay of the released PPO evaluation and its output must not be presented
    as reproduction of the camera-ready headline table. See
    docs/RESULT_PROVENANCE.md for the evidence attached to each paper result.
    """
    rng = random.Random(seed)
    latencies: list[float] = []
    quality: list[float] = []
    cost:    list[float] = []
    energy:  list[float] = []
    infeasible = 0
    violations = 0
    dist = {p: 0 for p in VALID_PROFILES}

    turns = list(turns)
    for turn in turns:
        profile = router(turn)
        dist[profile] += 1

        if pc.infeasible_for_turn(profile, turn.complexity):
            infeasible += 1
            violations += 1

        latencies.append(_sample_latency(profile, rng))
        quality.append(pc.QUALITY[profile])
        cost.append(pc.COST_USD[profile])
        energy.append(pc.ENERGY_MJ[profile])

    arr = np.asarray(latencies, dtype=np.float64)
    dist_norm = {k: v / max(len(turns), 1) for k, v in dist.items()}

    return BenchmarkResult(
        router=router.name,
        n_turns=len(turns),
        latency_ms_mean=float(arr.mean()),
        latency_ms_std=float(arr.std()),
        latency_ms_p50=float(np.percentile(arr, 50)),
        latency_ms_p95=float(np.percentile(arr, 95)),
        quality_mean=float(np.mean(quality)),
        cost_usd_mean=float(np.mean(cost)),
        energy_mj_mean=float(np.mean(energy)),
        coupling_violations=int(violations),
        infeasible_pct=100.0 * infeasible / max(len(turns), 1),
        profile_distribution=dist_norm,
    )
