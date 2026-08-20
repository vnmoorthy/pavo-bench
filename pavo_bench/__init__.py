"""
PAVO-Bench: a 50K-turn voice pipeline benchmark + pretrained router.

Checkpoint inspection:

    from pavo_bench import (
        load_dataset, PretrainedPAVORouter,
    )

    turns = load_dataset(split="test")          # 10K test turns
    pavo = PretrainedPAVORouter.from_released()
    print(pavo.action_logits(turns[0]).shape)  # torch.Size([48])

The release does not include the mapping from 48 analytic actions to concrete
deployment tuples, so the pretrained wrapper does not pretend to implement a
three-profile router. Reference and custom routers remain benchmarkable.

Evaluate your own routing strategy by subclassing BaseRouter:

    from pavo_bench import BaseRouter, Profile, benchmark_router

    class MyRouter(BaseRouter):
        def route(self, turn) -> Profile:
            # decide per-turn; return one of: "cloud_premium", "ondevice_fast",
            # "hybrid_balanced".
            return "hybrid_balanced" if turn.complexity >= 3 else "ondevice_fast"

    print(benchmark_router(MyRouter(), turns))
"""

from .coupling import reproduce_coupling_cliff
from .dataset import PAVOBenchTurn, load_dataset
from .evaluate import BenchmarkResult, benchmark_router
from .loader import load_pretrained
from .model import MetaController
from .routers import (
    AlwaysCloudRouter,
    AlwaysEdgeRouter,
    BaseRouter,
    HybridRouter,
    PretrainedPAVORouter,
    Profile,
    RandomRouter,
)
from .state import turn_to_state_vector

__all__ = [
    "AlwaysCloudRouter",
    "AlwaysEdgeRouter",
    "BaseRouter",
    "BenchmarkResult",
    "HybridRouter",
    "MetaController",
    "PAVOBenchTurn",
    "PretrainedPAVORouter",
    "Profile",
    "RandomRouter",
    "benchmark_router",
    "load_dataset",
    "load_pretrained",
    "reproduce_coupling_cliff",
    "turn_to_state_vector",
]

__version__ = "1.0.1"
