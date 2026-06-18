"""RHAE scoring + coverage/variance helpers.

Ported from `vendor/occam/solver/rhae.py` (the model we optimize against) with the
official 1.15 cap exposed. Dev/eval only — not part of the spliced agent.
"""
from __future__ import annotations

import math
from typing import Sequence

OFFICIAL_CAP = 1.15  # per the ARC-AGI-3 technical report; occam used 1.0 conservatively


def level_score(ai_actions: int, human_baseline: int, cap: float = OFFICIAL_CAP) -> float:
    """min(cap, (human_baseline / ai_actions)^2). ai_actions is cumulative (incl. resets)."""
    if ai_actions <= 0:
        return 0.0
    ratio = human_baseline / ai_actions
    return min(cap, ratio * ratio)


def game_score(level_scores: Sequence[float], total_levels: int) -> float:
    """1-indexed level-weighted average: sum((i+1)*S_i) / sum(1..total_levels)."""
    if not level_scores or total_levels <= 0:
        return 0.0
    weight_sum = sum(range(1, total_levels + 1))
    weighted = sum((i + 1) * s for i, s in enumerate(level_scores))
    return weighted / weight_sum


def total_score(game_scores: Sequence[float]) -> float:
    """Average of per-game scores across the (hidden) game set → 0..1."""
    return sum(game_scores) / len(game_scores) if game_scores else 0.0


def giveup_budget(human_baseline: int, n_actions: int = 4) -> int:
    """≈ 5x human baseline. Solve before this many actions or the level scores 0."""
    min_budget = n_actions + 30
    multiplier = max(5.0, (n_actions / max(human_baseline, 1)) + 3.0)
    return max(int(human_baseline * multiplier), min_budget)


def coverage(game_scores: Sequence[float]) -> float:
    """PRIMARY objective: fraction of games scored > 0 (lower-tail coverage)."""
    return sum(1 for s in game_scores if s > 0.0) / len(game_scores) if game_scores else 0.0


def bootstrap_std(game_scores: Sequence[float], iters: int = 2000, seed: int = 0) -> float:
    """Deterministic bootstrap estimate of the std of the mean score over games.

    Used to enforce the "ship a component only if delta > variance" gate on the frozen
    holdout. Seeded for reproducibility.
    """
    import random

    n = len(game_scores)
    if n == 0:
        return 0.0
    rng = random.Random(seed)
    means: list[float] = []
    scores = list(game_scores)
    for _ in range(iters):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    mean = sum(means) / len(means)
    var = sum((m - mean) ** 2 for m in means) / len(means)
    return math.sqrt(var)
