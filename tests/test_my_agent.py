"""Unit tests for the pure helpers in agent/my_agent.py.

These run offline with only numpy (the framework/SDK imports are guarded), proving the
perception + exploration logic deterministically.
"""
from __future__ import annotations

import numpy as np

from agent.my_agent import (
    FrontierPolicy,
    click_candidates,
    connected_components,
    frame_signature,
    grid_from_frame,
)


def _two_blob_grid() -> np.ndarray:
    g = np.zeros((64, 64), dtype=np.int16)
    g[2:5, 2:5] = 3      # 3x3 blob of color 3, centroid ~ (3,3)
    g[10:18, 40:48] = 7  # 8x8 blob of color 7 (bigger), centroid ~ (13,43)
    return g


def test_connected_components_counts_and_orders_by_size() -> None:
    comps = connected_components(_two_blob_grid())
    assert len(comps) == 2
    # Sorted by descending size → the 8x8 (color 7) comes first.
    assert comps[0]["color"] == 7 and comps[0]["size"] == 64
    assert comps[1]["color"] == 3 and comps[1]["size"] == 9
    assert comps[1]["centroid"] == (3, 3)


def test_connected_components_ignores_background() -> None:
    assert connected_components(np.zeros((64, 64), dtype=np.int16)) == []


def test_click_candidates_dedup_limit_and_xy_order() -> None:
    cands = click_candidates(_two_blob_grid(), limit=10)
    # (x, y) order; biggest object first. Blob rows 10..17 / cols 40..47 →
    # centroid (round(13.5), round(43.5)) = (14, 44) → (x, y) = (44, 14).
    assert cands[0] == (44, 14)
    assert (3, 3) in cands
    assert len(cands) == len(set(cands))  # deduped
    assert all(0 <= x < 64 and 0 <= y < 64 for x, y in cands)


def test_click_candidates_respects_limit() -> None:
    g = np.zeros((64, 64), dtype=np.int16)
    for i in range(30):  # 30 isolated single cells → 30 components
        g[i * 2, 0] = (i % 15) + 1
    assert len(click_candidates(g, limit=24)) == 24


def test_frame_signature_is_stable_and_distinct() -> None:
    g1 = _two_blob_grid()
    g2 = g1.copy()
    g2[0, 0] = 5
    assert frame_signature(g1) == frame_signature(g1.copy())
    assert frame_signature(g1) != frame_signature(g2)


def test_grid_from_frame_takes_last_grid() -> None:
    frame = [np.zeros((64, 64), dtype=int).tolist(), _two_blob_grid().tolist()]
    out = grid_from_frame(frame)
    assert out.shape == (64, 64)
    assert int(out[10, 40]) == 7


def test_frontier_policy_prefers_effective_then_clicks_then_resets() -> None:
    p = FrontierPolicy()
    p.effective.add("ACTION4")  # known-effective
    simple = ["ACTION1", "ACTION2", "ACTION4"]
    clicks = ["ACTION6:5,5"]
    # 1) effective simple first
    assert p.select("sigA", simple, clicks) == "ACTION4"
    # 2) then remaining simple (sorted), 3) then clicks
    got = {p.select("sigA", simple, clicks) for _ in range(3)}
    assert got == {"ACTION1", "ACTION2", "ACTION6:5,5"}
    # 4) exhausted → RESET to re-branch
    assert p.select("sigA", simple, clicks) == "RESET"
    assert p.consecutive_exhausted == 1


def test_frontier_policy_observe_marks_effective_transitions() -> None:
    p = FrontierPolicy()
    p.select("s0", ["ACTION1"], [])  # tries ACTION1 from s0
    p.observe("s1")  # frame changed → ACTION1 effective
    assert "ACTION1" in p.effective
    assert p.transitions[("s0", "ACTION1")] == "s1"


def test_frontier_policy_level_change_resets_per_level_state() -> None:
    p = FrontierPolicy()
    p.select("s0", ["ACTION1"], [])
    p.effective.add("ACTION1")
    p.on_level_change(1)
    assert p.tried == {} and p.transitions == {} and p.level == 1
    assert "ACTION1" in p.effective  # effective actions are banked across levels
