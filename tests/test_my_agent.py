"""Unit tests for agent/my_agent.py.

Pure helpers (numpy only) + the ReplayDFS solver driven against tiny deterministic
simulated environments. The framework/SDK imports in my_agent.py are guarded, so these
run fully offline.
"""
from __future__ import annotations

import numpy as np

from agent.my_agent import (
    ReplayDFS,
    click_candidates,
    connected_components,
    frame_signature,
    grid_from_frame,
    volatility_mask,
)


# ───────────────────────────── perception ─────────────────────────────
def _two_blob_grid() -> np.ndarray:
    g = np.zeros((64, 64), dtype=np.int16)
    g[2:5, 2:5] = 3      # 3x3 blob, centroid (3, 3)
    g[10:18, 40:48] = 7  # 8x8 blob (bigger), centroid (14, 44)
    return g


def test_connected_components_counts_and_orders_by_size() -> None:
    comps = connected_components(_two_blob_grid())
    assert len(comps) == 2
    assert comps[0]["color"] == 7 and comps[0]["size"] == 64
    assert comps[1]["color"] == 3 and comps[1]["size"] == 9
    assert comps[1]["centroid"] == (3, 3)


def test_connected_components_ignores_background() -> None:
    assert connected_components(np.zeros((64, 64), dtype=np.int16)) == []


def test_click_candidates_dedup_limit_and_xy_order() -> None:
    cands = click_candidates(_two_blob_grid(), limit=10)
    # (x, y) order; biggest object first → centroid (14, 44) → (x, y) = (44, 14).
    assert cands[0] == (44, 14)
    assert (3, 3) in cands
    assert len(cands) == len(set(cands))
    assert all(0 <= x < 64 and 0 <= y < 64 for x, y in cands)


def test_click_candidates_respects_limit() -> None:
    g = np.zeros((64, 64), dtype=np.int16)
    for i in range(30):
        g[i * 2, 0] = (i % 15) + 1
    assert len(click_candidates(g, limit=16)) == 16


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


# ───────────────────────────── ReplayDFS ─────────────────────────────
def run_sim(transitions, candidates, start="r", max_steps=300):
    """Drive ReplayDFS against a deterministic graph.

    transitions: {(sig, token): (next_sig, kind)} where kind ∈ {normal, dead, win}.
    Returns (solved: bool, env_actions: int, resets: int).
    """
    dfs = ReplayDFS()
    cur, game_over, actions, resets = start, False, 0, 0
    for _ in range(max_steps):
        tok = dfs.step(cur, candidates.get(cur, []), game_over=game_over)
        actions += 1
        game_over = False
        if tok == "RESET":
            resets += 1
            cur = start
            continue
        nxt, kind = transitions[(cur, tok)]
        if kind == "win":
            return True, actions, resets
        if kind == "dead":
            cur, game_over = nxt, True
        else:
            cur = nxt
    return False, actions, resets


def test_dfs_solves_a_deep_path() -> None:
    # r -A2-> s1 -A1-> s2 -A1-> WIN ; A1 from r is a trap.
    trans = {
        ("r", "ACTION1"): ("d", "dead"),
        ("r", "ACTION2"): ("s1", "normal"),
        ("r", "ACTION3"): ("r", "normal"),
        ("s1", "ACTION1"): ("s2", "normal"),
        ("s1", "ACTION2"): ("r", "normal"),
        ("s2", "ACTION1"): ("g", "win"),
    }
    cand = {
        "r": ["ACTION1", "ACTION2", "ACTION3"],
        "s1": ["ACTION1", "ACTION2"],
        "s2": ["ACTION1"],
        "d": [],
    }
    solved, actions, resets = run_sim(trans, cand)
    assert solved
    assert actions < 30  # found quickly, not via brute-force blowup


def test_dfs_descends_without_needless_resets() -> None:
    # A straight corridor r->a->b->WIN; DFS should follow it with ZERO resets.
    trans = {
        ("r", "ACTION1"): ("a", "normal"),
        ("a", "ACTION1"): ("b", "normal"),
        ("b", "ACTION1"): ("g", "win"),
    }
    cand = {"r": ["ACTION1"], "a": ["ACTION1"], "b": ["ACTION1"]}
    solved, actions, resets = run_sim(trans, cand)
    assert solved and resets == 0 and actions == 3


def test_dfs_backtracks_after_dead_end() -> None:
    # r -A1-> a (a loops on itself, no win) ; must backtrack and try r -A2-> WIN.
    trans = {
        ("r", "ACTION1"): ("a", "normal"),
        ("a", "ACTION1"): ("a", "normal"),
        ("r", "ACTION2"): ("g", "win"),
    }
    cand = {"r": ["ACTION1", "ACTION2"], "a": ["ACTION1"]}
    solved, _actions, _resets = run_sim(trans, cand)
    assert solved


def test_dfs_reports_unsolvable_without_crashing() -> None:
    # No win reachable; the search must terminate the budget gracefully.
    trans = {("r", "ACTION1"): ("r", "normal")}
    cand = {"r": ["ACTION1"]}
    solved, _actions, _resets = run_sim(trans, cand, max_steps=50)
    assert solved is False


def test_start_level_resets_state() -> None:
    dfs = ReplayDFS()
    dfs.step("r", ["ACTION1"])
    dfs.start_level("root2", ["ACTION1", "ACTION2"])
    assert dfs.root == "root2" and dfs.known == {"root2": []} and dfs.terminal == set()


# ───────────────────────────── volatility mask ─────────────────────────────
def test_volatility_mask_isolates_a_counter() -> None:
    grids = []
    for t in range(6):
        f = np.zeros((4, 4), dtype=np.int16)
        f[0, 0] = t + 1                 # counter: changes EVERY transition
        f[3, min(t // 3, 1)] = 9        # "player": moves only once across the run
        grids.append(f)
    m = volatility_mask(grids)
    assert m is not None
    assert bool(m[0, 0]) is True        # counter masked
    assert bool(m[3, 0]) is False       # player area not always-changing
    a, b = grids[0].copy(), grids[0].copy()
    b[0, 0] = 99                         # differ only in the counter
    assert frame_signature(a, m) == frame_signature(b, m)


def test_volatility_mask_bails_when_play_area_animates() -> None:
    grids = [np.full((4, 4), t, dtype=np.int16) for t in range(5)]
    assert volatility_mask(grids) is None  # whole grid volatile → do not mask
