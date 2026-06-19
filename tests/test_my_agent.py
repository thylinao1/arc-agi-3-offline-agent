"""Unit tests for agent/my_agent.py — pure perception + the BFS reset-replay search.

Run offline with numpy only (framework/SDK imports in my_agent.py are guarded).
"""
from __future__ import annotations

import numpy as np

from agent.my_agent import (
    ReplaySearch,
    frame_signature,
    grid_from_frame,
    identify_status_bars,
    priority_click_targets,
    segment_frame,
    status_bar_mask,
    volatility_mask,
)


def _two_blob_grid() -> np.ndarray:
    g = np.zeros((64, 64), dtype=np.int16)
    g[2:5, 2:5] = 3       # 3x3, color 3 (not salient), center (3, 3)
    g[10:18, 40:48] = 7   # 8x8, color 7 (salient), center (13, 43)
    return g


# ───────────────────────────── perception ─────────────────────────────
def test_segment_frame_finds_objects_and_background() -> None:
    segs = segment_frame(_two_blob_grid())
    assert len(segs) == 3  # background + two blobs
    by_color = {s.color: s for s in segs}
    assert by_color[7].area == 64 and by_color[7].bbox == (10, 40, 17, 47)
    assert by_color[3].area == 9 and by_color[3].center == (3, 3)


def test_priority_click_targets_salient_first_xy_and_dedup() -> None:
    cands = priority_click_targets(_two_blob_grid(), max_targets=10)
    assert cands[0] == (43, 13)          # salient 8x8 blob (tier 0), (x, y), floor center
    assert (3, 3) in cands               # color-3 blob (tier 1)
    assert len(cands) == len(set(cands))
    assert all(0 <= x < 64 and 0 <= y < 64 for x, y in cands)


def test_priority_click_targets_respects_limit() -> None:
    g = np.zeros((64, 64), dtype=np.int16)
    for r in range(6):
        for c in range(5):
            g[r * 3 : r * 3 + 2, c * 3 : c * 3 + 2] = 7  # 30 salient 2x2 blocks (area 4)
    assert len(priority_click_targets(g, max_targets=12)) == 12


def test_status_bar_mask_detects_thin_edge_bar() -> None:
    g = np.zeros((64, 64), dtype=np.int16)
    g[0:2, 5:45] = 8  # thin horizontal bar at the top edge (aspect ratio 20)
    g[30:34, 30:34] = 9  # a play object in the middle
    mask = status_bar_mask(g)
    assert mask is not None
    assert bool(mask[0, 5]) is True     # bar masked
    assert bool(mask[31, 31]) is False  # play object not masked


def test_identify_status_bars_catches_twins() -> None:
    g = np.zeros((64, 64), dtype=np.int16)
    for k in range(4):  # 4 identical "life" icons along the bottom edge
        g[62, 5 + k * 4] = 7
    segs = segment_frame(g)
    assert len(identify_status_bars(segs)) >= 4


def test_frame_signature_masks_counter() -> None:
    g = _two_blob_grid()
    mask = np.zeros((64, 64), dtype=bool)
    mask[0, 0] = True
    a, b = g.copy(), g.copy()
    b[0, 0] = 9  # differ only in the masked cell
    assert frame_signature(a, mask) == frame_signature(b, mask)
    assert frame_signature(a) != frame_signature(b)


def test_grid_from_frame_takes_last_grid() -> None:
    frame = [np.zeros((64, 64), dtype=int).tolist(), _two_blob_grid().tolist()]
    out = grid_from_frame(frame)
    assert out.shape == (64, 64) and int(out[10, 40]) == 7


def test_volatility_mask_isolates_counter_and_bails_on_animation() -> None:
    grids = []
    for t in range(6):
        f = np.zeros((4, 4), dtype=np.int16)
        f[0, 0] = t + 1                # counter changes every transition
        f[3, min(t // 3, 1)] = 9       # player moves once
        grids.append(f)
    m = volatility_mask(grids)
    assert m is not None and bool(m[0, 0]) and not bool(m[3, 0])
    busy = [np.full((4, 4), t, dtype=np.int16) for t in range(5)]
    assert volatility_mask(busy) is None


# ───────────────────────────── ReplaySearch (BFS) ─────────────────────────────
def run_sim(transitions, candidates, start="r", max_steps=400):
    s = ReplaySearch()
    cur, game_over, actions, resets = start, False, 0, 0
    for _ in range(max_steps):
        tok = s.step(cur, candidates.get(cur, []), game_over=game_over)
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


def test_search_solves_a_deep_path() -> None:
    trans = {
        ("r", "ACTION1"): ("d", "dead"),
        ("r", "ACTION2"): ("s1", "normal"),
        ("s1", "ACTION1"): ("s2", "normal"),
        ("s2", "ACTION1"): ("g", "win"),
    }
    cand = {"r": ["ACTION1", "ACTION2"], "s1": ["ACTION1"], "s2": ["ACTION1"], "d": []}
    solved, actions, _ = run_sim(trans, cand)
    assert solved and actions < 60


def test_search_corridor_uses_no_resets() -> None:
    trans = {("r", "ACTION1"): ("a", "normal"), ("a", "ACTION1"): ("b", "normal"),
             ("b", "ACTION1"): ("g", "win")}
    cand = {"r": ["ACTION1"], "a": ["ACTION1"], "b": ["ACTION1"]}
    solved, actions, resets = run_sim(trans, cand)
    assert solved and resets == 0 and actions == 3


def test_search_finds_shallow_win_via_breadth() -> None:
    # The win is the SECOND action from root; BFS must try it without diving via the first.
    trans = {
        ("r", "ACTION1"): ("a", "normal"),
        ("a", "ACTION1"): ("a", "normal"),  # first action leads to a dead loop
        ("r", "ACTION2"): ("g", "win"),
    }
    cand = {"r": ["ACTION1", "ACTION2"], "a": ["ACTION1"]}
    solved, _actions, _ = run_sim(trans, cand)
    assert solved


def test_search_reports_unsolvable_without_crashing() -> None:
    trans = {("r", "ACTION1"): ("r", "normal")}
    cand = {"r": ["ACTION1"]}
    solved, _a, _r = run_sim(trans, cand, max_steps=50)
    assert solved is False


def test_start_level_resets_state() -> None:
    s = ReplaySearch()
    s.step("r", ["ACTION1"])
    s.start_level("root2", ["ACTION1", "ACTION2"])
    assert s.root == "root2" and s.known == {"root2": []} and s.terminal == set()


def test_step_modulus_keys_by_depth() -> None:
    s = ReplaySearch(step_modulus=2)
    s._pending_depth = 0
    assert s._key("abc") == "abc#0"
    s._pending_depth = 3
    assert s._key("abc") == "abc#1"
    assert ReplaySearch()._key("abc") == "abc"  # default (mod=1) is a no-op


def test_dead_simple_pruning_when_enabled() -> None:
    s = ReplaySearch(prune_dead_simples=True)
    for _ in range(s.DEAD_AFTER):
        s._record_simple("ACTION3", changed=False)
    assert s._dead("ACTION3") is True            # ineffective simple is pruned
    s._record_simple("ACTION1", changed=True)
    assert s._dead("ACTION1") is False           # effective simple kept
    assert s._dead("ACTION6:1,2") is False       # clicks are never globally pruned
    assert ReplaySearch()._dead("ACTION3") is False  # default OFF (coverage-first)


def test_search_still_solves_with_pruning_enabled() -> None:
    # ACTION3 is a global no-op (ineffective everywhere); the win needs ACTION1 x3.
    trans = {
        ("r", "ACTION1"): ("a", "normal"), ("r", "ACTION3"): ("r", "normal"),
        ("a", "ACTION1"): ("b", "normal"), ("a", "ACTION3"): ("a", "normal"),
        ("b", "ACTION1"): ("g", "win"), ("b", "ACTION3"): ("b", "normal"),
    }
    cand = {k: ["ACTION1", "ACTION3"] for k in ("r", "a", "b")}
    solved, _a, _r = run_sim(trans, cand)
    assert solved
