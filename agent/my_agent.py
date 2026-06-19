"""ARC-AGI-3 offline agent — self-contained (this whole file is spliced into the
Kaggle submission notebook, so it may import ONLY: stdlib, numpy, arcengine,
agents.agent).

Strategy (see SPEC.md / CONTRACT.md / DECISION-LOG.md):
  - No model in the loop. Deterministic, symbolic, offline. Coverage first.
  - Perceive the 64x64 grid as colored objects (connected components, no scipy).
  - Mask counter / status-bar pixels so state de-dup works (geometric detection of
    edge-hugging thin/twinned segments, ported from occam/Just-Explore, plus a
    volatility fallback). Without this, a per-frame counter makes every frame look new.
  - Rank actions: effective actions first (ones that change the frame), clicks by visual
    salience tier (chromatic, medium objects before background / status bars).
  - Search by BREADTH-first reset-and-replay (occam's technique): try all of a state's
    actions before going deeper, so shallow wins (a single meaningful click/sequence) are
    found early. The engine is deterministic (confirmed), so any state is revisitable by
    RESET + replaying its action prefix; incremental replay skips shared prefixes.
  - Level-aware (RESET = level restart, completed levels kept). Deterministic throughout.
"""
from __future__ import annotations

import hashlib
import os
from collections import deque
from itertools import product
from typing import Any, Optional

import numpy as np

_DEBUG = bool(os.environ.get("ARC_DEBUG"))

try:  # pragma: no cover
    from arcengine import FrameData, GameAction, GameState  # type: ignore
except Exception:  # pragma: no cover
    FrameData = Any  # type: ignore
    GameAction = None  # type: ignore
    GameState = None  # type: ignore

try:  # pragma: no cover
    from agents.agent import Agent  # type: ignore
except Exception:  # pragma: no cover
    Agent = object  # type: ignore

GRID_SIZE = 64
RESET = "RESET"
MAX_CLICK_TARGETS = 24

# Salience / status-bar heuristics (from occam priority_tiers.py).
SALIENT_COLORS = frozenset(range(6, 16))
STATUS_BAR_EDGE_DIST = 3
STATUS_BAR_ASPECT_RATIO = 5
STATUS_BAR_MIN_TWINS = 3
SEG_MIN_DIM, SEG_MAX_DIM = 2, 32


# ───────────────────────────── perception (testable) ─────────────────────────────
def grid_from_frame(frame: Any) -> np.ndarray:
    if frame is None:
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int16)
    arr = np.asarray(frame[-1] if len(frame) else frame, dtype=np.int16)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1]) if arr.ndim >= 2 else arr
    return arr


def frame_signature(grid: np.ndarray, mask: Optional[np.ndarray] = None) -> str:
    """Deterministic short hash; masked cells are zeroed so counters don't fragment state."""
    if mask is not None:
        grid = np.where(mask, np.int16(0), grid)
    return hashlib.md5(np.ascontiguousarray(grid).tobytes()).hexdigest()[:16]


class Segment:
    """A connected component. Plain class (no @dataclass) so it loads under importlib."""

    __slots__ = ("color", "bbox", "area", "pixels")

    def __init__(
        self, color: int, bbox: tuple[int, int, int, int], area: int, pixels: list[tuple[int, int]]
    ) -> None:
        self.color = color
        self.bbox = bbox  # (y1, x1, y2, x2)
        self.area = area
        self.pixels = pixels

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def center(self) -> tuple[int, int]:  # (y, x)
        return ((self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2)

    @property
    def is_salient(self) -> bool:
        return self.color in SALIENT_COLORS

    @property
    def is_medium(self) -> bool:
        return SEG_MIN_DIM <= self.width <= SEG_MAX_DIM and SEG_MIN_DIM <= self.height <= SEG_MAX_DIM

    @property
    def aspect_ratio(self) -> float:
        return max(self.width, self.height) / max(1, min(self.width, self.height))


def segment_frame(grid: np.ndarray) -> list[Segment]:
    """4-connected flood-fill segmentation over ALL colors (incl. background)."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    segments: list[Segment] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pixels: list[tuple[int, int]] = []
            y1 = y2 = r
            x1 = x2 = c
            while stack:
                cr, cc = stack.pop()
                pixels.append((cr, cc))
                y1, y2 = min(y1, cr), max(y2, cr)
                x1, x2 = min(x1, cc), max(x2, cc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            segments.append(Segment(color, (y1, x1, y2, x2), len(pixels), pixels))
    return segments


def identify_status_bars(segments: list[Segment], frame_size: int = GRID_SIZE) -> set[int]:
    """Indices of edge-hugging segments that are thin (a bar) or twinned (repeated icons)."""
    twin_counts: dict[tuple[int, int], int] = {}
    for s in segments:
        twin_counts[(s.area, s.color)] = twin_counts.get((s.area, s.color), 0) + 1
    out: set[int] = set()
    for i, s in enumerate(segments):
        near_edge = (
            s.bbox[0] < STATUS_BAR_EDGE_DIST
            or s.bbox[2] >= frame_size - STATUS_BAR_EDGE_DIST
            or s.bbox[1] < STATUS_BAR_EDGE_DIST
            or s.bbox[3] >= frame_size - STATUS_BAR_EDGE_DIST
        )
        if not near_edge:
            continue
        if s.aspect_ratio >= STATUS_BAR_ASPECT_RATIO or twin_counts[(s.area, s.color)] >= STATUS_BAR_MIN_TWINS:
            out.add(i)
    return out


def status_bar_mask(grid: np.ndarray) -> Optional[np.ndarray]:
    """Boolean mask of pixels belonging to detected status-bar / counter segments."""
    segments = segment_frame(grid)
    ids = identify_status_bars(segments)
    if not ids:
        return None
    mask = np.zeros(grid.shape, dtype=bool)
    for i in ids:
        for (y, x) in segments[i].pixels:
            mask[y, x] = True
    return mask


def priority_click_targets(grid: np.ndarray, max_targets: int = MAX_CLICK_TARGETS) -> list[tuple[int, int]]:
    """(x, y) click targets ordered by salience tier (salient-medium first; bars last)."""
    segments = segment_frame(grid)
    status = identify_status_bars(segments)
    tiers: dict[int, list[Segment]] = {t: [] for t in range(5)}
    for i, s in enumerate(segments):
        if i in status:
            tiers[4].append(s)
        elif s.is_salient and s.is_medium:
            tiers[0].append(s)
        elif s.is_medium:
            tiers[1].append(s)
        elif s.is_salient:
            tiers[2].append(s)
        else:
            tiers[3].append(s)
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for t in range(5):
        for s in sorted(tiers[t], key=lambda s: (-s.area, s.center)):
            if s.area < 2:
                continue
            cy, cx = s.center
            xy = (int(cx), int(cy))
            if xy not in seen:
                seen.add(xy)
                out.append(xy)
            if len(out) >= max_targets:
                return out
    return out


def volatility_mask(grids: list[np.ndarray], max_fraction: float = 0.10) -> Optional[np.ndarray]:
    """Cells that change on EVERY warmup transition (a counter), if that region is small."""
    if len(grids) < 2:
        return None
    stack = np.stack(grids)
    always = (stack[1:] != stack[:-1]).all(axis=0)
    n = int(always.sum())
    if 0 < n <= max_fraction * always.size:
        return always
    return None


def combine_masks(*masks: Optional[np.ndarray]) -> Optional[np.ndarray]:
    out: Optional[np.ndarray] = None
    for m in masks:
        if m is None:
            continue
        out = m.copy() if out is None else (out | m)
    return out


# ─────────────────────── breadth-first reset-and-replay search ──────────────────────────
def _is_prefix(p: list[str], q: list[str]) -> bool:
    return len(p) <= len(q) and q[: len(p)] == p


class ReplaySearch:
    """Step-wise breadth-first reset-and-replay search over action tokens.

    Tokens: "ACTION1".."ACTION5" and "ACTION6:x,y"; "RESET" reserved. BFS expands a
    state's untried tokens (effective ones first) before moving to the next discovered
    state, so shallow wins are found early. Repositioning to a target replays its prefix,
    incrementally when the current state is an ancestor, else RESET + full replay.

    Efficiency (occam-ported):
      - Effective-action pruning: a simple action proven ineffective from several states is
        dropped globally, shrinking the branching factor (clicks stay per-state).
      - max_unique_states: stop enqueuing new states past a cap (memory/runaway guard).
      - step_modulus: when > 1, the node key is augmented with (depth % step_modulus) so a
        same-looking frame at a different phase of a cycle is a distinct state. Default 1
        (no-op) — it needs per-game periodicity to help, so it is off unless set.
    """

    DEAD_AFTER = 3  # ineffective probes from distinct states before a simple is pruned

    def __init__(
        self,
        step_modulus: int = 1,
        max_unique_states: int = 200_000,
        prune_dead_simples: bool = False,
    ) -> None:
        # NOTE: prune_dead_simples defaults OFF. Empirically (the public-set sweep) global
        # simple-action pruning costs COVERAGE — some directional actions are effective only
        # in later states, and coverage (solve rate) is the primary objective. The lever is
        # kept for games/configs where action-efficiency matters more than coverage.
        self.step_modulus = max(1, int(step_modulus))
        self.max_states = max_unique_states
        self.prune_dead_simples = prune_dead_simples
        self.effective: set[str] = set()
        self.simple_tries: dict[str, int] = {}
        self.simple_effects: dict[str, int] = {}
        self.start_level("", [])
        self.root = None  # type: Optional[str]

    def start_level(self, sig: str, candidates: list[str]) -> None:
        self._pending_depth = 0
        key = self._key(sig) if sig else sig
        self.root = key or None
        self.known: dict[str, list[str]] = {key: []} if sig else {}
        self.cand: dict[str, list[str]] = {key: list(candidates)} if sig else {}
        self.tried: dict[str, set[str]] = {}
        self.terminal: set[str] = set()
        self.queue: list[str] = [key] if sig else []
        self.qi = 0
        self.expanding: Optional[str] = key or None
        self.plan: list[str] = []
        self.awaiting: Optional[tuple[str, str]] = None
        self.cur: Optional[str] = key or None

    def note_reset(self) -> None:
        self.plan = []
        self.awaiting = None
        self.cur = None
        self._pending_depth = 0

    def _key(self, base: str) -> str:
        if self.step_modulus <= 1:
            return base
        return f"{base}#{self._pending_depth % self.step_modulus}"

    def _dead(self, tok: str) -> bool:
        if not self.prune_dead_simples or tok.startswith("ACTION6:"):
            return False
        return self.simple_tries.get(tok, 0) >= self.DEAD_AFTER and self.simple_effects.get(tok, 0) == 0

    def step(self, sig: str, candidates: list[str], game_over: bool = False) -> str:
        if self.root is None:
            self.start_level(sig, candidates)
            return self._next()

        key = self._key(sig)

        if game_over:
            if self.awaiting is not None:
                e, t = self.awaiting
                if key not in self.known:
                    self.known[key] = self.known[e] + [t]
                self.terminal.add(key)
                self._record_simple(t, changed=True)  # death is a state change
                self.awaiting = None
            self.plan = []
            self.cur = None
            self._pending_depth = 0
            return RESET

        if self.awaiting is not None:
            e, t = self.awaiting
            changed = key != e
            if key not in self.known:
                self.known[key] = self.known[e] + [t]
                self.cand[key] = list(candidates)
                if len(self.known) <= self.max_states:
                    self.queue.append(key)
            if changed:
                self.effective.add(t)
            self._record_simple(t, changed)
            self.awaiting = None
            self.cur = key
        else:
            self.cur = key
            self.cand.setdefault(key, list(candidates))

        if self.plan:
            return self._emit()
        return self._next()

    def _record_simple(self, tok: str, changed: bool) -> None:
        if tok.startswith("ACTION6:"):
            return
        self.simple_tries[tok] = self.simple_tries.get(tok, 0) + 1
        if changed:
            self.simple_effects[tok] = self.simple_effects.get(tok, 0) + 1

    def _emit(self) -> str:
        tok = self.plan.pop(0)
        if not self.plan and self.expanding is not None:
            self.awaiting = (self.expanding, tok)
        self._pending_depth = 0 if tok == RESET else self._pending_depth + 1
        return tok

    def _untried(self, sig: str) -> list[str]:
        done = self.tried.setdefault(sig, set())
        ut = [c for c in self.cand.get(sig, []) if c not in done and not self._dead(c)]
        return [c for c in ut if c in self.effective] + [c for c in ut if c not in self.effective]

    def _next(self) -> str:
        while True:
            if self.expanding is not None and self.expanding not in self.terminal and self._untried(self.expanding):
                break
            self.qi += 1
            if self.qi >= len(self.queue):
                self.expanding = None
                self._pending_depth = 0
                return RESET
            self.expanding = self.queue[self.qi]

        token = self._untried(self.expanding)[0]
        self.tried[self.expanding].add(token)
        target = self.known[self.expanding]
        if self.cur == self.expanding:
            self.plan = [token]
        elif self.cur is not None and _is_prefix(self.known.get(self.cur, ["x"]), target):
            self.plan = target[len(self.known[self.cur]):] + [token]
        else:
            self.plan = [RESET] + target + [token]
        return self._emit()


# ─────────────────────── reactive navigation solver (testable) ──────────────────────────
class ReactiveNav:
    """occam's reactive cursor→target solver, ported.

    Probe: from the level root, apply each simple action once and detect a moving "cursor"
    object and which action drives which direction. Reactive: each turn, move the cursor
    toward the nearest rare-colored target and interact on arrival. Built for movement games;
    the agent falls back to BFS when no cursor is found or the cursor stalls.
    """

    _SEMANTIC = {"ACTION1": "up", "ACTION2": "down", "ACTION3": "left", "ACTION4": "right"}

    def __init__(self) -> None:
        self.bg: Optional[int] = None
        self.cursor_color: Optional[int] = None
        self.arrow_map: dict[str, str] = {}  # direction -> action name
        self.interact: Optional[str] = None
        self.step_sizes: list[float] = []
        self.probed: dict[str, str] = {}     # action -> probed direction
        self.moved: set[str] = set()         # actions confirmed to move the cursor

    def probe(self, root: np.ndarray, result: np.ndarray, action: str) -> None:
        diff = root != result
        if not diff.any():
            return
        if self.bg is None:
            vals, counts = np.unique(root, return_counts=True)
            self.bg = int(vals[counts.argmax()])
        cand = ({int(v) for v in root[diff]} | {int(v) for v in result[diff]}) - {self.bg, 0}
        best_score, best, bdy, bdx = -1.0, None, 0.0, 0.0
        for c in cand:
            p0 = np.argwhere(root == c)
            p1 = np.argwhere(result == c)
            if len(p0) == 0 or len(p1) == 0:
                continue
            c0, c1 = p0.mean(0), p1.mean(0)
            dy, dx = float(c1[0] - c0[0]), float(c1[1] - c0[1])
            score = (abs(dy) + abs(dx)) / (1.0 + len(p0))  # small object moving far = cursor
            if score > best_score:
                best_score, best, bdy, bdx = score, c, dy, dx
        if best is None or best_score <= 0.01:
            return
        if self.cursor_color is None:
            self.cursor_color = best
        if self.cursor_color == best:
            d = ("up" if bdy < 0 else "down") if abs(bdy) > abs(bdx) else ("left" if bdx < 0 else "right")
            self.probed[action] = d
            self.moved.add(action)
            self.step_sizes.append(abs(bdy) + abs(bdx))

    def finalize(self, simples: list[str]) -> None:
        # Trust probed directions; fill gaps from the ACTION1-4 semantic convention so a
        # direction blocked at the root (never seen moving) is still usable for pathfinding.
        directionals = [a for a in simples if a in self._SEMANTIC]
        self.arrow_map = {}
        for a in directionals:
            if a in self.probed:
                self.arrow_map[self.probed[a]] = a
        for a in directionals:
            d = self._SEMANTIC[a]
            if d not in self.arrow_map and a not in self.arrow_map.values():
                self.arrow_map[d] = a
        used = set(self.arrow_map.values())
        if "ACTION5" in simples and "ACTION5" not in used:
            self.interact = "ACTION5"
        else:
            self.interact = next((a for a in simples if a not in used), None)

    def ready(self) -> bool:
        # Require ≥2 CONFIRMED cursor moves so non-movement games don't false-positive.
        return self.cursor_color is not None and len(self.moved) >= 2 and len(self.arrow_map) >= 2

    def candidate_targets(self, grid: np.ndarray, limit: int = 6) -> list[int]:
        """Distinct non-background, non-cursor colors, rarest first — goal hypotheses."""
        vals, counts = np.unique(grid, return_counts=True)
        cands = [(int(v), int(n)) for v, n in zip(vals, counts)
                 if int(v) not in (self.bg, 0, self.cursor_color)]
        cands.sort(key=lambda x: x[1])
        return [c for c, _ in cands[:limit]]

    _DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

    def _build_logical(self, grid: np.ndarray) -> Optional[tuple]:
        """Tile the grid by the cursor step size → (logical, gh, gw, cursor_cell, dir→action)."""
        if self.cursor_color is None or len(self.arrow_map) < 2 or self.bg is None:
            return None
        step = max(2, int(round(float(np.median(self.step_sizes)) if self.step_sizes else 4)))
        h, w = grid.shape
        gh, gw = h // step, w // step
        if gh < 2 or gw < 2:
            return None
        logical = np.zeros((gh, gw), dtype=np.int16)
        for r in range(gh):
            for c in range(gw):
                tile = grid[r * step:(r + 1) * step, c * step:(c + 1) * step]
                if tile.size:
                    logical[r, c] = int(np.bincount(tile.ravel().astype(int), minlength=16).argmax())
        cpos = np.argwhere(grid == self.cursor_color)
        if len(cpos) == 0:
            return None
        cc = cpos.mean(0)
        cur = (min(int(cc[0]) // step, gh - 1), min(int(cc[1]) // step, gw - 1))
        d2a = {self._DIRS[n]: a for n, a in self.arrow_map.items() if n in self._DIRS}
        return logical, gh, gw, cur, d2a

    def _bfs(self, logical, gh, gw, walls, d2a, start, goal) -> Optional[list[str]]:
        q: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])
        seen = {start}
        while q:
            pos, path = q.popleft()
            if pos == goal:
                return path
            if len(path) > gh * gw * 2:
                break
            for (dy, dx), act in sorted(d2a.items()):
                nr, nc = pos[0] + dy, pos[1] + dx
                if 0 <= nr < gh and 0 <= nc < gw and (nr, nc) not in seen and int(logical[nr, nc]) not in walls:
                    seen.add((nr, nc))
                    q.append(((nr, nc), path + [act]))
        return None

    def _setup_target(self, grid, target_color):
        """Shared setup → (logical, gh, gw, cur, d2a, walls, target_cells, target_color)."""
        built = self._build_logical(grid)
        if built is None:
            return None
        logical, gh, gw, cur, d2a = built
        colors = [int(c) for c in set(logical.ravel().tolist()) if c not in (self.bg, self.cursor_color)]
        if not colors:
            return None
        counts = {c: int((logical == c).sum()) for c in colors}
        if target_color is None:
            target_color = min(counts, key=lambda c: counts[c])
        elif target_color not in counts:
            return None
        walls = {c for c in colors if c != target_color and counts[c] > max(gh * gw * 0.2, 3)}
        cells = [tuple(int(v) for v in p) for p in np.argwhere(logical == target_color)]
        return logical, gh, gw, cur, d2a, walls, cells

    def plan_path(self, grid: np.ndarray, target_color: Optional[int] = None) -> Optional[list[str]]:
        """Path to a representative target tile (+ interact). Goal hypothesis = 'reach a tile'."""
        s = self._setup_target(grid, target_color)
        if s is None:
            return None
        logical, gh, gw, cur, d2a, walls, cells = s
        if not cells:
            return None
        tgt = cells[len(cells) // 2]  # representative tile (empirically the strongest pick)
        if cur == tgt:
            return None
        p = self._bfs(logical, gh, gw, walls, d2a, cur, tgt)
        if p is None:
            return None
        return p + ([self.interact] if self.interact else [])

    def plan_collect(self, grid: np.ndarray, target_color: Optional[int] = None) -> Optional[list[str]]:
        """Greedy nearest-neighbour tour visiting ALL target tiles. Goal hypothesis =
        'collect/sweep all of a color' (also passes through the first tile, so it subsumes
        reach-one). Returns the concatenated action path, or None."""
        s = self._setup_target(grid, target_color)
        if s is None:
            return None
        logical, gh, gw, cur, d2a, walls, cells = s
        remaining = [c for c in cells if c != cur]
        if len(remaining) < 2:
            return None  # single tile → plan_path already covers it
        full: list[str] = []
        pos = cur
        while remaining and len(full) <= gh * gw * 4:
            best_t, best_p = None, None
            for t in remaining:
                p = self._bfs(logical, gh, gw, walls, d2a, pos, t)
                if p is not None and (best_p is None or len(p) < len(best_p)):
                    best_t, best_p = t, p
            if best_t is None:
                break  # unreachable remainder
            full += best_p
            pos = best_t
            remaining.remove(best_t)
        return full or None


# ───────────────────────────────── the agent ──────────────────────────────────────────
class MyAgent(Agent):  # type: ignore[misc]
    """Deterministic offline solver: reactive navigation → BFS reset-replay fallback."""

    MAX_ACTIONS = 1000
    WARMUP_STEPS = 8
    CLICK_BUDGET = 24          # salient click targets per state (coverage > branching savings)
    GIVEUP_PER_LEVEL = 1000    # stop burning actions on a stuck level (≈ the grader's 5x cutoff)
    STEP_MODULUS = 1           # >1 only for known-periodic games (per-game lever, off by default)
    NAV_CAP = 500              # max goal-hypothesis-search steps before falling back to BFS
    # occam ensemble (Kaggle-portable, reset-replay): clickscan finds effective click positions
    # for click-puzzle games; combosearch enumerates short simple sequences. Both run only on the
    # nav-fail branch and terminate into BFS, with tight budgets so they never starve it.
    DENSE_SCAN_STEP = 8        # coarse grid step for clickscan (catches non-salient clicks)
    CLICKSCAN_BUDGET = 200     # max actions in the clickscan phase before falling to BFS
    COMBOSEARCH_CAP = 250      # max actions in combosearch before falling to BFS
    COMBO_MAX_DEPTH = 6        # iterative-deepening sequence length cap (6^7 explodes past this)
    COMBO_EXHAUSTIVE_MAX = 6   # only combosearch when effective-simple count is within [1, 6]
    OCCAM_MAX_ACTIONS = 2000   # per-level search budget for the occam orchestrator (when available)
    OCCAM_MAX_LEVELS = 40      # assumed upper bound on levels (baselines are hidden at runtime)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._level = 0
        self._nav_target_hint: Optional[int] = None  # target color that solved a prior level
        self._winning_combo: Optional[list] = None   # the prior level's solving action sequence
        self._since_reset: list = []                 # actions emitted since the last RESET
        self._reset_level()

    def _reset_level(self) -> None:
        # Phases: warmup → navprobe → navhypo (goal-hypothesis search) → bfs.
        self._phase = "warmup"
        self._mask: Optional[np.ndarray] = None
        self._warm: list[np.ndarray] = []
        self._level_actions = 0
        self._search = ReplaySearch(step_modulus=self.STEP_MODULUS)
        self._nav = ReactiveNav()
        self._nav_seq: deque[str] = deque()
        self._nav_last = RESET
        self._nav_root: Optional[np.ndarray] = None
        self._simples: list[str] = []
        self._nav_steps = 0
        self._hypo_targets: Optional[deque[int]] = None
        self._hypo_path: Optional[deque[str]] = None
        self._hypo_last = RESET
        self._hypo_target: Optional[int] = None
        self._has_click = False
        self._eff_simples: set[str] = set()          # simples that changed the masked frame (from navprobe)
        # clickscan state
        self._cs_seq: deque[str] = deque()
        self._cs_last: Optional[str] = RESET
        self._cs_root_sig: Optional[str] = None
        self._cs_seen: set[str] = set()
        self._eff_clicks: Optional[list[tuple[int, int]]] = None  # discovered effective click positions
        self._cs_actions = 0
        # combosearch state
        self._combo_simples: list[str] = []
        self._combo_depth = 1
        self._combo_iter: Any = None
        self._combo_plan: deque[str] = deque()
        self._combo_actions = 0
        # Cross-level reuse: replay the prior level's winning combo first (occam short-circuit).
        self._combo_q: deque = deque(self._winning_combo) if self._winning_combo else deque()
        if self._winning_combo:
            self._phase = "combo"

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return latest_frame.state is GameState.WIN

    def main(self) -> None:
        """Play one game with a HYBRID solver (best Kaggle-portable coverage we can reach):
          1. occam's algorithmic solver (MIT, $0, reset-replay, no LLM) — wins the movement
             games (~7/25). NB: occam's headline 17/25 relies on deepcopy-BFS to search the env
             for free, which is impossible against the Kaggle gateway (every probe is a counted
             action), so the portable ceiling is ~7/25.
          2. If occam solves nothing, fall back to the built-in step-wise agent, which wins a
             couple of non-movement games occam-portable misses (VC33, LP85).
        Net portable coverage ≈ 9/25. Always offline-safe: any occam failure → the step-wise agent."""
        occam_solved = False
        try:
            occam_solved = self._run_occam()
        except Exception as exc:  # import missing / env mismatch / runtime → safe fallback
            if _DEBUG:
                print(f"[occam] {self.game_id}: error, fallback to step-wise ({exc!r})", flush=True)
        if occam_solved:
            self.cleanup()
        else:
            try:
                self.arc_env.reset()  # clean root after occam, before the step-wise fallback
            except Exception:
                pass
            super().main()  # framework per-step loop drives the step-wise agent

    def _run_occam(self) -> bool:
        """Run occam's orchestrator on this game. Returns True iff it completed ≥1 level."""
        import asyncio
        import importlib.util
        import sys
        from pathlib import Path

        here = Path(__file__).resolve().parent
        bundle = here / "occam_bundle.py"
        if bundle.exists():
            # Self-contained flattened occam (what ships to Kaggle). sys.modules set first so
            # the bundle's @dataclass introspection resolves under importlib.
            spec = importlib.util.spec_from_file_location("occam_bundle", bundle)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["occam_bundle"] = mod
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            GameOrchestrator = mod.GameOrchestrator
            ArcEnvAdapter = mod.ArcEnvAdapter
        else:
            # Dev fallback: the vendored occam package.
            vendored = here.parents[0] / "vendor" / "occam"
            if vendored.exists() and str(vendored) not in sys.path:
                sys.path.insert(0, str(vendored))
            from solver.environments import ArcEnvAdapter  # type: ignore
            from solver.orchestrator import GameOrchestrator  # type: ignore

        # skip_deepcopy: deepcopy-BFS clones the env, impossible against the Kaggle gateway.
        orch = GameOrchestrator(max_actions_per_level=self.OCCAM_MAX_ACTIONS, skip_deepcopy=True)
        env = ArcEnvAdapter(self.arc_env)
        # Size occam's level loop to the real level count (FrameData.win_levels) so it does not
        # over-run a won game (which would replay it and tank the RHAE).
        try:
            n_levels = int(getattr(self.arc_env.reset(), "win_levels", 0)) or self.OCCAM_MAX_LEVELS
        except Exception:
            n_levels = self.OCCAM_MAX_LEVELS
        n_levels = max(1, min(n_levels, self.OCCAM_MAX_LEVELS))
        result = asyncio.run(orch.play_game(env, [self.OCCAM_MAX_ACTIONS] * n_levels))
        won = bool(result.get("levels_completed", 0)) if isinstance(result, dict) else False
        if _DEBUG:
            print(f"[occam] {self.game_id}: levels={result.get('levels_completed') if isinstance(result, dict) else '?'}", flush=True)
        return won

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        action = self._decide(frames, latest_frame)
        self._record(action)
        return action

    def _record(self, action: Any) -> None:
        """Track the action sequence since the last RESET, for winning-combo caching."""
        if action is GameAction.RESET:
            self._since_reset = []
            return
        name = getattr(action, "name", None)
        if name == "ACTION6":
            try:
                d = action.action_data.model_dump()
                self._since_reset.append(("ACTION6", int(d.get("x", 0)), int(d.get("y", 0))))
            except Exception:
                self._since_reset.append(("ACTION6", 0, 0))
        elif name:
            self._since_reset.append((name,))

    def _decide(self, frames: list[Any], latest_frame: Any) -> Any:
        state = latest_frame.state
        if state is GameState.NOT_PLAYED:
            self._search.note_reset()
            return GameAction.RESET

        if latest_frame.levels_completed > self._level:
            self._level = latest_frame.levels_completed
            if self._since_reset:  # the actions since the last RESET just solved a level
                self._winning_combo = list(self._since_reset)
            if self._hypo_target is not None:
                self._nav_target_hint = self._hypo_target
            self._since_reset = []
            self._reset_level()

        self._level_actions += 1
        if self._level_actions > self.GIVEUP_PER_LEVEL:
            return GameAction.RESET

        grid = grid_from_frame(latest_frame.frame)

        if self._phase == "combo":
            return self._combo_phase(grid, state)
        if self._phase == "warmup":
            return self._warmup_phase(latest_frame, grid, state)
        if self._phase == "navprobe":
            return self._navprobe_phase(grid, state)
        if self._phase == "navhypo":
            return self._navhypo_phase(grid, state)
        if self._phase == "clickscan":
            return self._clickscan_phase(grid, state)
        if self._phase == "combosearch":
            return self._combosearch_phase(grid, state)

        sig = frame_signature(grid, self._mask)
        candidates = self._candidates(latest_frame, grid)
        token = self._search.step(sig, candidates, game_over=state is GameState.GAME_OVER)
        return self._to_action(token)

    # -- phase: replay the previous level's winning combo (occam short-circuit) --
    def _combo_phase(self, grid: np.ndarray, state: Any) -> Any:
        if state is GameState.GAME_OVER or not self._combo_q:
            self._phase = "warmup"   # the combo didn't carry the new level → normal solving
            self._warm = []
            return GameAction.RESET
        item = self._combo_q.popleft()
        if _DEBUG and self._winning_combo and len(self._combo_q) == len(self._winning_combo) - 1:
            print(f"[combo] {self.game_id}: replaying {len(self._winning_combo)}-action combo", flush=True)
        if item[0] == "ACTION6":
            return self._to_action(f"ACTION6:{item[1]},{item[2]}")
        return self._to_action(item[0])

    # -- phase: warmup → freeze mask, then set up the nav probe --
    def _warmup_phase(self, latest_frame: Any, grid: np.ndarray, state: Any) -> Any:
        if state is GameState.GAME_OVER:
            return GameAction.RESET
        self._warm.append(grid)
        if len(self._warm) < self.WARMUP_STEPS:
            simples = [t for t in self._candidates(latest_frame, grid) if not t.startswith("ACTION6:")]
            return self._to_action(simples[len(self._warm) % len(simples)]) if simples else GameAction.RESET
        self._mask = combine_masks(status_bar_mask(grid), volatility_mask(self._warm))
        cands = self._candidates(latest_frame, grid)
        self._simples = [t for t in cands if not t.startswith("ACTION6:")]
        self._has_click = any(t.startswith("ACTION6:") for t in cands)
        seq: list[str] = []
        for i, a in enumerate(self._simples):  # probe each simple from root: [a1, RESET, a2, RESET, ...]
            seq.append(a)
            if i < len(self._simples) - 1:
                seq.append(RESET)
        self._nav_seq = deque(seq)
        self._nav_last = RESET
        self._phase = "navprobe"
        return GameAction.RESET  # → root, then probing begins

    # -- phase: probe each action once to detect a cursor + arrow directions --
    def _navprobe_phase(self, grid: np.ndarray, state: Any) -> Any:
        if self._nav_last == RESET:
            self._nav_root = grid
        elif self._nav_last is not None and self._nav_root is not None:
            if state is not GameState.GAME_OVER:
                self._nav.probe(self._nav_root, grid, self._nav_last)
                if frame_signature(grid, self._mask) != frame_signature(self._nav_root, self._mask):
                    self._eff_simples.add(self._nav_last)  # this simple changes the world
            else:
                self._eff_simples.add(self._nav_last)      # caused a state change (death)
        if self._nav_seq:
            self._nav_last = self._nav_seq.popleft()
            return self._to_action(self._nav_last)
        self._nav.finalize(self._simples)
        self._nav_last = RESET
        if self._nav.ready():
            if _DEBUG:
                print(f"[nav] {self.game_id}: NAVHYPO cursor={self._nav.cursor_color} "
                      f"arrows={self._nav.arrow_map} interact={self._nav.interact}", flush=True)
            self._phase = "navhypo"
            self._hypo_last, self._hypo_targets, self._hypo_path, self._nav_steps = RESET, None, None, 0
            return GameAction.RESET  # clean root start for navhypo
        if _DEBUG:
            print(f"[nav] {self.game_id}: no cursor → route (eff_simples={sorted(self._eff_simples)})", flush=True)
        return self._route_after_nav()

    # -- phase: goal-hypothesis search — pathfind to each candidate target, interact, test --
    def _navhypo_phase(self, grid: np.ndarray, state: Any) -> Any:
        self._nav_steps += 1
        if self._nav_steps > self.NAV_CAP:
            return self._route_after_nav()
        if state is GameState.GAME_OVER:
            return self._reset_for_next()  # this hypothesis died; try the next target
        if self._hypo_last == RESET:  # at the level root
            if self._hypo_targets is None:
                cands = self._nav.candidate_targets(grid)
                hint = [self._nav_target_hint] if self._nav_target_hint is not None else []
                colors = hint + [c for c in cands if c != self._nav_target_hint]
                # ALL reach+interact hypotheses first (efficient on reach-one games), then
                # collect-all as a true last resort. Interleaving regressed efficiency on the
                # public set; collect adds 0 coverage there but is kept for the hidden set.
                self._hypo_targets = deque(
                    [(c, "reach") for c in colors] + [(c, "collect") for c in colors]
                )
            return self._start_next_target(grid)
        if self._hypo_path:  # mid-path
            tok = self._hypo_path.popleft()
            self._hypo_last = tok
            return self._to_action(tok)
        return self._reset_for_next()  # path finished without solving → next hypothesis

    def _start_next_target(self, root_grid: np.ndarray) -> Any:
        while self._hypo_targets:
            tc, method = self._hypo_targets.popleft()
            path = (self._nav.plan_collect(root_grid, tc) if method == "collect"
                    else self._nav.plan_path(root_grid, tc))
            if path:
                if _DEBUG:
                    print(f"[nav] {self.game_id}: HYPO {method} target={tc} path={len(path)}", flush=True)
                self._hypo_target = tc
                self._hypo_path = deque(path)
                tok = self._hypo_path.popleft()
                self._hypo_last = tok
                return self._to_action(tok)
        return self._route_after_nav()

    def _reset_for_next(self) -> Any:
        self._hypo_path = None
        if not self._hypo_targets:
            return self._route_after_nav()
        self._hypo_last = RESET
        return GameAction.RESET

    def _fall_to_bfs(self) -> Any:
        self._phase = "bfs"
        self._search = ReplaySearch(step_modulus=self.STEP_MODULUS)
        return GameAction.RESET

    # -- router: after nav fails, try clickscan / combosearch before the BFS catch-all --
    def _route_after_nav(self) -> Any:
        # Movement games (a cursor was detected) go straight to BFS — clickscan/combosearch are
        # for non-movement click/puzzle games, and would only starve the BFS that solves these.
        if self._nav.cursor_color is not None:
            return self._fall_to_bfs()
        eff = sorted(s for s in self._simples if s in self._eff_simples)
        simples_dead = bool(self._simples) and not eff
        if self._has_click and simples_dead:
            self._phase = "clickscan"
            self._cs_seq, self._cs_last, self._cs_root_sig = deque(), RESET, None
            self._cs_seen, self._eff_clicks, self._cs_actions, self._cs_probed = set(), [], 0, 0
            if _DEBUG:
                print(f"[route] {self.game_id}: → clickscan", flush=True)
            return GameAction.RESET
        if 1 <= len(eff) <= self.COMBO_EXHAUSTIVE_MAX:
            self._phase = "combosearch"
            self._combo_simples, self._combo_depth = eff, 1
            self._combo_iter, self._combo_plan, self._combo_actions = None, deque(), 0
            if _DEBUG:
                print(f"[route] {self.game_id}: → combosearch eff={eff}", flush=True)
            return GameAction.RESET
        return self._fall_to_bfs()

    # -- phase: dense click scan → find effective click positions, then focus BFS on them --
    def _clickscan_targets(self, grid: np.ndarray) -> list[tuple[int, int]]:
        corners = [(2, 2), (61, 2), (2, 61), (61, 61), (31, 2), (2, 31)]
        salient = priority_click_targets(grid, self.CLICK_BUDGET)
        dense = [(x, y) for y in range(0, GRID_SIZE, self.DENSE_SCAN_STEP)
                 for x in range(0, GRID_SIZE, self.DENSE_SCAN_STEP)]
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for p in corners + salient + dense:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _clickscan_phase(self, grid: np.ndarray, state: Any) -> Any:
        self._cs_actions += 1
        if self._cs_actions > self.CLICKSCAN_BUDGET:
            return self._clickscan_done()
        if self._cs_last == RESET:
            if self._cs_root_sig is None:  # first entry: at root, build the scan sequence
                self._cs_root_sig = frame_signature(grid, self._mask)
                positions = self._clickscan_targets(grid)
                seq: list[str] = []
                for i, (x, y) in enumerate(positions):
                    seq.append(f"ACTION6:{x},{y}")
                    if i < len(positions) - 1:
                        seq.append(RESET)
                self._cs_seq = deque(seq)
        elif self._cs_last is not None and self._cs_last.startswith("ACTION6:"):
            if state is not GameState.GAME_OVER:
                sig = frame_signature(grid, self._mask)
                if sig != self._cs_root_sig and sig not in self._cs_seen:
                    self._cs_seen.add(sig)
                    x, y = (int(v) for v in self._cs_last.split(":", 1)[1].split(","))
                    self._eff_clicks.append((x, y))
            self._cs_probed += 1
            if self._cs_probed >= 6 and not self._eff_clicks:  # corner-skip: clicks do nothing
                self._eff_clicks = None
                return self._fall_to_bfs()
        if not self._cs_seq:
            return self._clickscan_done()
        self._cs_last = self._cs_seq.popleft()
        return self._to_action(self._cs_last)

    def _clickscan_done(self) -> Any:
        if not self._eff_clicks:  # nothing found → let BFS use the default salient targets
            self._eff_clicks = None
        if _DEBUG:
            n = len(self._eff_clicks) if self._eff_clicks else 0
            print(f"[clickscan] {self.game_id}: {n} effective clicks → BFS", flush=True)
        return self._fall_to_bfs()

    # -- phase: iterative-deepening over effective simple-action sequences --
    def _combosearch_phase(self, grid: np.ndarray, state: Any) -> Any:
        if self._combo_actions > self.COMBOSEARCH_CAP:
            return self._fall_to_bfs()
        if state is GameState.GAME_OVER:
            self._combo_plan = deque()  # current combo died → next combo (starts with RESET)
        if self._combo_plan:
            self._combo_actions += 1
            return self._to_action(self._combo_plan.popleft())
        if self._combo_iter is None:
            self._combo_iter = product(self._combo_simples, repeat=self._combo_depth)
        nxt = next(self._combo_iter, None)
        while nxt is None:
            self._combo_depth += 1
            if self._combo_depth > self.COMBO_MAX_DEPTH:
                return self._fall_to_bfs()
            self._combo_iter = product(self._combo_simples, repeat=self._combo_depth)
            nxt = next(self._combo_iter, None)
        self._combo_plan = deque([RESET, *nxt])
        self._combo_actions += 1
        return self._to_action(self._combo_plan.popleft())

    def _candidates(self, latest_frame: Any, grid: np.ndarray) -> list[str]:
        available = getattr(latest_frame, "available_actions", None)
        if not available:
            available = [a for a in GameAction]  # type: ignore[union-attr]
        available = [GameAction.from_id(a) if isinstance(a, int) else a for a in available]
        simple: list[str] = []
        has_click = False
        for a in available:
            if a is GameAction.RESET:
                continue
            if a.is_complex():
                has_click = True
            else:
                simple.append(a.name)
        simple.sort()
        if not has_click:
            clicks: list[str] = []
        elif self._eff_clicks is not None:  # clickscan discovered the effective positions → focus BFS
            clicks = [f"ACTION6:{x},{y}" for (x, y) in self._eff_clicks]
        else:
            clicks = [f"ACTION6:{x},{y}" for (x, y) in priority_click_targets(grid, self.CLICK_BUDGET)]
        return simple + clicks

    def _to_action(self, token: str) -> Any:
        if token == RESET:
            return GameAction.RESET
        if token.startswith("ACTION6:"):
            x, y = (int(v) for v in token.split(":", 1)[1].split(","))
            action = GameAction.ACTION6
            action.set_data({"x": x, "y": y})
            action.reasoning = {"why": "salient click", "x": x, "y": y}
            return action
        action = getattr(GameAction, token)
        action.reasoning = {"why": "bfs probe"}
        return action
