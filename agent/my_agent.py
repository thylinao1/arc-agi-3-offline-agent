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
from typing import Any, Optional

import numpy as np

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


# ───────────────────────────────── the agent ──────────────────────────────────────────
class MyAgent(Agent):  # type: ignore[misc]
    """Deterministic offline BFS reset-and-replay solver with counter masking."""

    MAX_ACTIONS = 1000
    WARMUP_STEPS = 8
    CLICK_BUDGET = 24          # salient click targets per state (coverage > branching savings)
    GIVEUP_PER_LEVEL = 1000    # stop burning actions on a stuck level (≈ the grader's 5x cutoff)
    STEP_MODULUS = 1           # >1 only for known-periodic games (per-game lever, off by default)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._level = 0
        self._reset_level()

    def _reset_level(self) -> None:
        self._search = ReplaySearch(step_modulus=self.STEP_MODULUS)
        self._mask: Optional[np.ndarray] = None
        self._warm: list[np.ndarray] = []
        self._frozen = False
        self._level_actions = 0

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        state = latest_frame.state
        if state is GameState.NOT_PLAYED:
            self._search.note_reset()
            return GameAction.RESET

        if latest_frame.levels_completed > self._level:
            self._level = latest_frame.levels_completed
            self._reset_level()

        # Per-level give-up: once a level has burned its budget, stop wasting actions on it.
        self._level_actions += 1
        if self._level_actions > self.GIVEUP_PER_LEVEL:
            return GameAction.RESET

        grid = grid_from_frame(latest_frame.frame)

        if not self._frozen:
            if state is GameState.GAME_OVER:
                return GameAction.RESET
            self._warm.append(grid)
            if len(self._warm) >= self.WARMUP_STEPS:
                self._mask = combine_masks(status_bar_mask(grid), volatility_mask(self._warm))
                self._frozen = True
                self._search = ReplaySearch()
                return GameAction.RESET
            return self._warmup_action(latest_frame, grid, len(self._warm))

        sig = frame_signature(grid, self._mask)
        candidates = self._candidates(latest_frame, grid)
        token = self._search.step(sig, candidates, game_over=state is GameState.GAME_OVER)
        return self._to_action(token)

    def _warmup_action(self, latest_frame: Any, grid: np.ndarray, i: int) -> Any:
        simples = [t for t in self._candidates(latest_frame, grid) if not t.startswith("ACTION6:")]
        if not simples:
            return GameAction.RESET
        return self._to_action(simples[i % len(simples)])

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
        clicks = (
            [f"ACTION6:{x},{y}" for (x, y) in priority_click_targets(grid, self.CLICK_BUDGET)]
            if has_click
            else []
        )
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
