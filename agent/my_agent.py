"""ARC-AGI-3 offline agent — self-contained (this whole file is spliced into the
Kaggle submission notebook, so it may import ONLY: stdlib, numpy, arcengine,
agents.agent).

Strategy (see SPEC.md / CONTRACT.md / DECISION-LOG.md):
  - No model in the loop. Deterministic, symbolic, offline. Coverage first.
  - Perceive the 64x64 grid as colored objects via connected components (no scipy).
  - Click object centroids only (ACTION6), collapsing 4096 pixels to a handful.
  - Solve via reset-and-replay DFS (occam's technique): explore the game's reachable
    states; because the engine is deterministic (confirmed), any state is revisitable by
    RESET + replaying the action prefix that first reached it. DFS goes deep first (good
    for reaching goals) and follows the agent's current position, so it resets only on
    dead-ends / backtracks rather than on every probe.
  - Level-aware: on level-up, start a fresh per-level search (RESET = level restart, so
    completed levels are kept).
  - Deterministic: seeded by nothing wall-clock; sorted iteration; no RNG.
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional

import numpy as np

# Guarded imports so the pure helpers + ReplayDFS stay importable in unit tests without
# the framework / SDK. On Kaggle and in `make play-local` these always resolve.
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
MAX_CLICK_CANDIDATES = 16  # bound ACTION6 branching to the biggest objects
RESET = "RESET"


# ───────────────────────── pure perception helpers (testable) ─────────────────────────
def grid_from_frame(frame: Any) -> np.ndarray:
    """Return the current 64x64 observation as an int array (last grid in the stack)."""
    if frame is None:
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int16)
    arr = np.asarray(frame[-1] if len(frame) else frame, dtype=np.int16)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1]) if arr.ndim >= 2 else arr
    return arr


def frame_signature(grid: np.ndarray, mask: Optional[np.ndarray] = None) -> str:
    """Stable short hash of a grid (deterministic, fast).

    If `mask` is given, masked cells are zeroed before hashing so volatile counter /
    status-bar pixels don't make every frame look like a new state.
    """
    if mask is not None:
        grid = np.where(mask, np.int16(0), grid)
    return hashlib.md5(np.ascontiguousarray(grid).tobytes()).hexdigest()[:16]


def volatility_mask(grids: list[np.ndarray], max_fraction: float = 0.10) -> Optional[np.ndarray]:
    """Detect counter/status-bar pixels: cells that change on EVERY warmup transition.

    A move counter increments each step (changes every transition); the player/objects
    change only on effective moves (not every transition). We mask only the always-changing
    cells, and bail out (return None) if that region is large (>max_fraction) — that means
    the play area itself is animating and masking would erase real state.
    """
    if len(grids) < 2:
        return None
    stack = np.stack(grids)
    diffs = stack[1:] != stack[:-1]
    always = diffs.all(axis=0)
    n = int(always.sum())
    if 0 < n <= max_fraction * always.size:
        return always
    return None


def connected_components(grid: np.ndarray) -> list[dict[str, Any]]:
    """4-connected same-color components over non-zero cells (background = 0).

    Pure numpy + iterative BFS, no scipy. Sorted by descending size then position for
    deterministic ordering. Each component: {color, size, centroid (y, x)}.
    """
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, Any]] = []
    for y in range(h):
        for x in range(w):
            color = int(grid[y, x])
            if color == 0 or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(grid[ny, nx]) == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            comps.append(
                {
                    "color": color,
                    "size": len(cells),
                    "centroid": (int(round(sum(ys) / len(ys))), int(round(sum(xs) / len(xs)))),
                }
            )
    comps.sort(key=lambda c: (-c["size"], c["centroid"][0], c["centroid"][1], c["color"]))
    return comps


def click_candidates(grid: np.ndarray, limit: int = MAX_CLICK_CANDIDATES) -> list[tuple[int, int]]:
    """Deterministic, deduplicated list of (x, y) click targets at object centroids."""
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for comp in connected_components(grid):
        cy, cx = comp["centroid"]
        xy = (int(cx), int(cy))  # API uses (x, y)
        if xy not in seen:
            seen.add(xy)
            out.append(xy)
        if len(out) >= limit:
            break
    return out


# ─────────────────────── reset-and-replay DFS solver (testable) ────────────────────────
class ReplayDFS:
    """Step-wise depth-first reset-and-replay search over discrete action tokens.

    Tokens are strings: "ACTION1".."ACTION5" and "ACTION6:x,y" for clicks; "RESET" is
    reserved. The driver calls `step()` once per real environment turn with the observed
    state signature + the candidate tokens legal there, and applies the returned token.

    Invariants (rely on confirmed determinism):
      - `known[sig]` is the action prefix (no RESET) from the level root to `sig`.
      - To probe an untried token from a target state we reposition there incrementally
        (replay the unseen suffix) when the current state is an ancestor, else RESET +
        full replay. DFS keeps the current state == target most of the time → few resets.
    """

    def __init__(self) -> None:
        self.root: Optional[str] = None
        self.known: dict[str, list[str]] = {}
        self.by_prefix: dict[tuple[str, ...], str] = {}
        self.cand: dict[str, list[str]] = {}
        self.tried: dict[str, set[str]] = {}
        self.terminal: set[str] = set()
        self.expanding: Optional[str] = None
        self.plan: list[str] = []
        self.awaiting: Optional[tuple[str, str]] = None
        self.cur: Optional[str] = None

    # -- level lifecycle ---------------------------------------------------------------
    def start_level(self, sig: str, candidates: list[str]) -> None:
        self.root = sig
        self.known = {sig: []}
        self.by_prefix = {(): sig}
        self.cand = {sig: list(candidates)}
        self.tried = {}
        self.terminal = set()
        self.expanding = sig
        self.plan = []
        self.awaiting = None
        self.cur = sig

    def note_reset(self) -> None:
        """Called when the env reports NOT_PLAYED — clear in-flight plan/awaiting."""
        self.plan = []
        self.awaiting = None
        self.cur = None

    # -- the per-turn decision ---------------------------------------------------------
    def step(self, sig: str, candidates: list[str], game_over: bool = False) -> str:
        if self.root is None:
            self.start_level(sig, candidates)
            return self._next_probe()

        if game_over:
            if self.awaiting is not None:
                e, t = self.awaiting
                if sig not in self.known:
                    self.known[sig] = self.known[e] + [t]
                    self.by_prefix[tuple(self.known[sig])] = sig
                self.terminal.add(sig)
                self.awaiting = None
            self.plan = []
            self.cur = None
            return RESET

        if self.awaiting is not None:
            e, t = self.awaiting
            is_new = sig not in self.known
            if is_new:
                self.known[sig] = self.known[e] + [t]
                self.by_prefix[tuple(self.known[sig])] = sig
                self.cand[sig] = list(candidates)
            self.awaiting = None
            self.cur = sig
            # DFS: descend into genuinely new, non-terminal states; else stay on `e`.
            self.expanding = sig if (is_new and sig not in self.terminal) else e
        else:
            self.cur = sig
            self.cand.setdefault(sig, list(candidates))

        if self.plan:
            return self._emit()
        return self._next_probe()

    # -- internals ---------------------------------------------------------------------
    def _emit(self) -> str:
        tok = self.plan.pop(0)
        if not self.plan and self.expanding is not None:
            self.awaiting = (self.expanding, tok)
        return tok

    def _untried(self, sig: str) -> list[str]:
        done = self.tried.setdefault(sig, set())
        return [c for c in self.cand.get(sig, []) if c not in done]

    def _next_probe(self) -> str:
        # Find a state with untried tokens, backtracking up the DFS tree as needed.
        while True:
            if self.expanding is None:
                return RESET  # exhausted: restart and keep trying within the budget
            ut = self._untried(self.expanding)
            if ut:
                break
            if self.expanding == self.root:
                self.expanding = None
                return RESET
            parent = self.by_prefix.get(tuple(self.known[self.expanding][:-1]))
            self.expanding = parent
            if parent is None:
                return RESET

        token = ut[0]
        self.tried[self.expanding].add(token)
        target_prefix = self.known[self.expanding]

        if self.cur == self.expanding:
            self.plan = [token]
        elif self.cur is not None and _is_prefix(self.known.get(self.cur, ["x"]), target_prefix):
            self.plan = target_prefix[len(self.known[self.cur]):] + [token]
        else:
            self.plan = [RESET] + target_prefix + [token]
        return self._emit()


def _is_prefix(p: list[str], q: list[str]) -> bool:
    return len(p) <= len(q) and q[: len(p)] == p


# ───────────────────────────────── the agent ──────────────────────────────────────────
class MyAgent(Agent):  # type: ignore[misc]
    """Deterministic offline reset-and-replay DFS solver."""

    MAX_ACTIONS = 1000   # generous per-game cap; the grader enforces the real ~5x cutoff
    WARMUP_STEPS = 8     # frames to gather before freezing the counter mask

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._level = 0
        self._reset_level()

    def _reset_level(self) -> None:
        self._dfs = ReplayDFS()
        self._mask: Optional[np.ndarray] = None
        self._warm: list[np.ndarray] = []
        self._frozen = False

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        state = latest_frame.state
        if state is GameState.NOT_PLAYED:
            self._dfs.note_reset()
            return GameAction.RESET

        if latest_frame.levels_completed > self._level:
            self._level = latest_frame.levels_completed
            self._reset_level()

        grid = grid_from_frame(latest_frame.frame)

        # Warmup: gather frames, then freeze the volatility mask and restart at root.
        if not self._frozen:
            if state is GameState.GAME_OVER:
                return GameAction.RESET
            self._warm.append(grid)
            if len(self._warm) >= self.WARMUP_STEPS:
                self._mask = volatility_mask(self._warm)
                self._frozen = True
                self._dfs = ReplayDFS()
                return GameAction.RESET
            return self._warmup_action(latest_frame, grid, len(self._warm))

        sig = frame_signature(grid, self._mask)
        candidates = self._candidates(latest_frame, grid)
        token = self._dfs.step(sig, candidates, game_over=state is GameState.GAME_OVER)
        return self._to_action(token)

    def _warmup_action(self, latest_frame: Any, grid: np.ndarray, i: int) -> Any:
        """Cycle the simple actions to provoke frame changes during warmup."""
        simples = [t for t in self._candidates(latest_frame, grid) if not t.startswith("ACTION6:")]
        if not simples:
            return GameAction.RESET
        return self._to_action(simples[i % len(simples)])

    # -- token <-> GameAction ----------------------------------------------------------
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
        clicks = [f"ACTION6:{x},{y}" for (x, y) in click_candidates(grid)] if has_click else []
        return simple + clicks

    def _to_action(self, token: str) -> Any:
        if token == RESET:
            return GameAction.RESET
        if token.startswith("ACTION6:"):
            x, y = (int(v) for v in token.split(":", 1)[1].split(","))
            action = GameAction.ACTION6
            action.set_data({"x": x, "y": y})
            action.reasoning = {"why": "centroid click", "x": x, "y": y}
            return action
        action = getattr(GameAction, token)
        action.reasoning = {"why": "dfs probe"}
        return action
