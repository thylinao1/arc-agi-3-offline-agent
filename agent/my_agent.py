"""ARC-AGI-3 offline agent — self-contained (this whole file is spliced into the
Kaggle submission notebook, so it may import ONLY: stdlib, numpy, arcengine,
agents.agent).

Strategy (see SPEC.md / CONTRACT.md / DECISION-LOG.md):
  - No model in the loop. Deterministic, symbolic, offline.
  - Coverage first: solve as many games/levels as possible (an unsolved level is 0).
  - Perceive the 64x64 grid as colored objects via connected components (no scipy).
  - Explore the frontier: try each action once per state, learn which actions are
    "effective" (change the frame), and prefer untried effective actions.
  - Click object centroids only (ACTION6), collapsing 4096 pixels to a handful.
  - Level-aware: on level-up, bank effective actions and refresh the per-level frontier.
  - Deterministic: seeded by game_id only; sorted iteration; no wall-clock, no RNG in
    the action choice path.

The full reset-and-replay BFS (occam's ReplayExplorer) is the next upgrade; this is a
better-than-random floor that exercises the whole offline pipeline end to end.
"""
from __future__ import annotations

import hashlib
from collections import deque
from typing import Any, Optional

import numpy as np

# Guarded imports so the pure helpers below stay importable in unit tests without the
# framework / SDK installed. On Kaggle and in `make play-local` these always resolve.
try:  # pragma: no cover - exercised on Kaggle / local-dev, not in pure unit tests
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
MAX_CLICK_CANDIDATES = 24  # bound ACTION6 branching to the biggest objects


# ───────────────────────── pure perception helpers (testable) ─────────────────────────
def grid_from_frame(frame: Any) -> np.ndarray:
    """Return the current 64x64 observation as an int array.

    FrameData.frame is a list of 64x64 grids; the last one is the current screen.
    """
    if frame is None:
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int16)
    arr = np.asarray(frame[-1] if len(frame) else frame, dtype=np.int16)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1]) if arr.ndim >= 2 else arr
    return arr


def frame_signature(grid: np.ndarray) -> str:
    """Stable short hash of a grid (deterministic, fast)."""
    return hashlib.md5(np.ascontiguousarray(grid).tobytes()).hexdigest()[:16]


def connected_components(grid: np.ndarray) -> list[dict[str, Any]]:
    """4-connected same-color components over non-zero cells (background = 0).

    Pure numpy + BFS, no scipy. Returns components sorted by descending size, then by
    position, for deterministic ordering. Each component: {color, size, centroid (y,x)}.
    """
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, Any]] = []
    for y in range(h):
        for x in range(w):
            color = int(grid[y, x])
            if color == 0 or seen[y, x]:
                continue
            q: deque[tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            cells: list[tuple[int, int]] = []
            while q:
                cy, cx = q.popleft()
                cells.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(grid[ny, nx]) == color:
                        seen[ny, nx] = True
                        q.append((ny, nx))
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


# ───────────────────────── exploration policy (testable core) ─────────────────────────
class FrontierPolicy:
    """Per-state frontier exploration over discrete action keys.

    Action keys are strings: "ACTION1".."ACTION5" for simple actions and
    "ACTION6:x,y" for clicks. Pure Python, deterministic, no GameAction dependency.
    """

    def __init__(self) -> None:
        self.tried: dict[str, set[str]] = {}
        self.transitions: dict[tuple[str, str], str] = {}
        self.effective: set[str] = set()
        self.last_sig: Optional[str] = None
        self.last_key: Optional[str] = None
        self.level: int = 0
        self.actions_this_level: int = 0
        self.consecutive_exhausted: int = 0

    def on_level_change(self, new_level: int) -> None:
        """Bank effective actions; refresh the per-level frontier for the new level."""
        self.level = new_level
        self.tried.clear()
        self.transitions.clear()
        self.last_sig = None
        self.last_key = None
        self.actions_this_level = 0
        self.consecutive_exhausted = 0

    def observe(self, sig: str) -> None:
        """Record the result of the previous action: transition + effective flag."""
        if self.last_sig is not None and self.last_key is not None:
            self.transitions[(self.last_sig, self.last_key)] = sig
            if sig != self.last_sig:
                self.effective.add(self.last_key)

    def select(self, sig: str, simple_keys: list[str], click_keys: list[str]) -> str:
        """Pick the next action key from `sig`. Returns a key or "RESET" to escape."""
        tried = self.tried.setdefault(sig, set())

        def first_untried(keys: list[str]) -> Optional[str]:
            for k in keys:
                if k not in tried:
                    return k
            return None

        # 1) untried effective simple actions, 2) untried simple, 3) untried clicks.
        ordered_simple = [k for k in simple_keys if k in self.effective] + [
            k for k in simple_keys if k not in self.effective
        ]
        choice = first_untried(ordered_simple) or first_untried(click_keys)

        if choice is None:
            # State exhausted and non-terminal: reset to re-branch from the start.
            self.consecutive_exhausted += 1
            self.last_sig = None
            self.last_key = None
            return "RESET"

        self.consecutive_exhausted = 0
        tried.add(choice)
        self.last_sig = sig
        self.last_key = choice
        self.actions_this_level += 1
        return choice


# ───────────────────────────────── the agent ──────────────────────────────────────────
class MyAgent(Agent):  # type: ignore[misc]
    """Deterministic offline frontier explorer."""

    MAX_ACTIONS = 600  # generous per-game cap; the grader enforces the real ~5x cutoff

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._policy = FrontierPolicy()

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        # Stop only on a win; on GAME_OVER we RESET and keep exploring.
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        # First contact or a death → reset the level.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._policy.last_sig = None
            self._policy.last_key = None
            return GameAction.RESET

        # Level-up: bank knowledge, refresh the frontier.
        if latest_frame.levels_completed > self._policy.level:
            self._policy.on_level_change(latest_frame.levels_completed)

        grid = grid_from_frame(latest_frame.frame)
        sig = frame_signature(grid)
        self._policy.observe(sig)

        simple_keys, click_keys, complex_action = self._candidate_keys(latest_frame, grid)
        key = self._policy.select(sig, simple_keys, click_keys)

        if key == "RESET":
            return GameAction.RESET
        if key.startswith("ACTION6:"):
            x, y = (int(v) for v in key.split(":", 1)[1].split(","))
            action = complex_action if complex_action is not None else GameAction.ACTION6
            action.set_data({"x": x, "y": y})
            action.reasoning = {"why": "centroid click", "x": x, "y": y}
            return action
        action = GameAction.from_name(key) if hasattr(GameAction, "from_name") else getattr(GameAction, key)
        action.reasoning = {"why": "frontier simple action"}
        return action

    def _candidate_keys(
        self, latest_frame: Any, grid: np.ndarray
    ) -> tuple[list[str], list[str], Any]:
        """Build (simple_keys, click_keys, complex_action) from available actions."""
        available = getattr(latest_frame, "available_actions", None)
        if not available:
            available = [a for a in GameAction]  # type: ignore[union-attr]
        # `available_actions` arrives as action ids (ints) at runtime; normalize to GameAction.
        available = [GameAction.from_id(a) if isinstance(a, int) else a for a in available]

        simple_keys: list[str] = []
        complex_action: Any = None
        for a in available:
            if a is GameAction.RESET:
                continue
            if a.is_complex():
                complex_action = a
            else:
                simple_keys.append(a.name)
        simple_keys.sort()

        click_keys: list[str] = []
        if complex_action is not None:
            click_keys = [f"ACTION6:{x},{y}" for (x, y) in click_candidates(grid)]
        return simple_keys, click_keys, complex_action
