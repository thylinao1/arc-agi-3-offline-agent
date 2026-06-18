# CONTRACT — interface seams (ground truth from the real SDK)

Verified against `vendor/ARC-AGI-3-Agents/agents/agent.py` and `vendor/occam/solver/rhae.py`. Build agents code
against THIS; do not redefine these seams.

## The agent entry point
```python
from arcengine import FrameData, GameAction, GameState   # core types
from agents.agent import Agent                            # framework base (on sys.path locally + on Kaggle)

class MyAgent(Agent):                 # class MUST be named MyAgent (notebook + registry rely on it)
    MAX_ACTIONS = 80                  # framework loop cap: while not is_done(...) and action_counter <= MAX_ACTIONS
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool: ...
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction: ...
```
- The framework constructs the agent: `MyAgent(card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags=...)`.
- `self.action_counter`, `self.frames`, `self.game_id`, `self.state`, `self.levels_completed` are provided.

## FrameData (fields the policy reads)
- `frame: list[list[list[int]]]` — a list of 64×64 grids; cell values are ints **0–15**. Use the **last** grid as
  the current observation: `grid = latest_frame.frame[-1]`.
- `state: GameState` — `NOT_PLAYED | NOT_FINISHED | WIN | GAME_OVER`.
- `available_actions: list[GameAction]` — the **ground truth** of what's legal this turn. Always intersect with it.
- `levels_completed: int` — increments when a level is solved (the progress signal).
- `score`, `win_levels`, `guid`, `full_reset` — also present.

## GameAction protocol
- Members: `RESET`, `ACTION1..ACTION6`. Semantics: 1=up, 2=down, 3=left, 4=right, 5=interact/select,
  6=click(x,y). (`ACTION7`/undo may appear in `available_actions` for some games — trust the list.)
- `action.is_complex()` → True for ACTION6 (needs coordinates).
- ACTION6: `a = GameAction.ACTION6; a.set_data({"x": cx, "y": cy})` with `cx, cy ∈ [0, 63]`.
- `GameAction.from_id(i)` builds from an id; `action.reasoning = {...}` attaches metadata (optional).
- When `state is GAME_OVER`, the only legal action is `RESET` (anything else → 400 on the live API).

## Scoring (from occam/solver/rhae.py — the model we optimize against)
- Per level: `S = min(CAP, (human_baseline / ai_actions) ** 2)`. occam uses `CAP = 1.0`; the official report says
  **1.15**. Optimize for **CAP=1.15** but never assume > human.
- `ai_actions` is **cumulative across resets/replays** (occam counts `total_steps`). Exploration is NOT free.
- Game score: `sum((i+1) * S_i) / sum(1..n)` — **1-indexed level weighting** (deep levels dominate).
- Give-up budget ≈ `baseline * max(5.0, n_actions/baseline + 3.0)` (≈ 5× human) — solve before this or score 0.
- Total = average of game scores over the hidden set → 0–100%.

## File ownership (no overlapping edits)
| Path | Owner | Notes |
|---|---|---|
| `agent/my_agent.py` | Solver | **Self-contained** (spliced into the Kaggle notebook). Perception + policy inline. |
| `eval/` | Eval | rhae port, frozen holdout, coverage/variance harness. Dev-only (not spliced). |
| `experiments/` | Experiments | reset-counting + determinism probes. Dev-only. |
| `tests/` | all | pytest; import pure helpers from `agent/my_agent.py`. |
| `scripts/`, `Makefile`, `notebooks/` | Harness | Kaggle lock-step (from the starter). |
| `vendor/` | — | reference clones; gitignored; never edited in place. |

## Coding rules (reproducibility is a scored gate)
- Deterministic: seed from `game_id` only (never wall-clock); `PYTHONHASHSEED=0`; sort all iteration; single-thread.
- Offline: no `requests`/`urllib`/`httpx`/`huggingface_hub`/`torch.hub`/`socket` in `agent/my_agent.py`.
- Imports allowed in `my_agent.py`: stdlib, `numpy`, `arcengine`, `agents.agent`. Nothing else.
