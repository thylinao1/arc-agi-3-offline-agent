"""Day-1 experiment: is the engine deterministic?

Replays the SAME scripted action sequence twice (each from a fresh make+RESET) and
compares the resulting frame signatures step by step. If they match, open-loop replay is
safe; if not, the agent must use closed-loop re-planning.

Runs locally in NORMAL mode against an anonymous game (no API key needed).

    python experiments/determinism.py --game ls20 --steps 40
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

from agents.agent import Agent  # noqa: E402


def _sig(frame) -> str:
    arr = np.asarray(frame[-1] if frame else [], dtype=np.int16)
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]


class _ScriptedAgent(Agent):
    SCRIPT: list[str] = []
    MAX_ACTIONS = 10_000

    def is_done(self, frames, latest_frame) -> bool:
        return self.action_counter >= len(self.SCRIPT)

    def choose_action(self, frames, latest_frame):
        return GameAction.from_name(self.SCRIPT[self.action_counter])


def run_once(arc, game_id: str, script: list[str]) -> list[str]:
    env = arc.make(game_id)
    _ScriptedAgent.SCRIPT = script
    agent = _ScriptedAgent(
        card_id="exp", game_id=game_id, agent_name="determinism",
        ROOT_URL="http://localhost", record=False, arc_env=env, tags=["exp"],
    )
    agent.main()
    return [_sig(f.frame) for f in agent.frames]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default="ls20")
    p.add_argument("--steps", type=int, default=40)
    args = p.parse_args()

    # A fixed exploratory script: RESET then cycle the simple actions.
    cycle = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]
    script = ["RESET"] + [cycle[i % len(cycle)] for i in range(args.steps)]

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    run_a = run_once(arc, args.game, script)
    run_b = run_once(arc, args.game, script)

    n = min(len(run_a), len(run_b))
    mismatches = [i for i in range(n) if run_a[i] != run_b[i]]
    print(f"game={args.game} steps={args.steps} frames_a={len(run_a)} frames_b={len(run_b)}")
    if not mismatches and len(run_a) == len(run_b):
        print("RESULT: DETERMINISTIC ✓  (open-loop replay is safe)")
    else:
        print(f"RESULT: NON-DETERMINISTIC ✗  first divergence at step {mismatches[0] if mismatches else n}")
        print("  → use closed-loop re-planning; do not bet on open-loop replay")


if __name__ == "__main__":
    main()
