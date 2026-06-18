"""Day-1 GO/NO-GO experiment: how does a LEVEL RESET affect the scored action count?

THE question that forks the architecture (see DECISION-LOG.md):
  - If the grader scores only the BEST / LAST attempt → explore-then-replay is allowed.
  - If it counts CUMULATIVE actions across resets → minimize first-exposure (the default).

This script drives one game two ways and dumps the scorecard so you can compare what the
grader actually counts. The TRUE answer requires ONLINE mode (set ARC_API_KEY in .env) so
you read the real three.arcprize.org scorecard. Locally (NORMAL) it shows the SDK's own
accounting, which is a strong hint but not the authoritative grader.

    python experiments/reset_counting.py --game ls20

What to compare in the printed scorecards:
  A) "clean" run: N actions, no extra resets.
  B) "wasteful" run: ~30 throwaway actions, a RESET, then the same N actions.
  If B's reported per-level action count ≈ N  → reset zeros/best-selects → enable replay.
  If B's reported count ≈ 30 + N            → cumulative            → minimize-first-exposure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from agents.agent import Agent  # noqa: E402


class _ScriptedAgent(Agent):
    SCRIPT: list[str] = []
    MAX_ACTIONS = 10_000

    def is_done(self, frames, latest_frame) -> bool:
        return self.action_counter >= len(self.SCRIPT) or latest_frame.state is GameState.WIN

    def choose_action(self, frames, latest_frame):
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET
        return GameAction.from_name(self.SCRIPT[self.action_counter])


def drive(arc, game_id: str, script: list[str], label: str) -> int:
    env = arc.make(game_id)
    _ScriptedAgent.SCRIPT = script
    a = _ScriptedAgent(
        card_id=f"exp-{label}", game_id=game_id, agent_name=f"reset-{label}",
        ROOT_URL="http://localhost", record=False, arc_env=env, tags=["exp", label],
    )
    a.main()
    print(f"[{label}] action_counter={a.action_counter} "
          f"levels_completed={a.frames[-1].levels_completed} state={a.frames[-1].state}")
    return a.action_counter


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default="ls20")
    args = p.parse_args()

    if not os.getenv("ARC_API_KEY"):
        print("NOTE: ARC_API_KEY not set — running NORMAL/local. The SDK scorecard below is a "
              "hint, not the authoritative grader. Re-run ONLINE for the real answer.\n")

    cycle = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]
    base = [cycle[i % len(cycle)] for i in range(20)]
    clean = ["RESET"] + base
    wasteful = ["RESET"] + cycle * 6 + ["RESET"] + base  # waste, reset, then the same base

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    drive(arc, args.game, clean, "clean")
    drive(arc, args.game, wasteful, "wasteful")

    sc = arc.get_scorecard()
    print("\n=== scorecard ===")
    try:
        print(json.dumps(sc.model_dump() if hasattr(sc, "model_dump") else sc, indent=2, default=str))
    except Exception:
        print(repr(sc))
    print("\nCompare the per-level action counts for clean vs wasteful (see module docstring).")


if __name__ == "__main__":
    main()
