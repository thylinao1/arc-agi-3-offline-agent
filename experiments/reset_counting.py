"""Reset-counting experiment (conclusive version).

THE question that forks the architecture (see DECISION-LOG.md): does the grader's per-level
action count reflect only the successful path, or does it count ALL actions on first
exposure (exploration + resets + replays)?

Now that the agent actually SOLVES some games (vc33/lp85/sp80) — and it solves them via
reset-and-replay, which spends MANY actions resetting and replaying before the win — we can
answer it directly. Run the real agent on a solvable game and read the scorecard's per-level
`level_actions` for the solved level:

  - If level_actions ≈ total actions taken (hundreds, incl. resets) → CUMULATIVE: exploration
    is paid for; minimize first-exposure (occam's model, our current design).
  - If level_actions ≈ the short human-comparable path → resets are NOT counted; explore-then-
    replay is "free" and we should switch strategy.

    python experiments/reset_counting.py --game vc33
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402

from agent.my_agent import MyAgent  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default="vc33")
    p.add_argument("--max-steps", type=int, default=800)
    args = p.parse_args()

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(args.game)
    MyAgent.MAX_ACTIONS = args.max_steps
    agent = MyAgent(
        card_id="reset-exp", game_id=args.game, agent_name=f"MyAgent.{args.game}",
        ROOT_URL="http://localhost", record=False, arc_env=env, tags=["reset-exp"],
    )
    agent.main()

    final = agent.frames[-1]
    print(f"\nagent: total actions taken = {agent.action_counter}, "
          f"levels_completed = {final.levels_completed}, state = {final.state}")

    sc = arc.get_scorecard()
    data = sc.model_dump() if hasattr(sc, "model_dump") else sc
    cards = data.get("cards", []) if isinstance(data, dict) else []
    # Per-card game_id is null (cards are keyed by guid); pick the one with real level data.
    card = next((c for c in cards if c.get("level_baseline_actions")), None)
    if card is None and cards:
        card = cards[0]
    if card is None:
        print("No scorecard card found.")
        print(json.dumps(data, indent=2, default=str)[:1500])
        return

    print("\n=== scorecard card (the grader's accounting) ===")
    for k in ("actions", "resets", "levels_completed", "level_count",
              "level_actions", "level_baseline_actions", "level_scores", "score"):
        if k in card:
            print(f"  {k}: {card[k]}")

    la = card.get("level_actions") or []
    base = card.get("level_baseline_actions") or []
    print("\n=== verdict ===")
    if la and any(la):
        solved_costs = [a for a in la if a]
        print(f"  per-level actions recorded: {la}")
        print(f"  human baselines:            {base}")
        if max(solved_costs) > 50:
            print("  → level_actions are LARGE (incl. resets/replays) ⇒ CUMULATIVE counting.")
            print("    Exploration is paid for; minimize-first-exposure is correct (current design).")
        else:
            print("  → level_actions are SMALL (≈ solution path only) ⇒ resets NOT counted.")
            print("    Explore-then-replay is 'free'; consider switching strategy.")
    else:
        print("  No per-level action data (game not solved within budget). Try --game lp85/sp80.")


if __name__ == "__main__":
    main()
