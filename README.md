# ARC-AGI-3 Offline Agent

An agent for the [ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
competition. It plays novel, instruction-free, turn-based grid games and tries to solve as many as it can,
running fully offline.

## The short version of the strategy

The prize track scores agents offline, with no internet at evaluation time. That rules out anything that calls a
hosted model like GPT or Claude. So this agent carries no model in the loop. It is a small, deterministic,
symbolic player.

Three facts about the scoring shape the whole design:

1. **Solving more games matters most.** Score is averaged over about 110 hidden games, and a level you never solve
   is a hard zero. So the first goal is coverage: solve as many games and levels as possible.
2. **Being faster than a human barely helps.** The per-level score is `min(1.15, (human_actions / ai_actions)^2)`.
   Past roughly 7% better than a human it stops paying. So the agent satisfices: find a near-human solution, then
   stop, and spend the saved effort on the next game.
3. **Deeper levels are worth more.** A game's score weights each level by its number, so the last level can be
   worth six times the first. The action budget follows that weighting.

The agent perceives each 64x64 grid as colored objects (classical connected components, no neural net), learns
which actions change the world, and explores toward new states. When it clicks, it clicks object centers rather
than random pixels, which collapses 4096 possible clicks down to a handful.

## Layout

```
agent/my_agent.py     The agent. Self-contained (it is spliced into the Kaggle notebook).
eval/                 RHAE scoring port + the frozen-holdout coverage/variance harness.
experiments/          Day-1 probes: reset-counting semantics, determinism.
tests/                pytest unit tests for the perception and exploration helpers.
scripts/, Makefile    Local dev and Kaggle submission, from the official starter.
vendor/               Reference clones (ARC-AGI-3-Agents, Kaggle starter, occam). Gitignored.
SPEC.md, CONTRACT.md  Scope and interface seams. DECISION-LOG.md tracks every choice.
```

## Run it

```bash
make setup          # one-time: venv (Python 3.12) + arc-agi + slim the framework
make verify-local   # 50 steps on ls20 + vc33 (no API key needed for the anonymous games)
make play-local GAME=ls20 STEPS=200
pytest -q           # unit tests (offline, deterministic)
```

Online scorecards and the full game set need an API key from https://three.arcprize.org. Put it in `.env` as
`ARC_API_KEY`. Submit to Kaggle with `make submit` (edit `notebooks/kernel-metadata.json` first).

## How it solves (hybrid)

`MyAgent.main()` runs a two-stage, fully-offline, deterministic solver:

1. **occam** (`agent/occam_bundle.py`) — Sean Donahoe's MIT, $0, no-LLM reset-replay solver. It wins the
   cursor/movement games (~7 of the 25 public games). Its headline 17/25 relies on deepcopy-BFS to search the
   environment for free, which is impossible against the Kaggle gateway (every probe is a counted action), so we
   run it with `skip_deepcopy=True` — the portable ~7/25 configuration.
2. **Step-wise fallback** — if occam solves nothing, the built-in symbolic agent (perception → nav → goal
   hypotheses → clickscan/combosearch → BFS) runs. It wins a couple of non-movement games occam-portable misses.

Net portable coverage is about **9/25**, three times the step-wise agent alone. The agent is always offline-safe:
any occam error falls back to the step-wise solver. The bundle is regenerated with `python scripts/bundle_occam.py`
and shipped to Kaggle by `scripts/build_notebook.py`.

## Status

Hybrid occam + step-wise agent, ~9/25 public games, fully offline and deterministic. See `DECISION-LOG.md` for the
full design history and the finding that occam's 17/25 does not transfer to the real (no-free-search) competition.

## License

MIT. See `LICENSE`. This project bundles occam (MIT, Sean Donahoe) — see `NOTICE`.
