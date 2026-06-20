# ARC-AGI-3 Offline Agent

An agent for the [ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
competition. It plays novel, instruction-free, turn-based grid games and tries to solve as many as it can,
running fully offline.

## The short version of the strategy

The prize track scores agents offline, with no internet at evaluation time. That rules out anything that calls a
hosted model like GPT or Claude. So this agent carries **no pretrained model and makes no network calls** — it
learns a small CNN *online, from scratch, during play* on each game.

The scoring metric drives the whole design. Each level scores `min(1.15, (human_actions / ai_actions)^2)`, averaged
over the hidden games (an unsolved level is a hard zero). The square makes this **dominated by efficiency**: you
have to solve a level in roughly a human's number of actions for it to score at all. An agent that *can* solve a
level but takes 10–100× the actions scores ≈ 0. (We learned this the hard way — see `DECISION-LOG.md`: a
coverage-first reset-replay agent that "solved" 9/25 games scored **0.00**, because resets and replays are counted
and crater the action ratio.)

So the agent is **reactive**: one action per step, no resetting-and-replaying to search. It learns which actions
change the world and steers toward new states, keeping its action count close to a human's.

## How it solves

`agent/my_agent.py` is a reactive CNN agent (adapted from the official Kaggle "Stochastic Goose" sample,
Apache-2.0 — see `NOTICE`):

- Each 64×64 frame is one-hot encoded to 16 colour channels.
- A small CNN (`ActionModel`) predicts, for the current frame, which of ACTION1–5 and which ACTION6 click
  coordinate is most likely to *change* the frame. Action selection samples from those predictions, masked to the
  currently available actions.
- After every step it records whether the chosen action actually changed the frame, and trains the CNN on that
  signal (online, per game). The model resets between levels.
- It plays purely forward (reset only on game-over), so action counts stay near human — which is what the RHAE
  metric rewards.

This is an online-learning agent with no pretrained weights and no internet, so it is competition-legal for the
offline prize track. It runs on a single GPU (the Kaggle submission uses a Tesla T4).

## Layout

```
agent/my_agent.py     The agent (reactive online-learning CNN). Self-contained — spliced into the Kaggle notebook.
eval/                 RHAE scoring port + coverage/variance helpers.
experiments/          Day-1 probes: reset-counting semantics, determinism.
tests/                pytest unit tests for the model + action sampling.
scripts/, Makefile    Local dev and Kaggle submission, from the official starter.
vendor/               Reference clones (ARC-AGI-3-Agents, Kaggle starter). Gitignored.
SPEC.md, CONTRACT.md  Scope and interface seams. DECISION-LOG.md tracks every choice (incl. the 0.00 post-mortem).
```

## Run it

```bash
make setup          # one-time: venv (Python 3.12) + arc-agi + torch + slim the framework
make play-local GAME=ls20 STEPS=400
pytest -q           # unit tests (skip automatically if torch/framework absent)
```

The agent uses PyTorch; locally it runs on CPU (slow — a few games for sanity), and on Kaggle it uses the GPU.
Online scorecards and the full game set need an API key from https://three.arcprize.org (put it in `.env` as
`ARC_API_KEY`). Submit to Kaggle with `make submit` (the kernel is configured for a Tesla T4, internet off).

## Status

Reactive online-learning CNN, fully offline (no pretrained weights, no internet, no hosted LLM), GPU on Kaggle.
The earlier reset-replay symbolic agent scored 0.00 on the live leaderboard because the metric rewards action
efficiency, not raw coverage; this agent is the efficiency-first redirection. See `DECISION-LOG.md` for the full
history and the leaderboard evidence.

## License

MIT for this repository (see `LICENSE`), except `agent/my_agent.py`, which is Apache-2.0 (adapted from the official
Kaggle Stochastic Goose sample — see `NOTICE`).
