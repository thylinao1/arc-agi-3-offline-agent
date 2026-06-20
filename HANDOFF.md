# HANDOFF — ARC-AGI-3 agent (state as of 2026-06-19)

Authoritative "where things are" for a fresh conversation. Read this + `DECISION-LOG.md` first.

## TL;DR
A **fully-offline ARC-AGI-3 agent** for the ARC Prize 2026 Kaggle track (`arc-prize-2026-arc-agi-3`, $850K,
Milestone #1 = 2026-06-30). The repo is **open-sourced** (public, MIT) at
**https://github.com/thylinao1/arc-agi-3-offline-agent** (milestone eligibility locked).

**⚠️ MAJOR PIVOT (2026-06-19→20).** The first submission — a hybrid reset-replay symbolic agent (occam + step-wise
BFS) we thought solved "9/25" — scored **0.00** on the live leaderboard. Root cause: the RHAE metric
(`min(1.15,(human/ai)²)`) is **efficiency-dominated**; reset-replay burns 10–100× human actions per level → ~0 even
when it "solves," and occam's solves aren't even credited by the scorecard (verified: hybrid=0.0136 vs
step-wise-only=0.1021 vs the leaderboard frontier of 0.43–1.21). **The whole "coverage-first" thesis was wrong.**

**Current agent (the fix): a REACTIVE online-learning CNN** in `agent/my_agent.py`, adapted from the official
Kaggle "Stochastic Goose" sample (Apache-2.0; the same approach scores **0.43** on the leaderboard). One action per
step, no reset-replay → near-human action counts → real RHAE. No pretrained weights, no internet (competition-legal
offline); trains a small CNN online per game. Runs on **GPU (Tesla T4)** on Kaggle. occam is fully removed; tests
rewritten (6 passing, torch-gated). Notebook rebuilt for T4. **Next: resubmit and confirm a nonzero score.**

## The agent (how it plays) — `agent/my_agent.py`
A **reactive online-learning CNN** (adapted from the official Kaggle "Stochastic Goose" sample, Apache-2.0). It uses
the framework's standard per-step hooks (`choose_action`/`is_done`), NOT a custom `main()`:
1. Each 64×64 frame → one-hot 16-colour tensor. `ActionModel` (a small CNN) predicts which of ACTION1–5 and which
   ACTION6 click coordinate is most likely to *change* the frame; selection samples those, masked to the available
   actions.
2. After each step it records whether the action changed the frame and trains the CNN online (per game); the model
   resets between levels. Plays forward, resets only on game-over → action counts stay near human → real RHAE.
- No pretrained weights, no internet, no hosted LLM (offline-legal). Self-protecting: any exception in
  `choose_action` falls back to a random action. Runs on GPU (Tesla T4) on Kaggle.

## Key findings (why we pivoted)
- **The metric is EFFICIENCY-dominated.** RHAE=`min(1.15,(human/ai)²)`, level-weighted. A level solved in 10–100×
  human actions scores ≈0. **Coverage without efficiency = 0.** This invalidated the prior "coverage-first" design.
- **The hybrid reset-replay agent scored 0.00** on the live board (rank ~1227/1303). Measured locally:
  hybrid (occam-first)=**0.0136**, step-wise-only=**0.1021**, leaderboard frontier=**0.43–1.21**. occam's "solves"
  are NOT credited by the scorecard (wa30 alone: occam `levels=9` → scorecard 0.0) AND occam blocked the step-wise
  agent's efficient solves → it made the score 7.5× worse. The "9/25 coverage" was occam's internal counter, never
  RHAE. Submission graded "Succeeded" (no crash) → 0.00 is the true RHAE.
- **The frontier is REACTIVE agents** (one action/step, ≈human action counts). StochasticGoose (the approach we
  adopted) = 0.43. Determinism CONFIRMED earlier; human baselines tiny: vc33=[7,18,44,…], ls20=[84,96,192,…].

## File map
```
agent/my_agent.py        The agent: reactive online-learning CNN (Apache-2.0; self-contained for Kaggle splice)
scripts/build_notebook.py Builds notebooks/submission.ipynb (splices my_agent.py only); ACCELERATOR="t4" (GPU)
scripts/play_local.py    Local runner (NORMAL mode; needs torch — `make setup` installs CPU torch)
eval/rhae.py             RHAE scoring + coverage/variance helpers (dev)
experiments/             determinism.py (resolved: deterministic), reset_counting.py (resolved: cumulative)
tests/test_my_agent.py   6 unit tests (ActionModel shape, frame→tensor, action masking, experience hash)
SPEC.md CONTRACT.md      5-field spec + interface seams (verified against the real SDK)
DECISION-LOG.md          Full design history + every empirical finding (append-only)
HACKATHON-BATTLEPLAN.md  Original strategy/battle plan
vendor/                  Cloned refs (ARC-AGI-3-Agents framework, Kaggle-Starter, occam) — gitignored
.env                     ARC_API_KEY (gitignored)
.kaggle/access_token     Kaggle token (gitignored, never committed)
notebooks/kernel-metadata.json  Kaggle kernel config (id=maksimsilchenko/…, CPU, internet off)
~/.claude/skills/ecc/hack-arc-agi-3/SKILL.md   Domain-authority skill (API, scoring, architecture)
```

## Run / test / benchmark (venv = `.venv`, Python 3.12; `set -a; source .env; set +a` first)
```bash
.venv/bin/python -m pytest -q                                  # 21 tests
.venv/bin/python scripts/play_local.py --game ls20 --max-steps 200   # run the agent on a game (NORMAL/local)
ARC_DEBUG=1 .venv/bin/python scripts/play_local.py --game sp80,vc33 --max-steps 800  # see [occam]/[route]/[nav] traces
.venv/bin/python scripts/bundle_occam.py                       # regenerate the occam bundle
# occam-portable benchmark (7/25): /tmp/occam_full.py exists; or run solver.benchmark from vendor/occam
```
Note: `play_local` per-game summary reads `agent.frames`, which occam does NOT populate (occam resets the env at
game end). Trust the **scorecard** / occam's internal `[occam] …: levels=N` debug, not the per-game print.

## HOW TO SUBMIT (everything is wired; this is the only remaining step — a user action)
```bash
cd "/Users/maksimsilchenko/ARC-AGI-3 Competition"
make submit     # rebuilds the notebook from agent/*.py and pushes to Kaggle (uses .kaggle/access_token)
make status     # watch the run
```
Then on **kaggle.com/competitions/arc-prize-2026-arc-agi-3** → your notebook → **Submit** to trigger Phase B
(the hidden-game eval that produces the leaderboard score). 5 submissions/day. `make setup` is NOT required — the
`.venv` already has arc-agi + kaggle + pandas, and the framework is in `vendor/`.
- **Prize/milestone eligibility:** open-source the repo under MIT/CC0 **before** private scores. We are MIT
  (`LICENSE`) + occam attributed (`NOTICE`). Push to a public GitHub repo before the milestone to qualify.

## Credentials (both gitignored, present locally)
- `.env` → `ARC_API_KEY` (anonymous + the user's key).
- `.kaggle/access_token` and `~/.kaggle/access_token` → Kaggle token (verified: `userHasEntered=True`).

## Open work / next directions (rough value order)
1. **Submit** (above) and confirm a real leaderboard score.
2. **Goal inference under the action-cost constraint** — the only path past ~9/25. The deepcopy-only games are
   fundamentally hard on the real competition (no free search); progress needs inferring each game's goal from the
   grid + sparse level-complete signal, not more reset-replay search.
3. Per-game tuning of the available-but-off step-wise levers (`prune_dead_simples`, `step_modulus`).

## Commits (newest first)
`fd22961` Kaggle config (CPU+username) · `a33ceea` hybrid occam+step-wise → 9/25 · `eec74b4` combo-cache+budget
diagnostic · `c626f24` arrow-fix+collect-all → 0.1021 · `f889d2f` goal-hypothesis nav → 6× · `cbdcec2` reactive
nav+pathfinding · `0e24596` efficiency levers + reset-counting resolved · `f8c8b08` BFS+occam-perception → 3/25 ·
`1262587` DFS+counter-mask · `23c7663` scaffold + proven vertical slice.
