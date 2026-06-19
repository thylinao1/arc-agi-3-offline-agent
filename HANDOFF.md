# HANDOFF — ARC-AGI-3 agent (state as of 2026-06-19)

Authoritative "where things are" for a fresh conversation. Read this + `DECISION-LOG.md` first.

## TL;DR
A **hybrid, fully-offline, deterministic ARC-AGI-3 agent** for the ARC Prize 2026 Kaggle track
(`arc-prize-2026-arc-agi-3`, $850K, Milestone #1 = 2026-06-30). It solves **9/25 public games** — the best
Kaggle-portable result we reached. Everything is committed, tested (21/21), and **submission-ready** (token +
username wired, notebook built for CPU). Only `make submit` remains, and it's an explicit user action.

## The agent (how it plays) — `agent/my_agent.py`
`MyAgent.main()` runs a hybrid, in priority order, all reset-replay (no internet, no LLM, no GPU):
1. **occam** (`agent/occam_bundle.py`) — Sean Donahoe's MIT $0 solver, run with `skip_deepcopy=True`. Wins the
   **7 movement games**: re86, m0r0, cn04, dc22, sp80, ka59, wa30.
2. **Step-wise fallback** (if occam solves nothing) — our symbolic agent: warmup→mask→navprobe→navhypo→
   clickscan/combosearch→BFS. Wins **2 non-movement games occam-portable misses**: vc33, lp85.
- Net **9/25**, verified. Always offline-safe: any occam error → the step-wise fallback.

## Key findings (the real intelligence)
- **occam's headline 17/25 does NOT transfer to Kaggle.** It relies on deepcopy-BFS, which clones the env to
  search *for free* (clones don't consume actions). The real gateway can't be deepcopied — every probe is a
  counted action — so the portable ceiling is **~7/25 (occam) / ~9/25 (hybrid)**. Reset-replay agents look capped
  near here on the real (cumulative-scoring, no-free-search) competition.
- **Scoring = cumulative RHAE** (`min(1.15,(human/ai)²)` per level, 1-indexed weighting, ~5× cutoff). Confirmed by
  experiment: resets/exploration ARE counted → minimize-first-exposure. **No per-frame reward** (only the grid +
  sparse `levels_completed`).
- **Determinism CONFIRMED** (open-loop replay is safe). Human baselines tiny: vc33=[7,18,44,…], ls20=[84,96,192,…].
- Every bounded step-wise lever (nav, pathfinding, collect-all, combo-caching, efficiency pruning) added **0**
  public coverage on top of 3/25; coverage gains came only from running occam. Best step-wise-alone RHAE = 0.1021.

## File map
```
agent/my_agent.py        The agent: hybrid main() + the step-wise fallback solver (self-contained for Kaggle splice)
agent/occam_bundle.py    occam's MIT solver, flattened to one module (GENERATED — see scripts/bundle_occam.py)
scripts/bundle_occam.py  Regenerates occam_bundle.py from vendor/occam/solver/*.py
scripts/build_notebook.py Builds notebooks/submission.ipynb (splices my_agent.py + occam_bundle.py); ACCELERATOR="cpu"
scripts/play_local.py    Local runner (NORMAL mode, no key needed for the 3 anonymous games)
eval/rhae.py             RHAE scoring + coverage/variance helpers (dev)
experiments/             determinism.py (resolved: deterministic), reset_counting.py (resolved: cumulative)
tests/test_my_agent.py   21 unit tests (perception, ReplaySearch BFS, ReactiveNav, clickscan/combosearch)
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
