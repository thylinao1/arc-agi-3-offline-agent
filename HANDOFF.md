# HANDOFF — ARC-AGI-3 agent (state as of 2026-06-21)

Authoritative "where things are" for a fresh conversation. Read this + `DECISION-LOG.md` (+ the auto-memory) first.

## TL;DR (current)
**Fully-offline ARC-AGI-3 agent** for ARC Prize 2026 (`arc-prize-2026-arc-agi-3`, $850K, Milestone #1 = 2026-06-30).
Public repo (MIT): **https://github.com/thylinao1/arc-agi-3-offline-agent** (eligibility locked). Git clean + pushed
(`270febb`). 6 tests pass.

**Leaderboard so far:** v1 (hybrid reset-replay) = **0.00**; v2 (reactive online CNN / StochasticGoose) = **0.08**;
**v3 (honest imitation warm-start student) = SUBMITTED 2026-06-21 ~09:44, status PENDING** (ref 53899428) — Phase B
running. **CHECK THE v3 SCORE FIRST:**
`KAGGLE_API_TOKEN="$(cat .kaggle/access_token)" .venv/bin/kaggle competitions submissions arc-prize-2026-arc-agi-3 | head -4`

**The goal is TOP-5 (> 0.66).** Honest paths explored cap well below that (online CNN ~0.25–0.32; our imitation
data is hard-capped ~93 demos; the source-reading "BFS solvers" hit 0.42–0.46 but are a leakage exploit, likely
prize-INELIGIBLE). >0.66 honestly needs far more/deeper demos than the offline teacher can produce (confirmed: BFS
*and* beam/A*/MCTS all fail the same hard games; more time doesn't deepen). So v3 is a genuine, prize-eligible
datapoint, NOT a frontier contender.

**Current agent = honest imitation student** (`agent/my_agent.py`): the StochasticGoose reactive CNN, **warm-started
from behavior-cloned weights** (`_load_pretrained` ← `/kaggle/input/arc-agi3-student-weights/student.pt`, the
trained imitation model, 98.7% train-acc on 93 teacher demos), then learns online. Plays HONESTLY (no game-source
access). GPU T4. Weights shipped via the Kaggle dataset `maksimsilchenko/arc-agi3-student-weights` (wired in
kernel-metadata `dataset_sources`).

## Pipeline (Track 2 — honest, prize-eligible)
- `scripts/gen_demos.py` — genuine offline BFS teacher (in gitignored `reference/teacher_agentv15.py`) → real
  (frame,action) demos on PUBLIC games only → `data/demos/demos.npz` (93 demos, 10 games, levels 0–1). Audit-verified
  (solutions transfer to real env; encoding `5+y*64+x` round-trips; pre-action frames).
- `scripts/train_student.py` — behavior cloning + color-aug; `--resume`, checkpoints every 10 epochs. Trains the
  ActionModel; saved `models/student.pt` (gitignored, 137MB).
- `agent/my_agent.py` `_load_pretrained` — warm-start; falls back to pure-online if no weights.
- `scripts/build_notebook.py` — splices my_agent.py; ACCELERATOR="t4"; kernel-metadata carries the weights dataset.

## NEXT STEPS (in priority order)
1. **Read the v3 leaderboard score** (command above). If v3 > 0.08/0.25 → imitation warm-start helps; if ≈ → it
   didn't (thin data). Either way it's the honest baseline.
2. The ONLY honest lever toward >0.66 is **much more/deeper demos** — the offline teacher is the bottleneck (caps
   ~10 games). Ideas not yet tried: longer per-game budgets + transfer across levels; a smarter goal-directed
   teacher; or accept the ceiling. (Track 1 hedge: the exploit 0.46 agent in `reference/agentv15` is ready but
   prize-risky — submit only for a LB datapoint, not the prize.)
3. Submission limit is **2/day**; runtime 9h; GPU RTX6000/T4.

## DEAD ENDS (do not repeat)
- **occam** = FAKE solves (reports wins it never reaches; scorecard credits 0). Deleted.
- **Coverage-first / reset-replay** = ~0 RHAE (efficiency metric). Wrong paradigm.
- **MultiSolver (beam/A*/MCTS)** = no better coverage than plain BFS at equal time.
- **Source-reading exploit** = 0.42–0.46 but leakage / likely prize-ineligible.

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
