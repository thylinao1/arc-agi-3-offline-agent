# SPEC — ARC-AGI-3 Offline Agent (the 5-field kernel)

Canonical scope contract. Build agents code against this + `CONTRACT.md`. Append changes to `DECISION-LOG.md`.

## Why
Frontier AI scores <1% on ARC-AGI-3; humans ~100%. The prize track (ARC Prize 2026, Kaggle
`arc-prize-2026-arc-agi-3`) runs **fully offline** — so the API-driven agents topping the *community* board are
ineligible. The eligible field is thin and symbolic; the best public, offline, MIT baseline (`occam`) scores
**57.6% RHAE**. Coverage on unseen games is unsolved and pays $850K. Milestone #1: **June 30, 2026**.

## Capabilities (measurable success criteria — tech-agnostic where possible)
1. **Runs fully offline.** No network, no hosted-model API calls at scoring time. Weights (if any) load from a
   pre-packaged dataset. (Disqualifier if violated.)
2. **Scores on the real grader.** A submission produces a valid `submission.parquet` and a non-zero leaderboard
   score. *(Day-2 vertical-slice gate.)*
3. **Coverage first.** Maximizes the count of games/levels scored > 0 on **unseen** games. (Primary objective; an
   unsolved level = hard 0.)
4. **Efficient enough, not optimal.** On solved levels, satisfices to ai_actions ≈ human/1.07 (the 1.15× cap),
   then stops. Never blows the ~5× human-action cutoff exploring.
5. **Level-weighted.** Spends the action budget proportional to the 1-indexed level number (deep levels weigh more).
6. **Deterministic & reproducible.** Identical output on reruns with a fixed seed; required for the open-source gate.

## Constraints
- Python **3.12**; `arc-agi` SDK + `ARC-AGI-3-Agents` framework; `MyAgent(Agent)` is the entry point.
- Offline at scoring; ~600 RPM dev rate limit; Kaggle notebook runtime; **open-source under MIT/CC0 before private
  scores**.
- `agent/my_agent.py` must be **self-contained** (it is spliced into the Kaggle notebook; sibling files are NOT).

## Non-goals (the cut list — do not build for Milestone #1)
- Online-learned `s'=f(s,a)` neural world model; planning inside a learned model.
- Neural / open-weights perception model (connected-components is exact and cheaper).
- Shortest-path optimization (BFS/A*/IDDFS to optimal) — the 1.15× cap makes it worthless.
- Reset-replay as a *load-bearing* assumption (it is an optional layer, gated on the day-1 experiment).
- Cross-level memory beyond the floor (Milestone #2, gated on holdout delta > variance).

## Success signal
Day 2: a real non-zero score from an offline-clean notebook. Day 4: an occam-equivalent floor score. By day 10:
coverage (games scored > 0) on the **frozen public holdout** beats the stock occam-fork by more than measured
variance — or we consciously ship the occam floor.
