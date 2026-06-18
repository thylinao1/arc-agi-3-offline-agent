# DECISION-LOG

Append-only. Every agent reads this on start and appends ONE line (date — decision — why) on finish.
Newest at the bottom.

---

- 2026-06-18 — **Track = ARC Prize 2026 ARC-AGI-3 (Kaggle), offline.** Why: the prize track disables internet at
  scoring; API-driven community leaders are ineligible, so the eligible field is thin/symbolic. (See HACKATHON-BATTLEPLAN.md §2.)
- 2026-06-18 — **Primary objective = coverage (games/levels scored > 0 on unseen games), efficiency second.** Why:
  total averages over ~110 hidden games; one unsolved level = hard 0; the RHAE efficiency multiplier is capped at
  1.15× and only moves already-solved levels. (Council + skeptic.)
- 2026-06-18 — **Satisfice, don't optimize.** Stop at ai_actions ≈ human/1.07; redirect compute to solving MORE
  levels. Why: `min(1.15, ratio²)` → zero marginal reward past ~7% better than human.
- 2026-06-18 — **Architecture = minimize-first-exposure (default); reset-replay is OPTIONAL, gated on the day-1
  experiment.** Why: human protocol (no revisiting completed levels; reset = within-level do-over) + the 5× cutoff
  + occam counting `total_steps` cumulatively all argue exploration is NOT free.
- 2026-06-18 — **Scoring model confirmed from `occam/solver/rhae.py`:** `min(1.0, human/ai)²` per level (official
  cap 1.15), `sum((i+1)·S)/sum(1..n)` 1-indexed weighting, give-up ≈ 5× baseline. This validates the level-weighting
  lever and the cumulative-action assumption. We optimize for CAP=1.15.
- 2026-06-18 — **`agent/my_agent.py` must be SELF-CONTAINED.** Why: `scripts/build_notebook.py` splices ONLY this
  file into the Kaggle notebook (`%%writefile`), and the offline install cell pulls only `arc-agi` + `python-dotenv`.
  Perception (connected-components), state hashing, and the explorer are INLINE in my_agent.py; sibling files are
  dev/test-only. This overrides the battle plan's "separate perception.py" — tests import helpers from my_agent.py.
- 2026-06-18 — **CUT from Milestone #1:** online-learned world model, learned-model planning, neural/open-weights
  perception, shortest-path optimization. All → Milestone #2, gated on "beats occam-fork on the frozen holdout by >
  measured variance."
- 2026-06-18 — **First cut shipped:** a deterministic stateful frontier explorer in my_agent.py — effective-action
  detection + **centroid-restricted ACTION6** (vs random 0–63) + per-state untried-action frontier + level-aware
  budget/satisficing. Better-than-random, offline, seeded by game_id. Next: port occam's ReplayExplorer
  (reset+replay BFS w/ incremental replay + step-modulus hashing) as the Solver upgrade.

- 2026-06-18 — **Vertical slice PROVEN (day-2 gate, early):** `make`-style venv (Python 3.12.13) + `arc-agi`
  installed; 9/9 unit tests pass; `scripts/play_local.py` runs `my_agent` on `ls20` fully offline at ~630 fps,
  auto-fetches an anonymous key, pulls 25 environments, and produces a scorecard. End-to-end pipeline works.
  Score is 0.0 in 120 steps (first-cut frontier explorer does not yet solve ls20 — that's the Solver upgrade).
- 2026-06-18 — **Runtime fix:** `latest_frame.available_actions` arrives as `list[int]` (action ids), not
  `GameAction`; normalized via `GameAction.from_id`. (CONTRACT.md "list[GameAction]" was the documented type;
  runtime is ids.)

## OPEN — day-1/3 experiments that gate the architecture (run via experiments/)
- [ ] **Reset-counting (GO/NO-GO):** does the grader's per-level action count reflect BEST / LAST / CUMULATIVE
      attempt, and do actions accumulate across resets toward the 5× cutoff? → `experiments/reset_counting.py`.
      Default to minimize-first-exposure unless this cleanly proves reset zeros/best-selects. (Needs ONLINE mode +
      ARC_API_KEY for the authoritative grader; local scaffold ready.)
- [x] **Determinism: CONFIRMED ✓** (2026-06-18) — `experiments/determinism.py --game ls20 --steps 40`: identical
      frame signatures across two replays. Open-loop replay is safe → exact hash-keyed transition table is valid.
- [ ] **First-exposure cost vs 5× cutoff:** measure naive solve cost vs the give-up budget on 2–3 public games.
- [ ] **Offline audit:** grep `agent/my_agent.py` for `requests|urllib|httpx|huggingface_hub|torch.hub|socket`;
      run the notebook with internet disabled locally before submitting.
