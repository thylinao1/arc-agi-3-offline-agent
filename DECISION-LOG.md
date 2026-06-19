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

- 2026-06-19 — **API key wired** (user registered + provided two keys; using `c2df11…`, the other is spare).
  Stored in gitignored `.env` (perms 600). The auto-fetched anonymous key already gives 25 local NORMAL-mode games.
- 2026-06-19 — **Solver upgraded: occam's reset-and-replay ported as a step-wise DFS** (`ReplayDFS` in
  my_agent.py). Validated on simulated deterministic graphs (solves deep paths, descends without needless resets,
  backtracks, handles unsolvable). 13/13 unit tests pass.
- 2026-06-19 — **Counter/status-bar masking added** (`volatility_mask`): a short warmup masks cells that change on
  EVERY transition (a counter), keeping the play area. Helped (ACTION3/4 now appear) but insufficient on ls20 etc.
- 2026-06-19 — **HONEST BASELINE: 0 levels solved** across ls20/ft09/vc33/tn36 (≤400 steps). Root cause:
  state-dedup still fails — the play area animates enough that the generic mask bails, so the DFS sees every frame
  as "new" and degenerately descends via ACTION1 instead of exploring breadth. This is the real frontier; occam
  reaches 57.6% only with per-game counter detection + priority-tier action ranking, and still solves just 17/25.
- 2026-06-19 — **ls20 human baselines (from scorecard) ≈ [84, 96, 192, 186, …] over 7 levels.** So the 5× cutoff
  (~420 on level 1) is generous; the gap is solver quality, not budget.

## OPEN — day-1/3 experiments that gate the architecture (run via experiments/)
- [~] **Reset-counting (GO/NO-GO): INCONCLUSIVE** (2026-06-19) — ran `experiments/reset_counting.py` with the key;
      it exposed the scorecard schema (`level_actions`, `level_baseline_actions`, `resets` tracked separately from
      `actions`) but could NOT resolve best/last/cumulative because the scripted agent solved 0 levels (no scored
      level to read). Needs a working solver to complete a level cleanly vs wastefully. Default stands:
      minimize-first-exposure (occam counts cumulatively and still works).
- [x] **Determinism: CONFIRMED ✓** (2026-06-18) — `experiments/determinism.py --game ls20 --steps 40`: identical
      frame signatures across two replays. Open-loop replay is safe → exact hash-keyed transition table is valid.

- 2026-06-19 — **Rewrote the solver: BFS reset-replay + occam's perception ported.** Geometric status-bar/counter
      masking (`identify_status_bars`: edge-hugging thin/twinned segments) + 5-tier salience click ranking
      (`priority_click_targets`) + effective-action-first ordering. BFS (breadth before depth) finds shallow wins
      the old DFS dove past. Fixed an `@dataclass`/importlib crash (plain `Segment` class). 13/13 tests pass.
- 2026-06-19 — **OFF ZERO ✓ — first solves.** Coverage scales with budget: 250 steps → **1/25** (vc33);
      800 steps → **3/25** (vc33, lp85, sp80), aggregate RHAE 0.0145. The pipeline genuinely solves now.

- 2026-06-19 — **Ported the efficiency levers** (effective-action PRUNING, `max_unique_states` cap, `step_modulus`
      depth-keying, per-level give-up budget). **EMPIRICAL FINDING: on the public set they are net-NEGATIVE for
      COVERAGE.** Dead-simple pruning dropped sp80 (some directional action is effective only in later states);
      cutting clicks 24→10 dropped vc33+sp80. Coverage (solve rate) needs BROAD exploration; these levers only buy
      action-efficiency, which the 1.15 cap makes secondary. So all are wired + unit-tested but **defaulted to the
      coverage-optimal config**: pruning OFF, `step_modulus`=1 (no-op), clicks=24, give-up=MAX_ACTIONS. Coverage
      held at **3/25 @800 steps (vc33, lp85, sp80), aggregate 0.0145**. 16/16 tests. (occam benefits from pruning
      only inside its full multi-solver pipeline + dense click scan, not a BFS-only agent.)
- 2026-06-19 — **RESET-COUNTING RESOLVED: CUMULATIVE ✓.** The agent SOLVES vc33 level 1 (801 actions) yet
      solved-game RHAE is ~0.01–0.05 (aggregate 0.0145), nowhere near the 1.15 cap. vc33 human baselines =
      [7, 18, 44, 61, 131, 34, 152] (level 1 = **7** actions!). Only cumulative counting, (7/hundreds)²≈0, matches
      the observed low score; if resets were free, solved levels would score near the cap and the aggregate would be
      ~0.3–1.0. ⇒ resets/exploration ARE counted ⇒ **minimize-first-exposure is correct** (no architecture change).

- 2026-06-19 — **Ported reactive navigation + logical-grid pathfinding** (`ReactiveNav.probe/next_move/plan_path`).
      Wired as phases BEFORE the BFS fallback: warmup → navprobe → (pathfind → greedy) → bfs, with fast bail-to-BFS
      on cursor death / stall. **FINDING: they ACTIVATE correctly but solve 0 NEW public games.** Probe detects a
      cursor + arrow map on ls20/m0r0/sp80/wa30; pathfinding finds BFS routes on ar25/cn04/dc22/ka59/re86/sp80/wa30
      and interacts on arrival — but the public games' goals are richer than "reach the rarest-color tile + press
      interact" (mazes with non-obvious goals, sequences, pushing). **No regression: 3/25 holds, fallbacks verified;
      19/19 tests.** The solvers stay — they'll crack matching simple-movement games on the hidden ~110-game set
      (occam keeps them for the same reason). A `ARC_DEBUG=1` env var prints nav activation for diagnostics.

## NEXT (to occam's 17/25 — goal inference + the rest of occam's pipeline)
- [ ] **Goal inference** is the real blocker: "rarest-color tile" is the wrong target for most games. Infer the
      goal from what changes when levels complete, or from reward-shaped exploration, not a color heuristic.
- [ ] Port occam's remaining solvers: **combo search** (exhaustive short action sequences), **deepcopy BFS**
      (perfect state cloning, no replay cost), **dense click scan** (find ALL effective click positions),
      `_solve_reactive_click`, and **cross-level winning-combo caching** (reuse level-1 solutions on later levels).
- [ ] Run at the real per-level 5× cutoff (human baselines 7–152) — efficiency matters for SCORE where coverage
      is already reached.
- [ ] **First-exposure cost vs 5× cutoff:** measure naive solve cost vs the give-up budget on 2–3 public games.
- [ ] **Offline audit:** grep `agent/my_agent.py` for `requests|urllib|httpx|huggingface_hub|torch.hub|socket`;
      run the notebook with internet disabled locally before submitting.
