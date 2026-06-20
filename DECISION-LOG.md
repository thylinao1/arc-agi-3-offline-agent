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

- 2026-06-19 — **Goal-hypothesis search** replaced the single rarest-target greedy nav: enumerate candidate target
      colors (rarest-first + a cross-level cached hint), and for EACH, pathfind + interact + test, falling to BFS if
      none win. **RESULT: aggregate RHAE 0.0144 → 0.0843 (≈6×).** sp80 (a movement game) now solves via a short
      pathfinding route (4–15 actions) instead of hundreds of BFS reset-replay actions, so its efficiency score
      shot up. **Coverage unchanged at 3/25** — the win was EFFICIENCY (the scored Y-axis), not coverage. Dropped
      the greedy `next_move` (it solved nothing); 20/20 tests. `ARC_DEBUG=1` prints `HYPO target=… path=…`.

- 2026-06-19 — **Robust arrow detection (semantic fill).** Probe was missing directions blocked at the root (ls20
      saw only up/left/right; ACTION2/down was mislabeled interact). Fix: trust probed directions, fill gaps from the
      ACTION1-4 convention (1=up,2=down,3=left,4=right), require ≥2 confirmed cursor moves to avoid false positives.
      Now ALL movement games get full 4-arrow maps (ls20/m0r0/wa30/re86/cn04/dc22/ka59). **But coverage + score are
      UNCHANGED (3/25, 0.0843)** — decisive finding: even with correct cursor + 4 arrows + pathfinding + multi-target
      hypothesis + interact, these games DON'T solve. Their goals are genuinely NOT "reach a colored tile" (ls20
      pathfinds to its rarest color; reaching it doesn't win). 20/20 tests; no regression; helps blocked-direction
      games on the hidden set.

- 2026-06-19 — **Collect-all hypothesis + ordering tuning.** Added `plan_collect` (greedy nearest-neighbour tour
      visiting ALL tiles of a color) as a goal hypothesis alongside reach-one; refactored pathfinding onto shared
      `_build_logical`/`_bfs`/`_setup_target` helpers. EMPIRICAL: collect adds **0 coverage** on the public set and,
      interleaved/collect-first, REGRESSED efficiency (0.0843→0.0563). Fix = **all reach hypotheses first, then
      collect as a last resort**; collect kept for collect-style games on the hidden set. With that ordering + full
      4-arrow maps (semantic fill) + the representative-tile pick, **aggregate RHAE reached 0.1021** (best yet, ~7×
      the 0.0144 baseline). 21/21 tests. Lesson: nav efficiency is brittle to target-tile + hypothesis-ordering
      choices — locked the best-measured config.

- 2026-06-19 — **Budget diagnostic + winning-combo caching (goal-inference research).** (1) Ran 6 unsolved games
      at 1000 actions (>> the real ~5× cutoff of hundreds): **0/6 solve** → new-game coverage is NOT a budget/depth
      problem, it's the goal-signal wall. (2) Implemented occam's winning-combo short-circuit: a thin choose_action
      wrapper records the action sequence since the last RESET; on level-up it's cached and replayed first on the
      next level (new `combo` phase). It ACTIVATES (replays 5–15-action combos on ar25/lp85/sp80/vc33) but **pushes
      NO game past level 1** — level-1 solutions don't carry to the harder level 2. No regression (0.1021 holds);
      21/21 tests. Kept for hidden-set games with repeated level structure.

- 2026-06-19 — **THE BIG LIFT: occam ensemble port → HYBRID agent (3/25 → ~9/25).** Ran a parallel research
      workflow over occam's source, then ported its full ensemble. Two decisive findings:
      1. **occam's 17/25 is a LOCAL-ONLY artifact.** It relies on **deepcopy-BFS**, which clones the env to search
         "for free" (clones don't consume env actions). The real Kaggle competition runs against a GATEWAY where
         the env can't be deepcopied (every probe is a counted action). With `skip_deepcopy=True` (the
         Kaggle-portable config), occam solves **7/25** — all movement games (WA30, RE86, KA59, SP80, DC22, M0R0,
         CN04). So occam's portable ceiling is ~7/25, and ~9/25 is likely near the achievable Kaggle max for any
         reset-replay agent.
      2. **My step-wise BFS/combo solves 2 non-movement games occam-portable MISSES** (VC33, LP85).
      Decision: **HYBRID.** `MyAgent.main()` runs occam's actual orchestrator (MIT, $0, no-LLM) first; if it solves
      nothing, falls back to the step-wise agent. Coverage = occam's 7 ∪ {VC33, LP85} = **~9/25 (3× the prior
      3/25)**, always offline-safe (occam error → step-wise fallback).
- 2026-06-19 — **Kaggle packaging.** occam's `solver/` package (MIT) is flattened into a single self-contained
      `agent/occam_bundle.py` by `scripts/bundle_occam.py` (so the splice ships it). `_run_occam` loads it via
      importlib from next to my_agent.py. `build_notebook.py` writes `occam_bundle.py` alongside `my_agent.py` and
      copies both into the framework on Kaggle. occam's deepcopy code is in the bundle but NEVER executed
      (`skip_deepcopy=True`) → no PicklingError on the gateway. Attribution: `NOTICE` + bundle header.

## STATUS: HYBRID agent — ~9/25 coverage (occam 7 movement + step-wise VC33/LP85), Kaggle-portable.
Reset-replay agents appear capped near here on the real (gateway, no-free-search) competition; occam's headline
17/25 does not transfer. Remaining frontier is genuine research (goal inference under the action-cost constraint).

## (earlier) bounded step-wise heuristics plateaued at 3/25 / 0.1021 RHAE.
**Evidence-backed ceiling:** five consecutive levers (efficiency levers, reactive nav, pathfinding, collect-all,
combo caching) each added **0 public coverage/depth**; the budget diagnostic shows it's not a budget problem. The
public games' goals are genuinely beyond reach/collect/sequence-reuse, and the only in-game signal is the grid +
sparse `levels_completed` (no per-frame reward). Moving the number further requires either occam's FULL solver
ensemble (combo search, deepcopy BFS, dense click scan, reactive-click — a large multi-day port) or novel goal-
inference research. The agent is a complete, eligible, deterministic, MIT-licensed baseline; recommend consolidating
and prepping the Kaggle submission rather than more single-lever additions.
The remaining coverage gap is genuine research, not tweaks. Empirically narrowed: the unsolved movement games need
**non-reach-target goals** (collect-all-of-color, ordered sequence, push-block, multi-tile arrangement). The only
in-game signal is the grid + sparse `levels_completed` (FrameData has NO per-frame score), so goal inference must
come from grid structure + diffing the frame at a level-complete event. NEXT, in rough value order:
- [ ] **Goal-pattern from completion**: when BFS/hypo accidentally wins, diff the pre-win → win frame to learn the
      goal pattern (what color/shape/arrangement = victory), then drive toward it on later levels & similar games.
- [ ] **Collect-all detection**: if the cursor removes target tiles on contact, sweep ALL target tiles (TSP-ish),
      not just the nearest — likely unlocks several movement games.
- [ ] occam's **combo search** + **deepcopy BFS** for the non-movement puzzle games (vc33/lp85 score low via BFS).
- [ ] Port occam's remaining solvers: **combo search** (exhaustive short action sequences), **deepcopy BFS**
      (perfect state cloning, no replay cost), **dense click scan** (find ALL effective click positions),
      `_solve_reactive_click`, and **cross-level winning-combo caching** (reuse level-1 solutions on later levels).
- [ ] Run at the real per-level 5× cutoff (human baselines 7–152) — efficiency matters for SCORE where coverage
      is already reached.
- [ ] **First-exposure cost vs 5× cutoff:** measure naive solve cost vs the give-up budget on 2–3 public games.
- [ ] **Offline audit:** grep `agent/my_agent.py` for `requests|urllib|httpx|huggingface_hub|torch.hub|socket`;
      run the notebook with internet disabled locally before submitting.

- 2026-06-19 — **OPEN-SOURCED + KAGGLE KERNEL PUSHED (milestone-eligibility locked).** (1) Verified git history
      clean (11 commits): `.env`/`.kaggle` never tracked; ARC key + Kaggle token never in any commit; no real
      GitHub/AWS/Slack/private-key token patterns anywhere — only placeholders (`ghp_xxx`, `ARC_API_KEY=test-key-123`).
      Pushed the MIT repo (occam attributed in NOTICE) to **public GitHub: github.com/thylinao1/arc-agi-3-offline-agent**
      (25 tracked files; vendor/.env/.kaggle/environment_files all gitignored → absent on remote). Done BEFORE any
      private scores, per the eligibility rule. (2) **Offline audit DONE** (last open item): zero network imports/URL
      literals in `my_agent.py` + `occam_bundle.py`. (3) Built the notebook (6 cells, splice verified — MyAgent +
      occam_bundle + skip_deepcopy embedded) and ran `make submit` → Kaggle kernel
      `maksimsilchenko/arc-prize-2026-arc-agi-3-starter` v1 pushed; commit run reached **COMPLETE** (offline wheel
      install clean; emitted valid `submission.parquet`). Remaining step is the USER clicking **Submit** on the
      notebook to trigger Phase B (the hidden-game gateway rerun that produces the leaderboard score).

- 2026-06-19 — **🔴 FIRST LEADERBOARD SCORE = 0.00 — root-caused; the coverage-first thesis was WRONG.** Investigated
      via browser (kaggle leaderboard/logs) + local RHAE measurement. Findings:
      1. **The metric is EFFICIENCY-dominated.** RHAE=`min(1.15,(human/ai)²)`; reset-replay (occam + our BFS) uses
         10–100× human actions per level → ~0 RHAE even when it solves. "Coverage dominates, satisfice efficiency"
         (the whole project premise) was backwards. Leaderboard: top 1.21; band at 0.43–0.70; **the SAME official
         StochasticGoose notebook structure scores 0.43**; single-entry teams reach 0.01. We're 0.00, ~rank 1227.
      2. **occam contributes ZERO and is HARMFUL.** Local scorecard: wa30 alone → occam `levels=9` but scorecard
         **0.0**; every occam movement game = 0. occam runs first and short-circuits, blocking the step-wise agent.
         **Measured over all 25 public games: hybrid (occam-first)=0.0136 vs step-wise-only=0.1021 → occam made it
         7.5× WORSE.** The "9/25 coverage" was occam's internal level counter, never the scorecard RHAE.
      3. **Submission graded "Succeeded" (no crash)** → 0.00 is the agent's true RHAE, not a rerun bug.
      VERIFIED FIX (local): remove occam / run it last → 0.1021 (keeps symbolic agent, off zero). COMPETITIVE path:
      the frontier is REACTIVE per-frame agents (≈human action counts); StochasticGoose (DriesSmit/ARC3-solution,
      MIT, offline, online-trained CNN) = 0.43. Reset-replay is structurally incompatible with the RHAE metric.

- 2026-06-20 — **PIVOT IMPLEMENTED: replaced the agent with the StochasticGoose reactive CNN.** User chose "go
      competitive." Adopted the official Kaggle sample agent (`inversion/arc3-sample-submission-stochastic-goose`,
      **Apache-2.0** — verified on the notebook page; the GitHub mirror DriesSmit/ARC3-solution is unlicensed, so we
      attribute the Apache-2.0 Kaggle notebook). Changes: `agent/my_agent.py` ← reactive CNN (one-hot frame →
      ActionModel CNN predicting frame-changing action/coord, online-trained per game, resets per level, forward
      play). Removed occam entirely (`agent/occam_bundle.py`, `scripts/bundle_occam.py` deleted). `build_notebook.py`
      → ACCELERATOR="t4" (GPU), drops the occam splice; kernel-metadata → enable_gpu + machine_shape=NvidiaTeslaT4.
      NOTICE/README/HANDOFF updated; `make setup` installs CPU torch; tests rewritten (6 passing: model shape,
      frame→tensor, action masking, experience hash). Local smoke test: agent runs clean via the framework loop
      (CPU ~1 fps → GPU needed, hence T4). Notebook rebuilt (5 cells, no occam, torch+ActionModel embedded).
      VALIDATION GAP: local CPU is too slow to confirm scoring; confidence rests on the proven sample (0.43) +
      correct integration. Phase B on the T4 is the real test. NEXT: resubmit, confirm nonzero leaderboard score.

- 2026-06-20 — **REACTIVE CNN SCORED 0.08 (off zero ✓).** v2 Phase B completed: publicScore **0.08** (was 0.00).
      The reactive paradigm works on the hidden set. Context: leaderboard 1st=1.24, the StochasticGoose frontier
      band=0.43–0.70; **0.08 is roughly the BASE sample's level** — the 0.43 entries are heavily-iterated versions.
      Feedback-loop reality: Kaggle rerun ≈4–8h (the agent self-caps at 8h; MAX_ACTIONS=inf) + 5 subs/day; local
      iteration is slow too (MPS available but ~CPU-speed at batch-1, and the CNN needs thousands of steps/game to
      learn → local public-set runs under-train and under-represent it). NEXT: research what lifted others to 0.43+
      (Kaggle Discussion/Code) rather than blind tinkering; prepare an improved v3 to submit tomorrow.

- 2026-06-20 — **HONEST-STUDENT PATH chosen + teacher VALIDATED.** Research showed the public 0.42–0.46 "BFS
      solvers" are a leakage exploit (load the game .py via importlib, deepcopy-simulate the real game offline, read
      internal _score/hidden fields) — likely prize/milestone-INELIGIBLE; honest public ceiling ≈0.25–0.32. User
      chose: use such a solver ONLY as an OFFLINE teacher on PUBLIC games to generate genuine (frame,action) demos,
      then train an HONEST reactive student (no source-reading) and submit the student. VALIDATED the genuine teacher
      (Agent v15 BFSSolver, Apache-2.0) on public game sources: sp80→4-action win, vc33→3-action win (occam's were
      FAKE). NEXT (multi-turn build): (1) rewrite scripts/gen_demos.py to use BFSSolver (not occam) over all 25
      public games×levels → demo dataset; (2) train student CNN (frame→action) on demos; (3) honest inference agent
      + ship weights (Kaggle dataset) in the offline notebook; (4) submit. NOTE: current scripts/gen_demos.py is the
      DEAD occam version (0 demos) — needs the BFSSolver rewrite; teacher proof in /tmp/teacher_src.py.

- 2026-06-20 — **GOAL = TOP-5 (>0.66); plan = BOTH tracks in parallel.** User clarified only top 5 qualify (5th ≈
      0.66). Honest learned ceiling ≈0.32 and exploit BFS ≈0.42–0.46 are both BELOW 0.66 → top-5 is the elite
      frontier (Tufa 1.21, Tong Hui Kang 0.70), a multi-week uncertain effort. Host code-reqs: runtime now 9h,
      RTX6000, NO explicit ban on reading game source (the 0.42–0.46 exploit). DECISION (both in parallel):
      (1) Track-1 LB: ship the public exploit solver (Agent v15, CPU, 0.46) for a leaderboard datapoint when a
      submission slot frees (2/day). (2) Track-2 prize: strengthen the OFFLINE teacher (legit on public games) →
      deep/rich demos → train an HONEST student that ships with no source access. Shared blocker = teacher depth
      (BFSSolver@45s only cracked L0 of most games). Launched a deeper gen run (bfs-timeout 150s, max-levels 12) to
      deepen the dataset + test whether depth is timeout-bound. Real lever for Track-2 depth: informed search (the
      solver may read game hidden state OFFLINE) instead of plain BFS. occam-based gen_demos is dead; current
      gen_demos.py uses the genuine BFSSolver teacher (reference/teacher_agentv15.py, gitignored).

- 2026-06-20 — **HONEST STUDENT TRAINED + PACKAGED + PUSHED (v3).** Completed the imitation pipeline: trained the
      student (behavior cloning on 93 demos × 8 color-aug = 837 samples; resumable checkpointing; final train-acc
      0.987 on CPU). agent/my_agent.py warm-starts from the weights (_load_pretrained) then learns online — plays
      HONESTLY (no source access). Packaged weights (137MB) as Kaggle dataset maksimsilchenko/arc-agi3-student-weights
      (ready), wired via kernel-metadata dataset_sources; my_agent loads /kaggle/input/arc-agi3-student-weights/
      student.pt. Pushed kernel v3 (GPU T4 + weights dataset). Commit run polling. REMAINING: user submits a slot →
      Phase B → the real number (does imitation warm-start beat our 0.08 / the 0.25 base?). Expectation: genuine
      prize-eligible entry, NOT >0.66 (data ceiling confirmed). Dead end recorded: occam (fake solves) + MultiSolver
      (no better coverage). gen_demos uses the genuine BFSSolver teacher (reference/, gitignored).
