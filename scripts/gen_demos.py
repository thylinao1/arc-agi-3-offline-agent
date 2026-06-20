"""Generate imitation-learning demonstrations: genuine (frame, action) pairs from
winning trajectories on the PUBLIC training games.

Teacher: an offline BFS solver that instantiates each public game's class and
deepcopy-simulates it to find a real winning action sequence per level (verified
genuine — it advances the game's own levels_completed / _current_level_index).
This solver is used ONLY offline on the public dev games to produce training
data; the *submitted* student plays honestly with no game-source access. The
teacher lives in the gitignored reference/ dir (adapted from the Apache-2.0
public "Agent v15" notebook) and is NOT shipped.

Output dataset (data/demos/demos.npz):
    frames  (N,64,64) uint8   — the grid the agent sees BEFORE acting
    act_id  (N,) int8         — raw action id 1..7
    cx, cy  (N,) int16        — click coords for ACTION6, else -1
    game    (N,) str          — source public game id
    level   (N,) int16        — level index within the game

Usage:
    .venv/bin/python scripts/gen_demos.py --games sp80,vc33 --bfs-timeout 30
    .venv/bin/python scripts/gen_demos.py            # all 25, default out dir
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "vendor" / "ARC-AGI-3-Agents", ROOT / "vendor" / "occam"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from arcengine import ActionInput, GameAction  # noqa: E402

GRID = 64
TEACHER_PATH = ROOT / "reference" / "teacher_agentv15.py"


def _load_teacher():
    if not TEACHER_PATH.exists():
        raise SystemExit(f"Teacher solver not found at {TEACHER_PATH} (gitignored). "
                         "Extract BFSSolver from the public Agent v15 notebook there.")
    spec = importlib.util.spec_from_file_location("teacher", TEACHER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def locate(gid: str):
    hits = glob.glob(str(ROOT / "environment_files" / gid / "*" / f"{gid}.py"))
    if not hits:
        return None, None
    content = Path(hits[0]).read_text()
    m = re.search(r"class\s+(\w+)\s*\(\s*ARCBaseGame", content)
    return hits[0], (m.group(1) if m else gid.capitalize())


def replay_extract(game_cls, level_idx: int, sol: list) -> list[dict]:
    """Replay a winning sequence on a fresh game at `level_idx`, recording the
    grid BEFORE each action. Mirrors the teacher's reset protocol exactly."""
    game = game_cls()
    game.set_level(level_idx)
    game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    r = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    out: list[dict] = []
    for act_id, data in sol:
        if not r.frame:
            break
        grid = np.asarray(r.frame[-1], dtype=np.uint8)
        cx = cy = -1
        if data and isinstance(data, dict):
            cx = int(data.get("x", -1)); cy = int(data.get("y", -1))
        out.append({"grid": grid, "act_id": int(act_id), "cx": cx, "cy": cy,
                    "level": level_idx})
        ai = (ActionInput(id=GameAction.from_id(act_id), data=data) if data
              else ActionInput(id=GameAction.from_id(act_id)))
        r = game.perform_action(ai, raw=True)
    return out


def gen_for_game(teacher, gid: str, max_levels: int, bfs_timeout: int) -> list[dict]:
    path, cls_name = locate(gid)
    if not path:
        print(f"  {gid}: no game file"); return []
    solver = teacher.BFSSolver(path, cls_name, scan_timeout=3, bfs_timeout=bfs_timeout)
    if not solver.load():
        print(f"  {gid}: failed to load class {cls_name}"); return []
    game_cls = solver.game_cls
    demos: list[dict] = []
    prev = None
    solved_levels = 0
    for lvl in range(max_levels):
        try:
            sol = solver.solve_level(lvl, prev_solution=prev)
        except Exception as exc:
            print(f"  {gid} L{lvl}: solver error {exc!r}"); break
        if not sol:
            break  # later levels are unreachable without solving this one
        pairs = replay_extract(game_cls, lvl, sol)
        for d in pairs:
            d["game"] = gid
        demos.extend(pairs)
        solved_levels += 1
        prev = sol
    print(f"  {gid}: solved {solved_levels} level(s), {len(demos)} raw pairs", flush=True)
    return demos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=None, help="comma list; default all 25")
    ap.add_argument("--out", default=str(ROOT / "data" / "demos"))
    ap.add_argument("--max-levels", type=int, default=20)
    ap.add_argument("--bfs-timeout", type=int, default=60)
    args = ap.parse_args()

    teacher = _load_teacher()
    import arc_agi
    from arc_agi import OperationMode
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    all_ids = [e.game_id.split("-")[0] for e in arc.get_environments()]
    games = [g.strip() for g in args.games.split(",")] if args.games else all_ids

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    frames, act_ids, cxs, cys, gms, lvls = [], [], [], [], [], []
    seen: set = set()
    for i, gid in enumerate(games, 1):
        print(f"[{i}/{len(games)}] {gid}", flush=True)
        for d in gen_for_game(teacher, gid, args.max_levels, args.bfs_timeout):
            key = (d["grid"].tobytes(), d["act_id"], d["cx"], d["cy"])
            if key in seen:
                continue
            seen.add(key)
            frames.append(d["grid"]); act_ids.append(d["act_id"])
            cxs.append(d["cx"]); cys.append(d["cy"])
            gms.append(d["game"]); lvls.append(d["level"])

    if not frames:
        print("\nNo demos generated."); return
    frames = np.stack(frames).astype(np.uint8)
    act_ids = np.array(act_ids, dtype=np.int8)
    cxs = np.array(cxs, dtype=np.int16); cys = np.array(cys, dtype=np.int16)
    np.savez_compressed(out / "demos.npz", frames=frames, act_id=act_ids,
                        cx=cxs, cy=cys, game=np.array(gms), level=np.array(lvls, dtype=np.int16))
    # distribution
    uniq_games = sorted(set(gms))
    dist = {int(a): int((act_ids == a).sum()) for a in sorted(set(act_ids.tolist()))}
    print(f"\nSAVED {len(act_ids)} unique demos to {out/'demos.npz'}")
    print(f"  games solved: {len(uniq_games)} -> {uniq_games}")
    print(f"  action-id distribution: {dist}  (6 = click)")


if __name__ == "__main__":
    main()
