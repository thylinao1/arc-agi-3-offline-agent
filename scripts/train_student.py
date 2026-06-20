"""Train the HONEST imitation student on teacher demonstrations.

Behavior cloning: frame -> action. The student is the StochasticGoose ActionModel
(5 action logits + 64*64 click-coord logits), trained by cross-entropy on genuine
(frame, action) pairs the offline teacher produced on PUBLIC games. The SHIPPED
student plays with no game-source access — only this learned policy.

Target encoding (verified by the demo-pipeline audit to round-trip through the
student decoder MyAgent._sample_from_combined_output):
    act 1..5  -> index 0..4
    ACTION6   -> 5 + y*64 + x      (decoder: y=idx//64, x=idx%64)

Augmentation: color permutations (colors are arbitrary labels in ARC; background
0 kept fixed) to stretch the thin dataset. Action labels are unchanged (moves are
color-invariant; clicks are by coordinate, also color-invariant).

Usage:
    .venv/bin/python scripts/train_student.py --demos data/demos/demos.npz \
        --out models/student.pt --epochs 40 --aug 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "vendor" / "ARC-AGI-3-Agents"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

GRID = 64
NUM_COLOURS = 16
NUM_ACTIONS = 5
COMBINED = NUM_ACTIONS + GRID * GRID  # 4101


def _device() -> torch.device:
    import os
    forced = os.environ.get("STUDENT_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_target(act_id: int, cx: int, cy: int) -> int | None:
    if 1 <= act_id <= 5:
        return act_id - 1
    if act_id == 6 and 0 <= cx < GRID and 0 <= cy < GRID:
        return 5 + cy * GRID + cx
    return None  # ACTION7 / malformed -> unrepresentable, drop


def _roundtrip_check() -> None:
    """The audit's must-have: encode(click)->decode must recover (x,y)."""
    for x, y in [(0, 0), (60, 32), (4, 30), (63, 63)]:
        t = encode_target(6, x, y)
        idx = t - 5
        assert idx // GRID == y and idx % GRID == x, f"encode/decode mismatch at {(x,y)}"


def build_dataset(npz_path: Path, aug: int, seed: int = 0):
    d = np.load(npz_path, allow_pickle=True)
    frames, act, cx, cy = d["frames"], d["act_id"], d["cx"], d["cy"]
    rng = np.random.RandomState(seed)
    X, Y = [], []
    dropped = 0
    for i in range(len(act)):
        tgt = encode_target(int(act[i]), int(cx[i]), int(cy[i]))
        if tgt is None:
            dropped += 1
            continue
        base = frames[i].astype(np.int64)
        variants = [base]
        for _ in range(aug):
            perm = np.arange(NUM_COLOURS)
            rest = perm[1:].copy(); rng.shuffle(rest); perm[1:] = rest  # keep bg(0) fixed
            variants.append(perm[base])
        for v in variants:
            X.append(v); Y.append(tgt)
    X = np.stack(X).astype(np.int64)
    Y = np.array(Y, dtype=np.int64)
    print(f"dataset: {len(Y)} samples ({aug}x color-aug from {len(act)-dropped} demos, "
          f"dropped {dropped}); clicks={(Y>=5).sum()}, moves={(Y<5).sum()}")
    return X, Y


def to_onehot(grids: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(grids)  # (B,64,64) long
    oh = F.one_hot(t, NUM_COLOURS).permute(0, 3, 1, 2).float()  # (B,16,64,64)
    return oh.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default=str(ROOT / "data" / "demos" / "demos.npz"))
    ap.add_argument("--out", default=str(ROOT / "models" / "student.pt"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--aug", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    _roundtrip_check()
    print("encode/decode round-trip: OK")
    from agent.my_agent import ActionModel

    dev = _device(); print("device:", dev)
    X, Y = build_dataset(Path(args.demos), args.aug)
    Yt = torch.from_numpy(Y).to(dev)

    model = ActionModel(input_channels=NUM_COLOURS, grid_size=GRID).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = len(Y)
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(n)
        tot = 0.0
        for s in range(0, n, args.batch):
            idx = perm[s:s + args.batch]
            xb = to_onehot(X[idx], dev)
            yb = Yt[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                accs = []
                for s in range(0, n, 256):
                    xb = to_onehot(X[s:s + 256], dev)
                    pred = model(xb).argmax(1).cpu().numpy()
                    accs.append(pred == Y[s:s + 256])
                acc = np.concatenate(accs).mean()
            print(f"epoch {epoch:3d}  loss {tot/n:.4f}  train-acc {acc:.3f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "arch": {"input_channels": NUM_COLOURS, "grid_size": GRID}}, out)
    print(f"saved student weights -> {out} ({out.stat().st_size/1e3:.1f} kB)")


if __name__ == "__main__":
    main()
