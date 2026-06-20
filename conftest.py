"""Make the project root + vendored framework importable for tests.

`agent/my_agent.py` (the StochasticGoose reactive CNN) imports torch and the
ARC-AGI-3-Agents framework (`agents.agent`, `arcengine`). Tests therefore need
both the repo root and the vendored framework on sys.path; they `importorskip`
torch/framework so a bare checkout (no `make setup`) skips rather than errors.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if VENDOR.exists() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
