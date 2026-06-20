"""Unit tests for agent/my_agent.py — the StochasticGoose reactive CNN agent.

These exercise the framework-independent pieces: the ActionModel CNN, the
frame→tensor encoding, the action-masking sampler, and the experience hash.
They skip (rather than fail) when torch or the vendored framework is absent
(i.e. before `make setup`), since the agent imports both at module load.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Importing the agent pulls in the ARC-AGI-3-Agents framework + arcengine.
# Skip the whole module if they're not installed/vendored yet.
try:
    from agent.my_agent import ActionModel, MyAgent
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"agent.my_agent unavailable ({exc!r})", allow_module_level=True)


GRID = 64
NUM_COLOURS = 16
NUM_ACTION_TYPES = 5
COMBINED = NUM_ACTION_TYPES + GRID * GRID  # 5 + 4096 = 4101


def _stub_self() -> types.SimpleNamespace:
    """Minimal stand-in carrying only the attributes the pure methods read."""
    return types.SimpleNamespace(
        grid_size=GRID,
        num_colours=NUM_COLOURS,
        num_coordinates=GRID * GRID,
        device=torch.device("cpu"),
    )


# ───────────────────────────── ActionModel ─────────────────────────────
def test_action_model_output_shape() -> None:
    model = ActionModel(input_channels=NUM_COLOURS, grid_size=GRID)
    out = model(torch.zeros(2, NUM_COLOURS, GRID, GRID))
    assert out.shape == (2, COMBINED)  # 5 action logits + 64*64 coord logits


def test_action_model_is_deterministic_in_eval() -> None:
    model = ActionModel(input_channels=NUM_COLOURS, grid_size=GRID).eval()
    x = torch.randn(1, NUM_COLOURS, GRID, GRID)
    with torch.no_grad():
        a, b = model(x), model(x)
    assert torch.allclose(a, b)


# ─────────────────────────── frame → tensor ────────────────────────────
def test_frame_to_tensor_onehot_shape_and_values() -> None:
    frame = np.zeros((1, GRID, GRID), dtype=np.int64)
    frame[0, 0, 0] = 5  # one cell of colour 5
    fd = types.SimpleNamespace(frame=frame)
    tensor = MyAgent._frame_to_tensor(_stub_self(), fd)
    assert tuple(tensor.shape) == (NUM_COLOURS, GRID, GRID)
    # one-hot: colour-5 channel hot at (0,0); colour-0 channel hot elsewhere
    assert tensor[5, 0, 0].item() == 1.0
    assert tensor[0, 0, 0].item() == 0.0
    assert tensor[0, 1, 1].item() == 1.0


# ────────────────────────── action masking ─────────────────────────────
def test_sampler_masks_to_only_available_simple_action() -> None:
    np.random.seed(0)
    logits = torch.zeros(COMBINED)
    # Only ACTION1 available, ACTION6 (coords) not → must pick action index 0.
    idx, coords, coord_idx, _ = MyAgent._sample_from_combined_output(
        _stub_self(), logits, available_actions=[1]
    )
    assert idx == 0
    assert coords is None


def test_sampler_picks_coordinate_when_only_action6_available() -> None:
    np.random.seed(0)
    logits = torch.zeros(COMBINED)
    idx, coords, coord_idx, _ = MyAgent._sample_from_combined_output(
        _stub_self(), logits, available_actions=[6]
    )
    assert idx == 5  # coordinate/ACTION6 branch
    assert coords is not None
    y, x = coords
    assert 0 <= x < GRID and 0 <= y < GRID


# ───────────────────────── experience hashing ──────────────────────────
def test_experience_hash_is_stable_and_action_sensitive() -> None:
    frame = np.zeros((NUM_COLOURS, GRID, GRID), dtype=bool)
    h_a = MyAgent._compute_experience_hash(_stub_self(), frame, 3)
    h_b = MyAgent._compute_experience_hash(_stub_self(), frame, 3)
    h_c = MyAgent._compute_experience_hash(_stub_self(), frame, 4)
    assert h_a == h_b          # deterministic for identical (frame, action)
    assert h_a != h_c          # differs when the action differs
