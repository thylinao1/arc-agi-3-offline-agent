# =====================================================================
# ARC-AGI-3 reactive agent — adapted from the official Kaggle sample
# "ARC3 Sample Submission - Stochastic Goose".
#
# Upstream source (Apache License 2.0):
#   https://www.kaggle.com/code/inversion/arc3-sample-submission-stochastic-goose
#   https://github.com/DriesSmit/ARC3-solution
#   Authors: Dries Smit (Lead), Jack Cole (Adviser) — Tufa Labs.
#
# This file is licensed under the Apache License, Version 2.0 (the upstream
# license), NOT the repository's MIT license. See NOTICE for attribution and
# the full Apache-2.0 reference.
#
# Modifications in this repo:
#   - Adopted as the agent spliced into the Kaggle submission notebook by
#     scripts/build_notebook.py (replaces the prior reset-replay symbolic
#     agent, which scored ~0 RHAE because the metric rewards action efficiency,
#     not coverage — see DECISION-LOG.md 2026-06-19/06-20).
#   - Self-contained for the %%writefile splice (no sibling imports).
# =====================================================================

# =====================================================================
# StochasticGoose – CNN-based action learning agent
# Source: https://github.com/DriesSmit/ARC3-solution
# Authors: Dries Smit (Lead), Jack Cole (Adviser) — Tufa Labs
#
# Compatibility notes:
#   - Original uses `latest_frame.score` — replaced with `levels_completed`
#   - Original uses `latest_frame.available_actions` — gateway sends raw ints [1,2,...,6]
# =====================================================================
import hashlib
import logging
import os
import random
import time
import traceback
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState


# --- Inlined from utils.py ---

def setup_experiment_directory(base_output_dir='runs'):
    """Create directories for outputs and logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(base_dir, exist_ok=True)
    log_file = os.path.join(base_dir, 'logs.log')
    print(f"Experiment directory created: {base_dir}")
    return base_dir, log_file


def get_environment_directory(base_dir, game_id):
    """Get or create environment-specific directory for a game_id."""
    env_dir = os.path.join(base_dir, game_id)
    os.makedirs(env_dir, exist_ok=True)
    return env_dir


def setup_logging_for_experiment(log_file_path):
    """Update logging configuration to use the experiment directory's log file."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setLevel(root_logger.level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


# --- ActionModel CNN ---

class ActionModel(nn.Module):
    """CNN that predicts which actions will result in new frames with shared conv backbone."""

    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 5  # ACTION1-ACTION5

        # Shared convolutional backbone
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Action head
        self.action_pool = nn.MaxPool2d(4, 4)
        action_flattened_size = 256 * 16 * 16
        self.action_fc = nn.Linear(action_flattened_size, 512)
        self.action_head = nn.Linear(512, self.num_action_types)

        # Coordinate head (64x64 action space)
        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        conv_features = F.relu(self.conv4(x))

        # Action head
        action_features = self.action_pool(conv_features)
        action_features = action_features.view(action_features.size(0), -1)
        action_features = F.relu(self.action_fc(action_features))
        action_features = self.dropout(action_features)
        action_logits = self.action_head(action_features)

        # Coordinate head
        coord_features = F.relu(self.coord_conv1(conv_features))
        coord_features = F.relu(self.coord_conv2(coord_features))
        coord_features = F.relu(self.coord_conv3(coord_features))
        coord_logits = self.coord_conv4(coord_features)
        coord_logits = coord_logits.view(coord_logits.size(0), -1)

        combined_logits = torch.cat([action_logits, coord_logits], dim=1)
        return combined_logits


# --- Action Agent (StochasticGoose) ---

class MyAgent(Agent):
    """CNN-based action learning agent that predicts which actions cause frame changes."""

    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10  # PERF: Keep only the last N frames (sliding window)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed % (2**32 - 1))
        self.start_time = time.time()

        # Device configuration
        # cuda on Kaggle (T4); mps lets local Mac dev use the Metal GPU; else cpu.
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Action agent using device: {self.device}")

        # Setup experiment directory and logging
        self.base_dir, log_file = setup_experiment_directory()
        setup_logging_for_experiment(log_file)

        env_dir = get_environment_directory(self.base_dir, self.game_id)
        self.current_score = -1

        self.logger = logging.getLogger(f"ActionAgent_{self.game_id}")

        # Visualization disabled for submission
        self.save_action_visualizations = False

        # Initialize action model
        self.grid_size = 64
        self.num_coordinates = self.grid_size * self.grid_size
        self.num_colours = 16
        self.action_model = None
        self.optimizer = None

        # Experience buffer
        self.experience_buffer = deque(maxlen=200000)
        self.experience_hashes = set()
        self.batch_size = 64
        self.train_frequency = 5

        # Track previous state/action
        self.prev_frame = None
        self.prev_action_idx = None

        # Action mapping
        self.action_list = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                           GameAction.ACTION4, GameAction.ACTION5]

        self.log_dir = env_dir
        self.logger.info(f"Action agent initialized for game_id: {self.game_id}")

    def append_frame(self, frame: FrameData) -> None:
        """Override to cap frames list at _MAX_FRAMES (sliding window)."""
        self.frames.append(frame)
        if len(self.frames) > self._MAX_FRAMES:
            self.frames = self.frames[-self._MAX_FRAMES:]
        if frame.guid:
            self.guid = frame.guid
        if hasattr(self, "recorder") and not self.is_playback:
            import json
            self.recorder.record(json.loads(frame.model_dump_json()))

    def _get_score(self, frame):
        """Get score from frame, compatible with both patched and standard FrameData."""
        return getattr(frame, 'score', None) or frame.levels_completed

    def _sample_from_combined_output(self, combined_logits, available_actions=None):
        """Sample from combined 5 + 64x64 action space with masking for invalid actions."""
        action_logits = combined_logits[:5]
        coord_logits = combined_logits[5:]

        if available_actions is not None and len(available_actions) > 0:
            action_mask = torch.full_like(action_logits, float('-inf'))
            action6_available = False
            for action in available_actions:
                # Gateway sends raw ints [1,2,...,6], not GameAction enums
                action_id = action.value if hasattr(action, 'value') else int(action)
                if 1 <= action_id <= 5:
                    action_mask[action_id - 1] = 0.0
                elif action_id == 6:
                    action6_available = True
            action_logits = action_logits + action_mask
            if not action6_available:
                coord_logits = coord_logits + torch.full_like(coord_logits, float('-inf'))

        action_probs = torch.sigmoid(action_logits)
        coord_probs_raw = torch.sigmoid(coord_logits)
        coord_probs_scaled = coord_probs_raw / self.num_coordinates

        all_probs_sampling = torch.cat([action_probs, coord_probs_scaled])
        all_probs_sampling = all_probs_sampling / all_probs_sampling.sum()
        all_probs_sampling_np = all_probs_sampling.cpu().numpy()

        selected_idx = np.random.choice(len(all_probs_sampling_np), p=all_probs_sampling_np)

        coord_probs_viz = torch.sigmoid(coord_logits)
        all_probs_viz = torch.cat([action_probs, coord_probs_viz])
        all_probs_viz_np = all_probs_viz.cpu().numpy()

        if selected_idx < 5:
            return selected_idx, None, None, all_probs_viz_np
        else:
            coord_idx = selected_idx - 5
            y_idx = coord_idx // self.grid_size
            x_idx = coord_idx % self.grid_size
            return 5, (y_idx, x_idx), coord_idx, all_probs_viz_np

    def _frame_to_tensor(self, frame_data):
        """Convert frame data to tensor format for the model."""
        frame = np.array(frame_data.frame, dtype=np.int64)
        frame = frame[-1]
        assert frame.shape == (self.grid_size, self.grid_size), \
            f"Expected frame shape ({self.grid_size}, {self.grid_size}), got {frame.shape}"
        tensor = torch.zeros(self.num_colours, self.grid_size, self.grid_size, dtype=torch.float32)
        tensor.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        return tensor.to(self.device)

    def _compute_experience_hash(self, frame, action_idx):
        """Compute hash for frame+action combination to ensure uniqueness."""
        frame_bytes = frame.tobytes()
        hash_input = frame_bytes + str(action_idx).encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()

    def _train_action_model(self):
        """Train the action model on collected experiences."""
        if len(self.experience_buffer) < self.batch_size:
            return

        batch_indices = np.random.choice(len(self.experience_buffer), self.batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]

        states = torch.stack([torch.from_numpy(exp['state']).float().to(self.device) for exp in batch])
        action_indices = torch.tensor([exp['action_idx'] for exp in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()

        combined_logits = self.action_model(states)
        selected_logits = combined_logits.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        main_loss = F.binary_cross_entropy_with_logits(selected_logits, rewards)

        all_probs = torch.sigmoid(combined_logits)
        action_entropy = all_probs[:, :5].mean()
        coord_entropy = all_probs[:, 5:].mean()

        total_loss = main_loss - 0.0001 * action_entropy - 0.00001 * coord_entropy
        total_loss.backward()
        self.optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _has_time_elapsed(self):
        """Check if 8 hours have elapsed since start."""
        return (time.time() - self.start_time) >= 8 * 3600 - 5 * 60

    def is_done(self, frames, latest_frame):
        """Decide if the agent is done playing."""
        try:
            return any([
                latest_frame.state is GameState.WIN,
                self._has_time_elapsed(),
            ])
        except Exception as e:
            print(f"[DEBUG] is_done crashed: {e}")
            traceback.print_exc()
            return True  # bail out on error

    def choose_action(self, frames, latest_frame):
        """Choose action using action model predictions."""
        try:
            # DEBUG: Log frame info on first call
            if self.action_counter == 0:
                print(f"[DEBUG] latest_frame type: {type(latest_frame)}")
                print(f"[DEBUG] latest_frame.state: {latest_frame.state}")
                print(f"[DEBUG] latest_frame.levels_completed: {latest_frame.levels_completed}")
                print(f"[DEBUG] has score: {hasattr(latest_frame, 'score')}")
                print(f"[DEBUG] available_actions: {getattr(latest_frame, 'available_actions', 'N/A')}")
                if hasattr(latest_frame, 'frame') and latest_frame.frame:
                    frame_arr = np.array(latest_frame.frame)
                    print(f"[DEBUG] frame shape: {frame_arr.shape}")

            # Check if score/level has changed (triggers model reset for new level)
            current_level = self._get_score(latest_frame)
            if current_level != self.current_score:
                self.logger.info(f"Score changed from {self.current_score} to {current_level} at action {self.action_counter}")
                print(f"Score changed from {self.current_score} to {current_level} at action {self.action_counter}")

                # Clear the experience buffer (old-level layouts/clicks don't apply to the
                # new level) but KEEP the model + optimizer warm. Within a game the control
                # scheme (which actions move/click, what a frame-change looks like) is usually
                # consistent across levels; the base sample re-initialised the CNN every level,
                # which threw that away and forced re-learning from scratch — wasted actions on
                # exactly the later levels RHAE weights most. Persisting the weights warm-starts
                # each new level. (v3 hypothesis — see DECISION-LOG 2026-06-20.)
                self.experience_buffer.clear()
                self.experience_hashes.clear()
                # Create the model lazily on the FIRST level; keep it warm thereafter.
                if self.action_model is None:
                    self.action_model = ActionModel(input_channels=self.num_colours, grid_size=self.grid_size).to(self.device)
                    self.optimizer = optim.Adam(self.action_model.parameters(), lr=0.0001)
                    print("Initialised action model")
                else:
                    print("Cleared experience buffer - new level reached (model kept warm)")

                self.prev_frame = None
                self.prev_action_idx = None
                self.current_score = current_level

            if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                self.prev_frame = None
                self.prev_action_idx = None
                action = GameAction.RESET
                action.reasoning = "Game needs reset."
                return action

            # Convert current frame to tensor
            current_frame = self._frame_to_tensor(latest_frame)

            if current_frame is None:
                self.prev_frame = None
                self.prev_action_idx = None
                action = random.choice(self.action_list[:5])
                action.reasoning = f"Skipped weird frame, random {action.value}"
                return action

            # Create experience from previous action
            if self.prev_frame is not None:
                experience_hash = self._compute_experience_hash(self.prev_frame, self.prev_action_idx)
                if experience_hash not in self.experience_hashes:
                    current_frame_np = current_frame.cpu().numpy().astype(bool)
                    frame_changed = not np.array_equal(self.prev_frame, current_frame_np)
                    experience = {
                        'state': self.prev_frame,
                        'action_idx': self.prev_action_idx,
                        'reward': 1.0 if frame_changed else 0.0
                    }
                    self.experience_buffer.append(experience)
                    self.experience_hashes.add(experience_hash)

            # Get action predictions
            available_actions = getattr(latest_frame, 'available_actions', None)
            with torch.no_grad():
                combined_logits = self.action_model(current_frame.unsqueeze(0))
                combined_logits = combined_logits.squeeze(0)
                action_idx, coords, coord_idx, all_probs = self._sample_from_combined_output(
                    combined_logits, available_actions
                )

                if action_idx < 5:
                    selected_action = self.action_list[action_idx]
                    selected_action.reasoning = f"{selected_action.name} (prob: {all_probs[action_idx]:.3f})"
                else:
                    selected_action = GameAction.ACTION6
                    y, x = coords
                    selected_action.set_data({"x": int(x), "y": int(y)})
                    selected_action.reasoning = f"ACTION6 at ({x}, {y}) (prob: {all_probs[coord_idx]:.3f})"

            # Store current frame and action for next experience creation
            self.prev_frame = current_frame.cpu().numpy().astype(bool)
            if action_idx < 5:
                self.prev_action_idx = action_idx
            else:
                self.prev_action_idx = 5 + coord_idx

            # Train model periodically
            if self.action_counter % self.train_frequency == 0:
                self._train_action_model()

            return selected_action

        except Exception as e:
            print(f"[DEBUG] choose_action CRASHED at action {self.action_counter}: {type(e).__name__}: {e}")
            traceback.print_exc()
            # Fallback: return a random action so the agent doesn't die
            action = random.choice(self.action_list[:5])
            action.reasoning = f"Fallback after error: {e}"
            return action
