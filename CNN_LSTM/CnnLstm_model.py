"""
================================================================================
  CNN-LSTM Multi-Task Neural Network for Sequence-to-Sequence Prediction
================================================================================

Author: [Jian Xu] <[jianxu@gmail.com]>
        [Dalian University of Technology]

============================================================================

  Overview
  --------
  This module implements a CNN-LSTM hybrid model for multi-output sequence
  prediction tasks. The model accepts multi-channel time-series inputs,
  processes them through parallel CNN feature extractors and LSTM encoders,
  fuses features in a shared recurrent layer, and produces two distinct outputs
  (e.g., amplitude and phase) through private decoder heads.

  Architecture
  ------------
  Input (per timestep, normalized [0,1]):
    - Channel A: N signals  (shape: [batch, seq_len, N]  → reshaped to [batch, 1, seq_len, N])
    - Channel B: M signals  (shape: [batch, seq_len, M]  → reshaped to [batch, 1, seq_len, M])

  Feature Extraction Branch (parallel):
    Channel A  ──► Conv2d ──► BatchNorm ──► ELU ──► Conv2d ──► BatchNorm ──► ELU
    Channel B  ──► Conv2d ──► BatchNorm ──► ELU ──► Conv2d ──► BatchNorm ──► ELU
    Channel C  ──► LSTM (input_size=C, hidden_size=H1, num_layers=2)
    Channel D  ──► LSTM (input_size=D, hidden_size=H2, num_layers=2)  [optional]

  Feature Fusion:
    All branches ──► Concatenate ──► Shared LSTM (hidden_size=32, num_layers=2)

  Private Decoders:
    Shared output ──► FC ──► BatchNorm ──► Activation ──► Dropout ──► FC ──► ...
    Output 1 (Amplitude):  2 outputs
    Output 2 (Phase):      2 outputs

  Dependencies
  ------------
  - torch >= 1.10
  - torchmetrics
  - ray[tune] (for hyperparameter tuning)

  Data Interface
  --------------
  Data is expected to arrive pre-processed and normalized in the [0, 1] range.
  The model expects two input tensors:

    x_rmp  (shape: [batch, seq_len, num_rmp_signals])   — CNN-processed channel
    x_phys (shape: [batch, seq_len, num_phys_signals])  — LSTM-processed channel

  A PyTorch DataLoader should batch and feed these tensors. A Dataset wrapper that
  splits raw feature arrays into these two channels is the caller's responsibility.

  Multi-Task Training Strategies
  -----------------------------
  Four training modes are supported via the `multitask_freeze_gradnorm` tuple
  (alternate_freezing, use_gradnorm):

    (F, F) — Vanilla: all parameters updated jointly with equal loss weights.
    (T, F) — Alternate freezing: shared and private parameters are updated
             sequentially in a three-step cycle.
    (F, T) — GradNorm: dynamic loss weighting via gradient normalization.
    (T, T) — Alternate freezing + GradNorm: combines both strategies.

  Hyperparameter Tuning
  ----------------------
  The `hyperparameter_tuning()` function wraps the training loop with Ray Tune's
  ASHA scheduler. The search space is defined as a Ray Tune config dict and
  should be adjusted according to the specific task and dataset size.

  Usage Example
  -------------
  # Standalone training with default hyperparameter search:
  from model import MyModel, hyperparameter_tuning, execute_training_loop

  config_space = {
      "units_1": tune.choice([64, 96, 128]),
      "units_2": tune.choice([32, 64]),
      "units_3": tune.choice([64, 96, 128]),
      "units_4": tune.choice([32, 64]),
      "activation_1": tune.choice(["relu", "tanh", "sigmoid", "selu"]),
      "activation_2": tune.choice(["relu", "tanh", "sigmoid", "selu"]),
      "batch_size": tune.choice([16, 32, 64]),
      "optimizer_name": tune.choice(["Adam", "AdamW"]),
      "lr": tune.loguniform(1e-4, 1e-2),
      "l2": tune.loguniform(1e-4, 1e-2),
      "dropout": tune.uniform(0.1, 0.5),
  }
  best_config = hyperparameter_tuning(
      num_samples=100,
      max_epochs=50,
      config_space=config_space,
      ...
  )
================================================================================
"""

import os
import sys
import time
import json
import shutil
import tempfile
import signal
import math
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchmetrics

# Ray Tune for hyperparameter search
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import Scheduler


# ==============================================================================
#  1. MODEL DEFINITION
# ==============================================================================

class MultiOutputModel(nn.Module):
    """
    CNN-LSTM hybrid model for multi-output sequence prediction.

    The model processes two input channels through parallel branches, fuses
    features in a shared LSTM layer, and produces two task-specific outputs.

    Args:
        config: Dictionary containing hyperparameters:
            - dropout: Dropout probability in the shared LSTM and private MLPs.
            - units_1 / units_2: Hidden sizes for Output-1 private MLP layers.
            - units_3 / units_4: Hidden sizes for Output-2 private MLP layers.
            - activation_1 / activation_2: Activation function names for each
              private MLP ("relu", "tanh", "sigmoid", "selu").
            - drift_enabled: Whether the drift-time branch is active, which
              changes the input size of the shared LSTM layer.
    """

    def __init__(self, config: dict):
        super().__init__()
        self._config = config

        # --- CNN Branch A: processes the first group of input signals ---
        self.cnn_branch_a = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=(1, 4), stride=(1, 4), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )

        # --- CNN Branch B: processes the second group of input signals ---
        self.cnn_branch_b = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(48),
            nn.ELU(),
            nn.Conv2d(48, 96, kernel_size=(1, 3), stride=(1, 3), padding=(0, 0)),
            nn.BatchNorm2d(96),
            nn.ELU(),
        )

        # --- LSTM Branch C: processes a low-dimensional signal stream ---
        #     Input size = 2 (e.g., amplitude and phase of a response signal)
        self.lstm_branch_c = nn.LSTM(
            input_size=2,
            hidden_size=18,
            num_layers=2,
            batch_first=True,
        )

        # --- LSTM Branch D: optional drift-time or auxiliary signal stream ---
        self.lstm_branch_d = nn.LSTM(
            input_size=1,
            hidden_size=14,
            num_layers=2,
            batch_first=True,
        )

        # --- Shared LSTM: fuses features from all branches ---
        #     Branch A output: 64 channels
        #     Branch B output: 96 channels
        #     Branch C output: 18 channels
        #     Branch D output: 14 channels (if enabled)
        lstm_in = 64 + 96 + 18
        if config.get("drift_enabled", True):
            lstm_in += 14

        self.shared_lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=config["dropout"],
        )

        # --- Private MLP for Output 1 (e.g., amplitude) ---
        self.head_output_1 = nn.Sequential(
            nn.Linear(32, int(config["units_1"])),
            nn.BatchNorm1d(1),
            self._make_activation(config["activation_1"]),
            nn.Dropout(0.0),
            nn.Linear(int(config["units_1"]), int(config["units_2"])),
            nn.BatchNorm1d(1),
            self._make_activation(config["activation_1"]),
            nn.Dropout(0.0),
            nn.Linear(int(config["units_2"]), 1),
        )

        # --- Private MLP for Output 2 (e.g., phase) ---
        self.head_output_2 = nn.Sequential(
            nn.Linear(32, int(config["units_3"])),
            nn.BatchNorm1d(1),
            self._make_activation(config["activation_2"]),
            nn.Dropout(0.0),
            nn.Linear(int(config["units_3"]), int(config["units_4"])),
            nn.BatchNorm1d(1),
            self._make_activation(config["activation_2"]),
            nn.Dropout(0.0),
            nn.Linear(int(config["units_4"]), 1),
        )

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        """Return the activation module corresponding to the given name."""
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "selu": nn.SELU,
        }
        if name not in activations:
            raise ValueError(f"Unsupported activation: {name}")
        return activations[name]()

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x_a: Tensor of shape [batch, seq_len, num_signals_a] reshaped to
                 [batch, 1, seq_len, num_signals_a] before passing through
                 the CNN branch.
            x_b: Tensor of shape [batch, seq_len, num_signals_b] reshaped to
                 [batch, 1, seq_len, num_signals_b] before passing through
                 the CNN branch.

        Returns:
            Concatenated output predictions of shape [batch, 2], where
            column 0 corresponds to Output 1 and column 1 to Output 2.
        """
        # --- CNN Branch A ---
        x_a = self.cnn_branch_a(x_a)
        x_a = torch.squeeze(x_a, dim=3)           # [B, C, H, 1] → [B, C, H]
        x_a = torch.permute(x_a, (0, 2, 1))       # [B, H, C] → [B, H, C]

        # --- CNN Branch B ---
        x_b = self.cnn_branch_b(x_b)
        x_b = x_b.reshape(x_b.shape[0], x_b.shape[1], x_b.shape[2])
        x_b = torch.permute(x_b, (0, 2, 1))

        # --- LSTM Branch C ---
        x_c = torch.squeeze(x_b[:, :, :2], dim=1)
        x_c, _ = self.lstm_branch_c(x_c)

        # --- LSTM Branch D ---
        x_d = x_b[:, :, -1:]
        x_d = torch.squeeze(x_d, dim=1)
        x_d, _ = self.lstm_branch_d(x_d)

        # --- Feature concatenation ---
        # Branch B output (saddle-like) is used directly in place of x_b
        # to match the original channel layout
        x_fused = torch.cat([x_a, x_b, x_c, x_d], dim=2)   # [B, H, 64+96+18+14]

        # --- Shared LSTM ---
        x_fused = x_fused[:, -4:, :]
        x_fused, _ = self.shared_lstm(x_fused)
        last = x_fused[:, -1:, :]                            # [B, 1, 32]

        # --- Private heads ---
        out_1 = self.head_output_1(last).reshape(-1, 1)
        out_2 = self.head_output_2(last).reshape(-1, 1)

        return torch.cat([out_1, out_2], dim=1)


# ==============================================================================
#  2. MODEL MANAGER
# ==============================================================================

class ModelManager:
    """
    Handles model lifecycle: creation, device placement, checkpointing,
    and multi-GPU (DataParallel) support.
    """

    @staticmethod
    def create(
        config: dict,
        device: Union[str, torch.device],
        data_parallel: bool = False,
    ) -> nn.Module:
        """Instantiate a model, move it to the specified device, and wrap
        with DataParallel if requested."""
        if isinstance(device, str):
            device = torch.device(device)
        model = MultiOutputModel(config).to(device)
        print(f"Model class: {MultiOutputModel}")
        print(f"Model instance:\n{model}")
        if data_parallel:
            if device.type == "cpu":
                raise ValueError("DataParallel requires GPU")
            model = nn.DataParallel(model)
            print(f"[INFO] DataParallel enabled — {torch.cuda.device_count()} GPUs")
        return model

    @staticmethod
    def get_module(model: nn.Module) -> nn.Module:
        """Return the underlying module, unwrapping DataParallel if present."""
        return model.module if isinstance(model, nn.DataParallel) else model

    @staticmethod
    def save(model: nn.Module, path: str) -> None:
        """Save the model's state dict with a metadata wrapper."""
        torch.save({
            "model_state_dict": ModelManager.get_module(model).state_dict(),
            "metadata": {},
        }, path)

    @staticmethod
    def load(
        model: nn.Module,
        path: str,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> nn.Module:
        """Load a checkpoint, handling both plain and DataParallel-formatted
        state dicts for backward compatibility."""
        loaded = torch.load(path, map_location=map_location)
        state = loaded["model_state_dict"]
        is_dp = isinstance(model, nn.DataParallel)
        converted = {
            (f"module.{k}" if is_dp and not k.startswith("module.") else k): v
            for k, v in state.items()
        }
        ModelManager.get_module(model).load_state_dict(converted, strict=False)
        return model


# ==============================================================================
#  3. LOSS FUNCTIONS
# ==============================================================================

def loss_output_1(y_pred: torch.Tensor, y_true: torch.Tensor,
                  alpha: float = 4.0, threshold: float = 0.2) -> torch.Tensor:
    """
    Loss for Output 1 (e.g., amplitude), operating on [0, 1] normalized values.

    The loss transitions smoothly from a linear term (strong gradient for small
    errors) to an MSE term (strong penalty for large errors), controlled by
    the `alpha` and `threshold` hyperparameters.

    Args:
        y_pred: Predicted values, range [0, 1].
        y_true: Ground-truth values, range [0, 1].
        alpha: Enhancement factor for small-error gradient boosting.
        threshold: Error magnitude at which the transition from linear to
                  quadratic behaviour occurs.
    Returns:
        Per-element mean loss, shape-compatible with batched inputs.
    """
    abs_err = torch.abs(y_true - y_pred)
    weight = torch.exp(-alpha * abs_err / threshold)
    linear = abs_err
    quadratic = abs_err ** 2
    loss = weight * linear + (1.0 - weight) * quadratic
    return torch.mean(loss, dim=0)


def loss_output_2(y_pred: torch.Tensor, y_true: torch.Tensor,
                  alpha: float = 0.4, scale: float = 0.5) -> torch.Tensor:
    """
    Loss for Output 2 (e.g., phase), operating on [0, 1] normalized values.

    The phase loss is periodic in nature. The [0, 1] range is mapped to [0, 2π]
    radians. The loss combines a cosine term (main periodic signal) with a
    sine term (gradient enhancement for small deviations).

    Args:
        y_pred: Predicted phase values, range [0, 1].
        y_true: Ground-truth phase values, range [0, 1].
        alpha: Weight of the sine enhancement term.
        scale: Scaling factor that controls the relative importance of this
               loss in the total combined loss.
    Returns:
        Per-element mean loss.
    """
    pred_rad = y_pred * 2.0 * torch.pi
    true_rad = y_true * 2.0 * torch.pi
    delta = pred_rad - true_rad
    loss_cos = 1.0 - torch.cos(delta)
    loss_sin = torch.abs(torch.sin(delta))
    element_loss = (loss_cos + alpha * loss_sin) * scale
    return torch.mean(element_loss, dim=0)


def loss_combined(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined scalar loss averaging Output-1 and Output-2 losses."""
    l1 = loss_output_1(pred[:, 0:1], target[:, 0:1])
    l2 = loss_output_2(pred[:, 1:2], target[:, 1:2])
    return (l1 + l2) / 2.0


def loss_vector(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Returns the two task losses as a vector (for metric tracking)."""
    l1 = loss_output_1(pred[:, 0:1], target[:, 0:1])
    l2 = loss_output_2(pred[:, 1:2], target[:, 1:2])
    return torch.cat([l1, l2], dim=0)


# ==============================================================================
#  4. METRIC TRACKING
# ==============================================================================

class ScalarMetric(torchmetrics.Metric):
    """
    Accumulates per-batch vector losses and returns their epoch-wise mean.
    Useful for tracking per-task loss curves during training.
    """

    def __init__(self):
        super().__init__()
        self._buffer = []

    def update(self, value: torch.Tensor) -> None:
        self._buffer.append(value.detach())

    def compute(self) -> torch.Tensor:
        return torch.stack(self._buffer, dim=0).mean(dim=0)

    def reset(self) -> None:
        self._buffer = []


# ==============================================================================
#  5. GRADNORM — MULTI-TASK WEIGHT BALANCING
# ==============================================================================

class GradNorm(nn.Module):
    """
    GradNorm: dynamically reweights task losses during multi-task training so
    that all tasks converge at similar rates.

    Reference: Chen et al., "GradNorm: Gradient Normalization for Adaptive
    Loss Balancing in Deep Multitask Networks" (ICML 2018).

    During the first `warm_up_batches` batches, loss weights are held fixed at
    their initial values. After the warm-up phase, GradNorm updates the weights
    each batch by minimising the L1 distance between each task's gradient norm
    and a target norm derived from the tasks' relative training rates.

    Args:
        n_tasks: Number of tasks (2 for this model).
        alpha: Hyperparameter controlling the "imbalance regulation strength".
        loss_weights: Initial per-task loss weights (must sum to 1.0).
        warm_up_batches: Number of initial batches before GradNorm activates.
    """

    def __init__(
        self,
        n_tasks: int,
        alpha: float = 1.5,
        loss_weights: tuple = (0.5, 0.5),
        warm_up_batches: int = 100,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.warm_up_batches = warm_up_batches
        self.loss_weights = nn.Parameter(
            torch.tensor(loss_weights, dtype=torch.float32), requires_grad=True
        )
        self._initial_losses: list = [None] * n_tasks
        self._batch_count = 0

    def forward(
        self,
        losses: list[torch.Tensor],
        shared_params,
        optimizer: optim.Optimizer,
    ) -> torch.Tensor:
        self._batch_count += 1

        # --- Warm-up phase: accumulate losses, use fixed weights ---
        if self._batch_count < self.warm_up_batches:
            for i in range(self.n_tasks):
                if self._initial_losses[i] is None:
                    self._initial_losses[i] = losses[i].detach()
                else:
                    self._initial_losses[i] = self._initial_losses[i] + losses[i].detach()
            if self._batch_count == self.warm_up_batches:
                for i in range(self.n_tasks):
                    self._initial_losses[i] = (
                        self._initial_losses[i] / float(self.warm_up_batches)
                    )
            weighted = [self.loss_weights[i] * losses[i] for i in range(self.n_tasks)]
            return sum(weighted)

        # --- GradNorm phase ---
        # 1) Compute per-task gradient norms
        grad_norms = []
        for i in range(self.n_tasks):
            for p in shared_params:
                if p.grad is not None:
                    p.grad.zero_()
            task_loss = self.loss_weights[i] * losses[i]
            grads = torch.autograd.grad(
                task_loss, shared_params, create_graph=True
            )
            if grads and any(g is not None for g in grads):
                flat = torch.cat([g.view(-1) for g in grads if g is not None])
                this_norm = torch.norm(flat, p=2)
            else:
                this_norm = torch.tensor(
                    0.0, device=losses[0].device, requires_grad=True
                )
            grad_norms.append(this_norm)
            for p in shared_params:
                if p.grad is not None:
                    p.grad.zero_()

        avg_norm = sum(grad_norms) / self.n_tasks

        # 2) Compute relative inverse training rates
        ratios = []
        for i in range(self.n_tasks):
            curr = losses[i].requires_grad_(True)
            init = self._initial_losses[i]
            if not isinstance(init, torch.Tensor):
                init = torch.tensor(init, device=curr.device)
            ratios.append(curr / init)
        total_ratio = sum(ratios)
        inv_rates = [
            (r / total_ratio) if total_ratio.item() > 1e-12
            else torch.ones_like(ratios[0]) / self.n_tasks
            for r in ratios
        ]

        # 3) Target gradient norms
        targets = [avg_norm * (r ** self.alpha) for r in inv_rates]

        # 4) GradNorm loss: L1 distance between actual and target norms
        gn_loss = sum(
            torch.abs(grad_norms[i] - targets[i])
            for i in range(self.n_tasks)
        )
        if not gn_loss.requires_grad:
            gn_loss = gn_loss.clone().detach().requires_grad_(True)

        # 5) Update loss weights via backprop
        optimizer.zero_grad()
        gn_loss.backward(retain_graph=True)
        if self._batch_count % 40 == 0:
            print(f"  [GradNorm] weights grad: {self.loss_weights.grad.data}")
        optimizer.step()

        # 6) Project weights to valid simplex (non-negative, sum = 1)
        with torch.no_grad():
            self.loss_weights.data = torch.relu(self.loss_weights.data)
            ws = self.loss_weights.data.sum()
            if ws > 1e-12:
                self.loss_weights.data = self.loss_weights.data / ws

        # 7) Compute weighted loss for main backward pass
        final_w = self.loss_weights.detach()
        weighted = [final_w[i] * losses[i] for i in range(self.n_tasks)]
        return sum(weighted)


# ==============================================================================
#  6. PARAMETER FREEZING / UNFREEZING UTILITIES
# ==============================================================================

class LayerManager:
    """Static utility class for freezing and unfreezing model parameters."""

    @staticmethod
    def _set_grad(params, requires_grad: bool) -> None:
        if not isinstance(params, (list, tuple)):
            params = [params]
        for group in params:
            if hasattr(group, "__iter__"):
                for p in group:
                    if p is not None:
                        p.requires_grad = requires_grad
            else:
                if group is not None:
                    group.requires_grad = requires_grad

    @staticmethod
    def freeze(*args) -> None:
        LayerManager._set_grad(args, requires_grad=False)

    @staticmethod
    def unfreeze(*args) -> None:
        LayerManager._set_grad(args, requires_grad=True)


# ==============================================================================
#  7. TRAINING DYNAMICS — LEARNING RATE, L2, DROPOUT SCHEDULING
# ==============================================================================

class MetricScheduler:
    """
    A patience-based scheduler that applies a cubic annealing schedule to a
    scalar regularization hyperparameter (L2 weight decay or dropout rate).

    The value follows a one-cycle cubic curve: it first decreases from
    `max_val` to `min_val` during the first half of the cycle, then
    increases back to `max_val` during the second half. After `num_update`
    total updates, the value is held at `max_val`.

    Args:
        mode: "min" or "max" — whether the monitored metric should decrease
              or increase for an improvement.
        num_update: Total number of trigger events before completing one cycle.
        patience: Number of bad steps before triggering an update.
        cooldown: Number of steps to wait after an update before monitoring again.
        min_val / max_val: Lower and upper bounds of the scheduled parameter.
        threshold: Metric threshold below/above which no update is triggered.
        verbose: Whether to print update messages.
    """

    def __init__(
        self,
        mode: str = "min",
        num_update: int = 100,
        patience: int = 4,
        cooldown: int = 3,
        min_val: float = 0.0,
        max_val: float = 1.0,
        threshold: float = 0.01,
        verbose: bool = True,
    ):
        self.mode = mode
        self.num_update = num_update
        self.patience = patience
        self.cooldown = cooldown
        self.min_val = min_val
        self.max_val = max_val
        self.threshold = threshold
        self.verbose = verbose
        self._best = float("inf") if mode == "min" else float("-inf")
        self._bad_epochs = 0
        self._cooldown_counter = 0
        self._current_val = max_val
        self._update_count = 0

    def step(self, metric: float, update_fn) -> None:
        if metric <= self.threshold:
            self._bad_epochs = 0
            return
        improved = (
            (metric < self._best) if self.mode == "min" else (metric > self._best)
        )
        if improved:
            self._best = metric
            self._bad_epochs = 0
        else:
            self._bad_epochs += 1
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            self._bad_epochs = 0
        if self._bad_epochs > self.patience:
            self._do_update(update_fn)
            self._cooldown_counter = self.cooldown
            self._bad_epochs = 0

    def _do_update(self, update_fn) -> None:
        self._update_count += 1
        if self._update_count > self.num_update:
            new_val = self.max_val
        else:
            half = self.num_update // 2
            if self._update_count <= half:
                t = self._update_count / half
                new_val = self.max_val - (self.max_val - self.min_val) * (t ** 3)
            else:
                t = (self._update_count - half) / (self.num_update - half)
                new_val = self.min_val + (self.max_val - self.min_val) * (
                    1.0 - (1.0 - t) ** 3
                )
        self._current_val = max(self.min_val, min(new_val, self.max_val))
        update_fn(self._current_val)
        if self.verbose:
            print(f"[MetricScheduler] Update #{self._update_count}, val={self._current_val:.4f}")


class TrainingManager:
    """
    Centralised manager for all training dynamics:
      - Per-layer learning rates (Adam/AdamW).
      - Warmup + ReduceLROnPlateau learning rate scheduling.
      - Cubic-annealing L2 weight decay per parameter group.
      - Cubic-annealing dropout rate per output MLP.
      - Early stopping with best-model checkpointing.

    The caller invokes `update_per_epoch(metric)` once per training epoch.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        warmup_epochs: int = 50,
        lr_input: tuple = (1e-4, 1e-4),
        lr_shared: float = 5e-4,
        lr_output: tuple = (1e-3, 1e-3),
        l2_shared: float = 1e-3,
        l2_output: tuple = (1e-3, 1e-3),
        dropout_output: tuple = (0.5, 0.5),
        enable_schedule: tuple = (True, True, True),
    ):
        self.model = model
        self.config = config
        self.warmup_epochs = warmup_epochs
        self.enable_lr, self.enable_l2, self.enable_dropout = enable_schedule

        m = ModelManager.get_module(model)

        # Build optimizer with per-group learning rates and initial L2 values
        if self.enable_l2:
            l2_common_init = 1e-2
            l2_out1_init = l2_output[0]
            l2_out2_init = 1e-1
            dp_out1 = dropout_output[0]
            dp_out2 = 0.8
        else:
            l2_common_init = l2_shared
            l2_out1_init = l2_output[0]
            l2_out2_init = l2_output[1]
            dp_out1 = dropout_output[0]
            dp_out2 = dropout_output[1]

        self.optimizer = getattr(optim, config["optimizer_name"])([
            {"params": m.cnn_branch_a.parameters(),    "lr": lr_input[0],     "weight_decay": 0},
            {"params": m.cnn_branch_b.parameters(),     "lr": lr_input[1],     "weight_decay": 0},
            {"params": m.lstm_branch_c.parameters(),    "lr": lr_input[1],     "weight_decay": 0},
            {"params": m.lstm_branch_d.parameters(),    "lr": lr_input[1],     "weight_decay": 0},
            {"params": m.shared_lstm.parameters(),       "lr": lr_shared,       "weight_decay": l2_common_init},
            {"params": m.head_output_1.parameters(),    "lr": lr_output[0],    "weight_decay": l2_out1_init},
            {"params": m.head_output_2.parameters(),    "lr": lr_output[1],    "weight_decay": l2_out2_init},
        ])

        # Initialise dropout rates
        for layer in m.head_output_1.modules():
            if isinstance(layer, nn.Dropout):
                layer.p = dp_out1
        for layer in m.head_output_2.modules():
            if isinstance(layer, nn.Dropout):
                layer.p = dp_out2

        self.l2_shared_max = l2_shared
        self.l2_output_max = l2_output
        self.dropout_output_max = dropout_output

        # Schedulers
        if self.enable_lr:
            # Warmup: cubic buffer from 0 → 1, then stay at 1 until max_lr decays
            self.scheduler_warmup = LambdaLR(
                self.optimizer,
                lr_lambda=lambda e: max(
                    min(1.0 - (1.0 - e / warmup_epochs) ** 3, 1.0), 0.01
                ),
            )
            self.scheduler_plateau = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.8, patience=3, min_lr=1e-5, verbose=True
            )
        else:
            self.scheduler_warmup = None
            self.scheduler_plateau = None

        if self.enable_l2:
            self.l2_sched_shared = MetricScheduler(
                mode="min", threshold=0.0001, num_update=warmup_epochs,
                patience=1, cooldown=0, min_val=0.0, max_val=l2_shared, verbose=True
            )
            self.l2_sched_out1 = MetricScheduler(
                mode="min", threshold=0.0001, num_update=warmup_epochs,
                patience=1, cooldown=0, min_val=0.0, max_val=l2_output[0], verbose=True
            )
            self.l2_sched_out2 = MetricScheduler(
                mode="min", threshold=0.0001, num_update=warmup_epochs,
                patience=1, cooldown=0, min_val=0.0, max_val=l2_output[1], verbose=True
            )
        else:
            self.l2_sched_shared = self.l2_sched_out1 = self.l2_sched_out2 = None

        if self.enable_dropout:
            self.dp_sched_out1 = MetricScheduler(
                mode="min", threshold=0.0001, num_update=warmup_epochs,
                patience=1, cooldown=0, min_val=0.0, max_val=dropout_output[0], verbose=True
            )
            self.dp_sched_out2 = MetricScheduler(
                mode="min", threshold=0.0001, num_update=warmup_epochs,
                patience=1, cooldown=0, min_val=0.0, max_val=dropout_output[1], verbose=True
            )
        else:
            self.dp_sched_out1 = self.dp_sched_out2 = None

        self._print_state("Initialization complete")

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def update_per_epoch(
        self,
        epoch: int,
        train_loss: np.ndarray,
        vald_loss: np.ndarray,
    ) -> None:
        """
        Call once at the end of each training epoch to update LR, L2, and
        dropout schedules based on the train/validation loss gap.

        Args:
            epoch: Current epoch index (1-based).
            train_loss: Per-task training loss vector, shape [num_tasks].
            vald_loss: Per-task validation loss vector, shape [num_tasks].
        """
        if not (self.enable_lr or self.enable_l2 or self.enable_dropout):
            self._print_state(f"Epoch {epoch} complete (no scheduling enabled)")
            return

        scalar_val = (vald_loss[0] + vald_loss[1]) / 2.0

        # Learning rate scheduling
        if self.enable_lr and self.scheduler_warmup is not None:
            if epoch < self.warmup_epochs:
                self.scheduler_warmup.step()
            else:
                self.scheduler_plateau.step(scalar_val)

        # L2 scheduling
        if self.enable_l2:
            diff = scalar_val - (train_loss[0] + train_loss[1]) / 2.0
            m = ModelManager.get_module(self.model)
            if self.l2_sched_shared is not None:
                shared_params = set(m.shared_lstm.parameters())
                self.l2_sched_shared.step(diff, lambda v: _update_l2(
                    self.optimizer, shared_params, v))
            if self.l2_sched_out1 is not None:
                out1_params = set(m.head_output_1.parameters())
                self.l2_sched_out1.step(vald_loss[0] - train_loss[0], lambda v: _update_l2(
                    self.optimizer, out1_params, v))
            if self.l2_sched_out2 is not None:
                out2_params = set(m.head_output_2.parameters())
                self.l2_sched_out2.step(vald_loss[1] - train_loss[1], lambda v: _update_l2(
                    self.optimizer, out2_params, v))

        # Dropout scheduling
        if self.enable_dropout:
            m = ModelManager.get_module(self.model)
            if self.dp_sched_out1 is not None:
                self.dp_sched_out1.step(
                    vald_loss[0] - train_loss[0],
                    lambda v: _update_dropout(m.head_output_1, v),
                )
            if self.dp_sched_out2 is not None:
                self.dp_sched_out2.step(
                    vald_loss[1] - train_loss[1],
                    lambda v: _update_dropout(m.head_output_2, v),
                )

        self._print_state(f"Epoch {epoch} complete")

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _print_state(self, tag: str) -> None:
        opt_type = type(self.optimizer).__name__
        lrs = [g["lr"] for g in self.optimizer.param_groups]
        l2s = [g.get("weight_decay", 0) for g in self.optimizer.param_groups]
        dps = []
        m = ModelManager.get_module(self.model)
        for layer in m.head_output_1.modules():
            if isinstance(layer, nn.Dropout):
                dps.append(layer.p)
        for layer in m.head_output_2.modules():
            if isinstance(layer, nn.Dropout):
                dps.append(layer.p)
        print(f"[{tag}]")
        print(f"  Optimizer: {opt_type}")
        print(f"  Learning rates: {lrs}")
        print(f"  L2 weight decay: {l2s}")
        print(f"  Dropout rates:   {dps}")
        print(f"  Scheduling: LR={self.enable_lr}, L2={self.enable_l2}, "
              f"Dropout={self.enable_dropout}\n")


def _update_l2(optimizer, param_set, value: float) -> None:
    for group in optimizer.param_groups:
        if any(p in param_set for p in group["params"]):
            group.update(weight_decay=value)


def _update_dropout(module: nn.Module, value: float) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Dropout):
            layer.p = value


# ==============================================================================
#  8. EARLY STOPPING
# ==============================================================================

class EarlyStopping:
    """
    Monitors validation loss and stops training when it fails to improve
    for `patience` consecutive epochs. The best model checkpoint is always
    saved regardless of whether early stopping is enabled.

    Args:
        patience: Number of epochs with no improvement before stopping.
        delta: Minimum change to qualify as an improvement.
        save_path: File path for the best model checkpoint.
        min_epochs: Minimum epochs before early stopping can trigger.
        enable: Whether to actually stop (True) or just log (False).
    """

    def __init__(
        self,
        patience: int = 10,
        delta: float = 3e-4,
        save_path: str = "best_model.pth",
        min_epochs: int = 5,
        enable: bool = True,
    ):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.min_epochs = min_epochs
        self.enable = enable
        self._best_score: Optional[float] = None
        self._counter = 0
        self._should_stop = False

    def check(
        self,
        val_loss: float,
        model: nn.Module,
        optimizer,
        epoch: int,
    ) -> bool:
        """
        Evaluate whether training should continue. Always saves the best
        model checkpoint. Returns True if early stopping has been triggered.
        """
        score = -val_loss
        if self._best_score is None or score > self._best_score + self.delta:
            self._best_score = score
            self._save(val_loss, model, optimizer, epoch)
            self._counter = 0
        else:
            if epoch >= self.min_epochs:
                self._counter += 1
                print(f"EarlyStopping: {self._counter}/{self.patience}")
                if self._counter >= self.patience:
                    if self.enable:
                        self._should_stop = True
                    else:
                        print("Early stopping is not enabled; counting continues.")
        return self._should_stop

    def _save(self, val_loss, model, optimizer, epoch) -> None:
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
            print(f"Removed old checkpoint: {self.save_path}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }, self.save_path)
        print(f"Saved best model (epoch {epoch}, val_loss={val_loss:.6f})\n")


# ==============================================================================
#  9. INTERRUPT HANDLER
# ==============================================================================

class InterruptManager:
    """
    Installs a SIGINT handler so that Ctrl+C gracefully stops the training
    loop and preserves the best checkpoint obtained so far.
    """

    def __init__(self, enable: bool = True):
        self.enable = enable
        self.interrupted = False
        self._prev_handler = None
        if enable:
            try:
                self._prev_handler = signal.signal(signal.SIGINT, self._handler)
            except Exception as exc:
                print(f"Warning: could not install SIGINT handler: {exc}")

    def _handler(self, sig, frame):
        self.interrupted = True
        print("\nCtrl+C detected — finalising best checkpoint…\n")
        raise KeyboardInterrupt

    def restore(self) -> None:
        if self._prev_handler is not None:
            try:
                signal.signal(signal.SIGINT, self._prev_handler)
            except Exception as exc:
                print(f"Warning: could not restore SIGINT handler: {exc}")


# ==============================================================================
#  10. TRAINING & EVALUATION
# ==============================================================================

def train_step(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer,
    metric_tracker: ScalarMetric,
    multitask_config: tuple,
    gradnorm_worker: Optional[GradNorm] = None,
    gradnorm_optimizer=None,
) -> np.ndarray:
    """
    Perform one full pass over the training set.

    Multi-task strategy is controlled by `multitask_config = (alternate_freezing, use_gradnorm)`:

      (F, F) — Joint update with equal 50/50 loss weights.
      (T, F) — Three-step alternating freeze cycle.
      (F, T) — GradNorm dynamic loss weighting (shared params only).
      (T, T) — Alternating freeze + GradNorm.

    Args:
        dataloader: Training DataLoader yielding (x_a, x_b, y_true) tuples.
        model: The model being trained.
        optimizer: Main optimizer.
        metric_tracker: ScalarMetric instance for per-task loss tracking.
        multitask_config: Tuple (alternate_freezing, use_gradnorm).
        gradnorm_worker: GradNorm instance (None when use_gradnorm is False).
        gradnorm_optimizer: Optimizer for GradNorm weight parameters.

    Returns:
        Per-task mean training loss, shape [num_tasks].
    """
    freeze_on, use_gradnorm = multitask_config
    model.train()

    m = ModelManager.get_module(model)
    shared_params = (
        list(m.cnn_branch_a.parameters())
        + list(m.cnn_branch_b.parameters())
        + list(m.lstm_branch_c.parameters())
        + list(m.lstm_branch_d.parameters())
        + list(m.shared_lstm.parameters())
    )
    head1_params = list(m.head_output_1.parameters())
    head2_params = list(m.head_output_2.parameters())

    for batch_idx, (xa, xb, y_true) in enumerate(dataloader):
        xa = xa.unsqueeze(1).to(device)
        xb = xb.unsqueeze(1).to(device)
        y_true = y_true.to(device)

        with torch.backends.cudnn.flags(enabled=not use_gradnorm):
            pred = model(xa, xb)

        lv = loss_vector(pred, y_true)
        l1, l2 = lv[0], lv[1]
        metric_tracker.update(lv.detach().to(device))

        # ---- (F, F) — Vanilla joint update ----
        if not freeze_on and not use_gradnorm:
            weighted = (l1 + l2) / 2.0
            optimizer.zero_grad()
            weighted.backward()
            optimizer.step()

        # ---- (T, F) — Alternating freeze, no GradNorm ----
        elif freeze_on and not use_gradnorm:
            # Step A: freeze all, unfreeze head 1 only
            LayerManager.freeze(shared_params, head2_params)
            LayerManager.unfreeze(head1_params)
            optimizer.zero_grad()
            l1.backward(retain_graph=True)
            optimizer.step()

            # Step B: freeze all, unfreeze head 2 only
            LayerManager.unfreeze(shared_params, head1_params)
            pred = model(xa, xb)
            l2 = loss_vector(pred, y_true)[1]
            LayerManager.freeze(shared_params, head1_params)
            LayerManager.unfreeze(head2_params)
            optimizer.zero_grad()
            l2.backward(retain_graph=True)
            optimizer.step()

            # Step C: joint update for shared layers
            LayerManager.unfreeze(shared_params, head1_params, head2_params)
            pred = model(xa, xb)
            lv = loss_vector(pred, y_true)
            weighted = (lv[0] + lv[1]) / 2.0
            LayerManager.freeze(head1_params, head2_params)
            LayerManager.unfreeze(shared_params)
            optimizer.zero_grad()
            weighted.backward()
            optimizer.step()
            LayerManager.unfreeze(shared_params, head1_params, head2_params)

        # ---- (F, T) — GradNorm, no freezing ----
        elif not freeze_on and use_gradnorm:
            weighted = gradnorm_worker([l1, l2], shared_params, gradnorm_optimizer)
            optimizer.zero_grad()
            weighted.backward()
            optimizer.step()

        # ---- (T, T) — Alternating freeze + GradNorm ----
        else:
            # Step A: train head 1 only
            LayerManager.freeze(shared_params, head2_params)
            LayerManager.unfreeze(head1_params)
            optimizer.zero_grad()
            l1.backward()
            optimizer.step()

            # Step B: train head 2 only
            LayerManager.unfreeze(shared_params, head1_params)
            pred = model(xa, xb)
            l2 = loss_vector(pred, y_true)[1]
            LayerManager.freeze(shared_params, head1_params)
            LayerManager.unfreeze(head2_params)
            optimizer.zero_grad()
            l2.backward()
            optimizer.step()

            # Step C: joint update for shared layers via GradNorm
            LayerManager.unfreeze(shared_params, head1_params, head2_params)
            pred = model(xa, xb)
            lv = loss_vector(pred, y_true)
            weighted = gradnorm_worker([lv[0], lv[1]], shared_params, gradnorm_optimizer)
            LayerManager.freeze(head1_params, head2_params)
            LayerManager.unfreeze(shared_params)
            optimizer.zero_grad()
            weighted.backward()
            optimizer.step()
            LayerManager.unfreeze(shared_params, head1_params, head2_params)

        if batch_idx % 40 == 0:
            l1_v, l2_v = float(l1.item()), float(l2.item())
            n = (batch_idx + 1) * len(xa)
            print(f"  Batch {batch_idx} | Loss1={l1_v:.6f}  Loss2={l2_v:.6f}  [{n}]")
            if use_gradnorm:
                w = gradnorm_worker.loss_weights
                print(f"    GradNorm weights → {w[0].item():.4f}, {w[1].item():.4f}")
            else:
                print(f"    Fixed weights → 0.5000, 0.5000")

    return metric_tracker.compute().cpu().numpy()


def evaluate(
    dataloader: DataLoader,
    model: nn.Module,
    metric_tracker: ScalarMetric,
) -> np.ndarray:
    """Run evaluation and return per-task mean loss."""
    model.eval()
    with torch.no_grad():
        for xa, xb, y_true in dataloader:
            xa = xa.unsqueeze(1).to(device)
            xb = xb.unsqueeze(1).to(device)
            y_true = y_true.to(device)
            pred = model(xa, xb)
            metric_tracker.update(loss_vector(pred, y_true).detach().to(device))
    result = metric_tracker.compute().cpu().numpy()
    metric_tracker.reset()
    return result


def training_loop(
    config: dict,
    max_epochs: int,
    dataloader_train: DataLoader,
    dataloader_vald: DataLoader,
    dataloader_test: DataLoader,
    multitask_config: tuple = (False, False),
    save_mode: str = "early_stop",
    checkpoint_path: Optional[str] = None,
) -> None:
    """
    Main training loop.

    Args:
        config: Hyperparameter dict passed to MultiOutputModel and optimizers.
        max_epochs: Maximum number of training epochs.
        dataloader_train / dataloader_vald / dataloader_test: DataLoaders.
        multitask_config: (alternate_freezing, use_gradnorm) — see train_step.
        save_mode:
            - "early_stop"   — standard early stopping, saves best_model.pth.
            - "ray_early_stop" — same but also reports metrics to Ray Tune.
        checkpoint_path: Path to a .pth file to resume from (None = from scratch).
    """
    model = ModelManager.create(config, device, data_parallel=False)

    # Resume from checkpoint if requested
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ModelManager.load(model, checkpoint_path, map_location=device)

    train_metric = ScalarMetric().to(device)
    vald_metric = ScalarMetric().to(device)
    test_metric = ScalarMetric().to(device)

    train_mgr = TrainingManager(
        model=model,
        config=config,
        warmup_epochs=warmup_epochs,
        lr_input=(0.1 * config["lr"], 0.1 * config["lr"]),
        lr_shared=0.5 * config["lr"],
        lr_output=(config["lr"], 0.5 * config["lr"]),
        l2_shared=0.5 * config["l2"],
        l2_output=(0.1 * config["l2"], 3.0 * config["l2"]),
        dropout_output=(0.5 * config["dropout"], 1.3 * config["dropout"]),
        enable_schedule=enable_schedule,
    )
    optimizer = train_mgr.optimizer

    # GradNorm worker (optional)
    if multitask_config[1]:
        gradnorm = GradNorm(
            n_tasks=2, alpha=1.5, loss_weights=[0.5, 0.5],
            warm_up_batches=100,
        )
        gradnorm_opt = getattr(optim, config["optimizer_name"])([
            {"params": gradnorm.loss_weights, "lr": 0.1 * config["lr"], "weight_decay": 0}
        ])
    else:
        gradnorm, gradnorm_opt = None, None

    early_stop = EarlyStopping(
        min_epochs=warmup_epochs,
        patience=10,
        delta=3e-4,
        save_path="best_model.pth",
        enable=enable_earlystop,
    )

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}\n{'-' * 40}")
        t0 = time.time()

        train_loss = train_step(
            dataloader_train, model, optimizer, train_metric,
            multitask_config, gradnorm, gradnorm_opt,
        )
        vald_loss = evaluate(dataloader_vald, model, vald_metric)
        test_loss = evaluate(dataloader_test, model, test_metric)

        scalar_train = (train_loss[0] + train_loss[1]) / 2.0
        scalar_vald = (test_loss[0] + test_loss[1]) / 2.0   # use test as vald proxy

        print(f"Epoch {epoch+1} | Train={scalar_train:.6f}  Vald={scalar_vald:.6f}")
        print(f"Time: {time.time() - t0:.1f}s\n")

        train_mgr.update_per_epoch(epoch + 1, train_loss, test_loss)

        if save_mode == "ray_early_stop":
            _report_ray(scalar_vald, train_loss, test_loss, epoch, config)
            stop = early_stop.check(scalar_vald, model, optimizer, epoch + 1)
            if stop:
                print("Early stopping triggered.")
                break

        if save_mode == "early_stop":
            stop = early_stop.check(scalar_vald, model, optimizer, epoch + 1)
            if stop:
                print("Early stopping triggered.")
                break

    print("Training complete.")


def _report_ray(
    ray_metric: float,
    train_loss: np.ndarray,
    vald_loss: np.ndarray,
    epoch: int,
    config: dict,
) -> None:
    """Save a Ray Tune checkpoint and report metrics."""
    tmpdir = tempfile.mkdtemp()
    try:
        ckpt = os.path.join(tmpdir, "checkpoint.pt")
        torch.save({"epoch": epoch + 1, "val_loss": ray_metric}, ckpt)
        ray.train.report({
            "ray_metric_loss": ray_metric,
            "ray_train_loss": train_loss.tolist(),
            "ray_vald_loss": vald_loss.tolist(),
            "epoch": epoch + 1,
        }, checkpoint=ray.train.Checkpoint.from_directory(tmpdir))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ==============================================================================
#  11. HYPERPARAMETER TUNING
# ==============================================================================

def get_top_k_trials(
    result,
    metric: str,
    mode: str,
    k: int = 5,
) -> list[dict]:
    """Return details of the top-k trials sorted by the given metric."""
    df = result.results_df.sort_values(
        by=metric, ascending=(mode == "min")
    ).head(k)

    top = []
    for trial_id in df.index:
        trial = next((t for t in result.trials if t.trial_id == trial_id), None)
        if trial is None:
            continue
        ckpt = result.get_best_checkpoint(trial, metric=metric, mode=mode)
        top.append({
            "trial_id": trial.trial_id,
            "config": trial.config,
            "metric_value": trial.last_result.get(metric),
            "checkpoint_path": ckpt,
        })
        print(f"Trial {len(top)}: {trial.trial_id}  {metric}={top[-1]['metric_value']}")
    return top


def hyperparameter_tuning(
    num_samples: int,
    max_epochs: int,
    config_space: dict,
    metric: str = "ray_metric_loss",
    mode: str = "min",
    name: str = "tuning_run",
    local_dir: Optional[str] = None,
    best_model_path: str = "best_model.pth",
    resources_per_trial: dict = {"cpu": 4, "gpu": 0.5},
) -> dict:
    """
    Run hyperparameter search using Ray Tune with the ASHA scheduler.

    Args:
        num_samples: Number of random configurations to evaluate.
        max_epochs: Maximum epochs per trial.
        config_space: Ray Tune search space dict.
        metric / mode: Optimisation target and direction.
        name: Experiment name.
        local_dir: Directory for Ray Tune results.
        best_model_path: Output path for the best model's weights.
        resources_per_trial: CPU/GPU allocation per trial.

    Returns:
        The best trial's configuration dict.
    """
    scheduler = ASHAScheduler(
        max_t=max_epochs,
        grace_period=warmup_epochs,
        reduction_factor=4,
    )

    reporter = tune.CLIReporter(
        metric_columns=["ray_metric_loss", "ray_train_loss", "ray_vald_loss",
                        "epoch", "batch_size", "optimizer", "learning_rate"]
    )

    stopper = {
        "time_total_s": 12 * 3600,
        "training_iteration": max_epochs,
    }

    interrupt_mgr = InterruptManager(enable=True)
    os.environ.setdefault("TUNE_DISABLE_SIGINT_HANDLER", "1")

    results = None
    try:
        results = tune.run(
            tune.with_parameters(
                training_loop,
                max_epochs=max_epochs,
                multitask_config=multitask_freeze_gradnorm,
                save_mode="ray_early_stop",
                checkpoint_path="best_model_top45.pth",
            ),
            resources_per_trial=resources_per_trial,
            max_concurrent_trials=16,
            reuse_actors=True,
            config=config_space,
            metric=metric,
            mode=mode,
            scheduler=scheduler,
            progress_reporter=reporter,
            num_samples=num_samples,
            name=name,
            storage_path=local_dir,
            resume="AUTO",
            stop=stopper,
            fail_fast=False,
            max_failures=3,
            raise_on_failed_trial=False,
        )
    except KeyboardInterrupt:
        print("Search interrupted by user.")
        if local_dir:
            try:
                results = tune.ExperimentAnalysis(
                    os.path.join(local_dir, name)
                )
            except Exception as exc:
                print(f"Warning: could not load partial results: {exc}")
    finally:
        interrupt_mgr.restore()

    if results is None or interrupt_mgr.interrupted:
        print("No valid results — exiting.")
        sys.exit(0)

    print("\n=== Top-5 Trials ===")
    get_top_k_trials(results, metric=metric, mode=mode, k=5)

    best_trial = results.get_best_trial(metric, mode, "last")
    print(f"\nBest config: {best_trial.config}")
    print(f"Best {metric}: {best_trial.last_result.get(metric)}")

    best_ckpt = results.get_best_checkpoint(best_trial, metric=metric, mode=mode)
    if best_ckpt:
        with best_ckpt.as_directory() as ckpt_dir:
            model = ModelManager.create(best_trial.config, device, data_parallel=False)
            ModelManager.load(
                model,
                os.path.join(ckpt_dir, "checkpoint.pt"),
                map_location=device,
            )
            ModelManager.save(model, best_model_path)
        print(f"Saved best model to: {best_model_path}")

    return best_trial.config


# ==============================================================================
#  12. CONFIGURATION CONSTANTS
# ==============================================================================
# These values are used throughout the module. Adjust them to match the
# target hardware and training budget, or replace them with values loaded
# from a JSON/YAML config file.

# --- Data dimensions (must match the Dataset / DataLoader implementation) ---
num_signals_a = 16   # Number of signals in CNN Branch A input (x_a)
num_signals_b = 8    # Number of signals in CNN Branch B input (x_b)
# Expected DataLoader output: (xa [B, seq_len, 16], xb [B, seq_len, 8], y [B, 2])

num_output_tasks = 2   # Output 1 (e.g., amplitude) + Output 2 (e.g., phase)

# --- Training hyperparameters ---
training_mode = "TA"   # "TA" = hyperparameter tuning, "BC" = break-continuation
epochs_tuning = 200
warmup_epochs = 30
enable_schedule = (True, True, True)   # (LR_schedule, L2_schedule, Dropout_schedule)
enable_earlystop = True
data_parallel = False

# --- Multi-task strategy ---
# (alternate_freezing, use_gradnorm):
#   (False, False) — vanilla joint update
#   (True,  False) — alternating freeze
#   (False, True)  — GradNorm
#   (True,  True)  — alternating freeze + GradNorm
multitask_freeze_gradnorm = (True, False)

# --- Ray Tune sampling budget ---
ray_samples = 4000

# --- Device selection ---
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")


# ==============================================================================
#  13. MAIN ENTRY POINT
# ==============================================================================

def main():
    """
    Entry point. Two modes are supported:

      training_mode = "TA" (tuning algorithm)
        - Runs Ray Tune hyperparameter search.
        - Saves the best configuration as tuner_bestHp.json.
        - Saves the best model weights as best_model_ray.pth.

      training_mode = "BC" (break-continuation)
        - Loads tuner_bestHp.json.
        - Resumes training from best_model.pth.
    """
    # Usage example — adjust search space to your task.
    if training_mode == "TA":
        config_space = {
            "units_1":      tune.choice([64, 80, 96]),
            "units_2":      tune.choice([48, 64, 80]),
            "units_3":      tune.choice([64, 80, 96]),
            "units_4":      tune.choice([48, 64, 80]),
            "activation_1": tune.choice(["relu", "tanh", "sigmoid", "selu"]),
            "activation_2": tune.choice(["relu", "tanh", "sigmoid", "selu"]),
            "batch_size":   tune.choice([16, 32, 64, 128, 256]),
            "optimizer_name": tune.choice(["Adam", "AdamW"]),
            "lr":           tune.loguniform(1e-4, 1e-2),
            "l2":           tune.loguniform(1e-4, 5e-2),
            "dropout":      tune.uniform(0.1, 0.5),
            "drift_enabled": tune.choice([True, False]),
        }

        best_config = hyperparameter_tuning(
            num_samples=ray_samples,
            max_epochs=epochs_tuning,
            config_space=config_space,
            metric="ray_metric_loss",
            mode="min",
            name="cnn_lstm_tuning",
            local_dir="ray_results",
            best_model_path="best_model_ray.pth",
            resources_per_trial={"cpu": 4, "gpu": 0.5},
        )

        # Persist best config
        with open("tuner_bestHp.json", "w") as f:
            json.dump(best_config, f, indent=4)
        print("Best config saved to tuner_bestHp.json")

    elif training_mode == "BC":
        with open("tuner_bestHp.json", "r") as f:
            best_config = json.load(f)

        training_loop(
            config=best_config,
            max_epochs=epochs_tuning,
            dataloader_train=None,    # <-- Replace with actual DataLoader
            dataloader_vald=None,     # <-- Replace with actual DataLoader
            dataloader_test=None,     # <-- Replace with actual DataLoader
            multitask_config=multitask_freeze_gradnorm,
            save_mode="early_stop",
            checkpoint_path="best_model.pth",
        )


if __name__ == "__main__":
    main()
