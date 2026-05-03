"""
CNN-iTransformer: Multi-Task Neural Network for Plasma Response Prediction
============================================================================

Author: [Jian Xu] <[jianxu@gmail.com]>
        [Dalian University of Technology]

============================================================================

Architecture: CNN feature extraction + LSTM temporal encoding + iTransformer backbone
              with dual task-specific heads for amplitude and phase prediction.

Forward pass pipeline:
  Stage 1 — CNN feature branches:
    rmp_upper_branch: Conv2d*2  -> (N,4,64)
    rmp_lower_branch: Conv2d*2  -> (N,4,64)
    saddle_branch:    Conv2d*2  -> (N,4,96)
  Stage 2 — LSTM temporal branches:
    response_branch: LSTM       -> (N,8,18)
    drift_branch:    LSTM       -> (N,8,14)  [optional]
  Stage 3 — Backbone:
    Concatenate -> (N,4,242/256)
    iTransformer backbone -> (N,1,128)
  Stage 4 — Task heads:
    amp_head:   Linear*3 -> (N,1)
    phase_head: Linear*3 -> (N,1)
  Output: concat -> (N,2)  [amplitude, phase]

Multi-task learning: Supports alternate parameter freezing and GradNorm (Chen et al., ICLR 2018).
Dynamic scheduling:  Per-epoch cubic schedule for learning rate, L2 weight decay, and dropout.
Hyperparameter tuning: Ray Tune integration with ASHA scheduler.

================================================================================
KEY VARIABLES (arguments / config fields)
================================================================================
Model configuration:
  sequence_length       — length of input time window for sliding-window slicing
  use_drift_feature     — whether to include drift time as a feature channel
  dropout               — dropout probability for task heads (also used in transformer)
  units_1..units_4      — hidden layer sizes for amp_head and phase_head MLPs
  activation_1          — activation function name for amp_head ('relu' or 'tanh')
  activation_2          — activation function name for phase_head ('relu' or 'tanh')

Training configuration:
  lr                   — base learning rate
  l2                   — base L2 weight decay
  batch_size           — mini-batch size
  optimizer_name       — optimizer class ('Adam' or 'AdamW')
  warmup_epochs        — number of warm-up epochs before ReduceLROnPlateau takes over
  scheduling_flags     — (enable_lr, enable_l2, enable_dropout) as bool tuple
  enable_early_stop    — whether to trigger early stopping
  data_parallel        — whether to wrap model with nn.DataParallel
  mtl_strategy         — (enable_alternate_freeze, enable_gradnorm) as bool tuple
  dimen_input          — total number of input feature channels
  dimen_output         — number of output targets (always 2: amplitude + phase)

Data tensors (shapes as seen by model forward):
  x_rmp   (N,1,8,16)  — RMP coil signals: upper 8 + lower 8 channels
  x_phys  (N,1,8,8)   — Physical observables: 6 saddle coils + 2 response channels
                        or (N,1,8,9) if drift feature is enabled (last channel)
  y       (N,2)        — Labels: [amplitude, phase]

External dependencies:
  - PyTorch >= 2.0
  - iTransformer (pip install i-transformer)
  - Ray Tune (for hyperparameter search)
================================================================================

Usage:
    # As a library:
    from cnn_itransformer import (
        PlasmaResponseNet, ModelManager, TrainManager, GradNorm,
        EarlyStopping, CustomLossMetric, MetricScheduler,
        vector_loss, amplitude_loss, phase_loss,
        hyperparameter_tuning, execute_training_loop,
    )

    # Standalone training:
    from cnn_itransformer import execute_training_loop, ModelManager
    model = ModelManager.create(config, device='cuda')
    execute_training_loop(config, max_epochs=200, save_mode='early_stop', device='cuda')
"""

import os
import json
import time
import shutil
import tempfile
import signal
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from typing import Union, Dict, Tuple, List, Optional, Any

# =============================================================================
# PART 0 — LOSS FUNCTIONS
# =============================================================================

def amplitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    enhancement_factor: float = 4.0,
    transition_threshold: float = 0.2,
) -> torch.Tensor:
    """
    Enhanced MSE loss for better gradient flow in small-error regions.

    Combines a linear term (strong gradients near zero) with MSE (quadratic
    for large errors) via a smooth exponential weight that transitions
    between the two regimes near the threshold.

    Args:
        pred:                 Predicted values, normalized to [0, 1].
        target:               Ground truth values, normalized to [0, 1].
        enhancement_factor:    Controls gradient strength for small errors.
        transition_threshold:  Controls where the transition from linear to MSE occurs.

    Returns:
        Mean loss over the batch.
    """
    abs_error = torch.abs(target - pred)
    weight = torch.exp(-enhancement_factor * abs_error / transition_threshold)
    loss = weight * abs_error + (1.0 - weight) * (abs_error ** 2)
    return torch.mean(loss, dim=0)


def phase_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.4,
    scale_factor: float = 1.0 / 30.0,
) -> torch.Tensor:
    """
    Periodic cosine-based loss for [0, 1] normalized phase predictions.

    Maps [0, 1] -> [0, 2*pi] radians, then computes:
        L = scale_factor * (1 - cos(delta) + alpha * |sin(delta)|)

    The cosine term captures periodicity; the sine term enhances gradients
    near the wrap-around boundary at 0/1.

    Args:
        pred:         Normalized phase prediction [0, 1].
        target:       Normalized phase ground truth [0, 1].
        alpha:        Sine-term weight for gradient enhancement.
        scale_factor: Scales the loss range to balance with amplitude loss.

    Returns:
        Mean loss over the batch.
    """
    pred_rad = pred * 2.0 * torch.pi
    target_rad = target * 2.0 * torch.pi
    delta = pred_rad - target_rad
    loss_cos = 1.0 - torch.cos(delta)
    loss_sin = torch.abs(torch.sin(delta))
    elementwise = (loss_cos + alpha * loss_sin) * scale_factor
    return torch.mean(elementwise, dim=0)


def vector_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes amplitude and phase losses and returns them as a 2-element vector.
    Used by CustomLossMetric for per-task loss tracking and accumulation.

    Args:
        pred:   Model output, shape (N, 2), where [:, 0] = amplitude, [:, 1] = phase.
        target: Ground truth, shape (N, 2).

    Returns:
        Tensor of shape (2,) — [loss_amplitude, loss_phase].
    """
    loss_amp = amplitude_loss(
        pred[:, 0].reshape(-1, 1), target[:, 0].reshape(-1, 1)
    )
    loss_phase = phase_loss(
        pred[:, 1].reshape(-1, 1), target[:, 1].reshape(-1, 1)
    )
    return torch.cat([loss_amp, loss_phase], dim=0)


# =============================================================================
# PART 1 — iTRANSFORMER BACKBONE
# =============================================================================
# NOTE: This module requires the `i-transformer` pip package.
# Install with: pip install i-transformer

try:
    from iTransformer import iTransformer
except ImportError:
    raise ImportError(
        "i-transformer package not found. Install with: pip install i-transformer"
    )


# =============================================================================
# PART 2 — MODEL ARCHITECTURE
# =============================================================================

class PlasmaResponseNet(nn.Module):
    """
    Multi-input multi-output network for plasma response prediction.

    Accepts RMP coil signals and physical observables, predicts amplitude and phase.

    Architecture:
      1. CNN branches extract spatial features from RMP and saddle coil signals.
      2. LSTM branches encode temporal dynamics from response and drift channels.
      3. iTransformer backbone models cross-channel dependencies.
      4. Dual task heads independently predict amplitude and phase.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        seq_len = config["sequence_length"]
        use_drift = config.get("use_drift_feature", False)

        # -------------------------------------------------------------------------
        # Stage 1: CNN feature extraction for RMP signals
        # -------------------------------------------------------------------------
        self.rmp_upper_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=(1, 4), stride=(1, 4)),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        self.rmp_lower_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=(1, 4), stride=(1, 4)),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )

        # Saddle coil signals: shape (N,1,8,6)
        self.saddle_branch = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(48),
            nn.ELU(),
            nn.Conv2d(48, 96, kernel_size=(1, 3), stride=(1, 3)),
            nn.BatchNorm2d(96),
            nn.ELU(),
        )

        # -------------------------------------------------------------------------
        # Stage 2: LSTM temporal encoding for physical observables
        # -------------------------------------------------------------------------
        self.response_branch = nn.LSTM(
            input_size=2,
            hidden_size=18,
            num_layers=2,
            batch_first=True,
        )

        if use_drift:
            self.drift_branch = nn.LSTM(
                input_size=1,
                hidden_size=14,
                num_layers=2,
                batch_first=True,
            )

        # -------------------------------------------------------------------------
        # Stage 3: iTransformer backbone
        # -------------------------------------------------------------------------
        num_variates = 128 + 96 + 18 if not use_drift else 128 + 96 + 18 + 14
        self.transformer_backbone = iTransformer(
            num_variates=num_variates,
            lookback_len=int(seq_len / 3),
            dim=128,
            depth=6,
            heads=8,
            dim_head=64,
            pred_length=1,
            num_tokens_per_variate=1,
            use_reversible_instance_norm=True,
            attn_dropout=config.get("dropout", 0.0),
            ff_dropout=config.get("dropout", 0.0),
        )

        # -------------------------------------------------------------------------
        # Stage 4: Task-specific output heads
        # -------------------------------------------------------------------------
        self.amp_head = nn.Sequential(
            nn.Linear(num_variates, int(config["units_1"])),
            nn.BatchNorm1d(1),
            self._get_activation(config["activation_1"]),
            nn.Dropout(0.0),
            nn.Linear(int(config["units_1"]), int(config["units_2"])),
            nn.BatchNorm1d(1),
            self._get_activation(config["activation_1"]),
            nn.Dropout(0.0),
            nn.Linear(int(config["units_2"]), 1),
        )
        self.phase_head = nn.Sequential(
            nn.Linear(num_variates, int(config["units_3"])),
            nn.BatchNorm1d(1),
            self._get_activation(config["activation_2"]),
            nn.Dropout(0.0),
            nn.Linear(int(config["units_3"]), int(config["units_4"])),
            nn.BatchNorm1d(1),
            self._get_activation(config["activation_2"]),
            nn.Dropout(0.0),
            nn.Linear(int(config["units_4"]), 1),
        )

    @staticmethod
    def _get_activation(name: str):
        mapping = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "selu": nn.SELU,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported activation: {name}")
        return mapping[name]()

    def forward(self, x_rmp: torch.Tensor, x_phys: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rmp:  RMP signals, shape (N,1,8,16).
            x_phys: Physical observables, shape (N,1,8,8) or (N,1,8,9) with drift.

        Returns:
            Concatenated [amplitude, phase] predictions, shape (N,2).
        """
        use_drift = hasattr(self, "drift_branch")

        # --- CNN branches for RMP signals ---
        x_up = self.rmp_upper_branch(x_rmp[:, :, :, 0:8])    # (N,64,4,1)
        x_up = torch.squeeze(x_up, dim=3)
        x_up = torch.permute(x_up, (0, 2, 1))                # (N,4,64)

        x_lo = self.rmp_lower_branch(x_rmp[:, :, :, 8:16])   # (N,64,4,1)
        x_lo = torch.squeeze(x_lo, dim=3)
        x_lo = torch.permute(x_lo, (0, 2, 1))                # (N,4,64)

        # --- Split physical observables into saddle, response, and drift channels ---
        x_drift = x_phys[:, :, :, -1:]                       # (N,1,8,1)
        x_saddle = x_phys[:, :, :, 0:6]                      # (N,1,8,6)
        x_resp = x_phys[:, :, :, 6:8]                        # (N,1,8,2)

        # --- CNN branch for saddle coils ---
        x_saddle = self.saddle_branch(x_saddle)              # (N,96,4,1)
        x_saddle = torch.squeeze(x_saddle, dim=3)
        x_saddle = torch.permute(x_saddle, (0, 2, 1))        # (N,4,96)

        # --- LSTM branches for response and drift ---
        x_resp = torch.squeeze(x_resp, dim=1)                # (N,8,2)
        x_resp, _ = self.response_branch(x_resp)             # (N,8,18)

        x_drift = torch.squeeze(x_drift, dim=1)              # (N,8,1)
        if use_drift:
            x_drift, _ = self.drift_branch(x_drift)          # (N,8,14)

        # --- Concatenate all feature branches ---
        if use_drift:
            x_phys_feat = torch.cat([x_saddle, x_resp, x_drift], dim=2)   # (N,4,128)
        else:
            x_phys_feat = torch.cat([x_saddle, x_resp], dim=2)            # (N,4,128)

        x = torch.cat([x_up, x_lo, x_phys_feat], dim=2)   # (N,4,242/256)
        x = x[:, -int(self.transformer_backbone.lookback_len):, :].clone()

        # --- Transformer backbone ---
        x = self.transformer_backbone(x)                     # Dict[int, Tensor]
        x = x[1]                                             # (N,4,128) — pred_length=1

        # --- Task heads ---
        amp = self.amp_head(x)                               # (N,1,1)
        amp = torch.squeeze(amp, dim=2)                      # (N,1)

        phase = self.phase_head(x)                           # (N,1,1)
        phase = torch.squeeze(phase, dim=2)                  # (N,1)

        return torch.cat([amp, phase], dim=1)               # (N,2)


# =============================================================================
# PART 3 — MODEL MANAGEMENT
# =============================================================================

class ModelManager:
    """
    Handles model creation, device placement, DataParallel wrapping,
    and checkpoint save/load operations.
    """

    @staticmethod
    def create(
        config: Dict[str, Any],
        device: Union[str, torch.device],
        data_parallel: bool = False,
    ) -> nn.Module:
        if isinstance(device, str):
            device = torch.device(device)
        model = PlasmaResponseNet(config).to(device)
        if data_parallel:
            if device.type == "cpu":
                raise ValueError("DataParallel requires GPU")
            model = nn.DataParallel(model)
        return model

    @staticmethod
    def get_module(model: nn.Module) -> nn.Module:
        """Unwrap DataParallel if present; otherwise return model as-is."""
        return model.module if isinstance(model, nn.DataParallel) else model

    @staticmethod
    def save(model: nn.Module, path: str) -> None:
        """Save model state dict to a .pth file."""
        torch.save(
            {
                "model_state_dict": ModelManager.get_module(model).state_dict(),
                "metadata": {},
            },
            path,
        )

    @staticmethod
    def load(
        model: nn.Module,
        path: str,
        map_location: Union[str, torch.device, None] = None,
    ) -> nn.Module:
        """Load saved weights into model, handling DataParallel key mismatches."""
        loaded = torch.load(path, map_location=map_location)
        state_dict = loaded["model_state_dict"]
        is_dp = isinstance(model, nn.DataParallel)
        converted = {
            (f"module.{k}" if is_dp and not k.startswith("module.") else k.replace("module.", "")): v
            for k, v in state_dict.items()
        }
        ModelManager.get_module(model).load_state_dict(converted, strict=False)
        return model


# =============================================================================
# PART 4 — LOSS METRICS & MULTI-TASK LEARNING
# =============================================================================

class CustomLossMetric(torchmetrics.Metric):
    """
    Accumulates per-batch vector losses [amp_loss, phase_loss] and returns
    per-task mean losses over the full epoch.
    """

    def __init__(self):
        super().__init__()
        self.losses: List[torch.Tensor] = []

    def update(self, loss: torch.Tensor) -> None:
        self.losses.append(loss)

    def compute(self) -> torch.Tensor:
        return torch.stack(self.losses, dim=0).mean(dim=0)

    def reset(self) -> None:
        self.losses = []


class GradNorm(nn.Module):
    """
    GradNorm: dynamically balances multi-task loss weights based on relative
    training rates (Chen et al., ICLR 2018).

    Warm-up phase: uses fixed initial weights.
    After warm-up: computes per-task gradient norms and updates loss weights
    via a separate optimizer, projecting onto the probability simplex.
    """

    def __init__(
        self,
        n_tasks: int,
        alpha: float,
        loss_weights: List[float],
        warm_up_batches: int,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.warm_up_batches = warm_up_batches
        self.loss_weights = nn.Parameter(
            torch.tensor(loss_weights, dtype=torch.float32), requires_grad=True
        )
        self.initial_losses: List[Optional[torch.Tensor]] = [None] * n_tasks
        self.batch_count = 0

    def forward(
        self,
        losses: Tuple[torch.Tensor, ...],
        shared_params: List[nn.Parameter],
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> torch.Tensor:
        self.batch_count += 1

        # --- Warm-up phase: accumulate initial losses ---
        if self.batch_count < self.warm_up_batches:
            for i in range(self.n_tasks):
                if self.initial_losses[i] is None:
                    self.initial_losses[i] = losses[i].detach()
                else:
                    self.initial_losses[i] = self.initial_losses[i] + losses[i].detach()
            if self.batch_count == self.warm_up_batches:
                for i in range(self.n_tasks):
                    self.initial_losses[i] /= float(self.warm_up_batches)
            weighted = [self.loss_weights[i] * losses[i] for i in range(self.n_tasks)]
            return sum(weighted)

        # --- GradNorm phase: update weights based on gradient norms ---
        if optimizer is None:
            raise ValueError("Optimizer for loss_weights is required")

        grad_norm_list: List[torch.Tensor] = []
        for i in range(self.n_tasks):
            for p in shared_params:
                if p.grad is not None:
                    p.grad.zero_()
            task_loss = self.loss_weights[i] * losses[i]
            grads = torch.autograd.grad(task_loss, shared_params, create_graph=True)
            grads_concat = torch.cat([g.view(-1) for g in grads if g is not None])
            this_norm = (
                torch.norm(grads_concat, p=2)
                if grads_concat.numel()
                else torch.tensor(0.0, device=losses[0].device, requires_grad=True)
            )
            grad_norm_list.append(this_norm)
            for p in shared_params:
                if p.grad is not None:
                    p.grad.zero_()

        avg_grad_norm = sum(grad_norm_list) / self.n_tasks

        loss_ratios: List[torch.Tensor] = []
        for i in range(self.n_tasks):
            curr = losses[i]
            if not curr.requires_grad:
                curr = curr.clone().detach().requires_grad_(True)
            init = self.initial_losses[i]
            if not isinstance(init, torch.Tensor):
                init = torch.tensor(init, device=curr.device)
            loss_ratios.append(curr / init)

        sum_ratio = sum(loss_ratios)
        if sum_ratio.item() < 1e-12:
            inverse_rates = [
                torch.ones_like(loss_ratios[0]) / self.n_tasks
                for _ in range(self.n_tasks)
            ]
        else:
            inverse_rates = [r / sum_ratio for r in loss_ratios]

        target_norms = [
            avg_grad_norm * torch.pow(r, self.alpha) for r in inverse_rates
        ]

        gradnorm_loss = sum(
            torch.abs(grad_norm_list[i] - target_norms[i])
            for i in range(self.n_tasks)
        )

        if not gradnorm_loss.requires_grad:
            gradnorm_loss = gradnorm_loss.clone().detach().requires_grad_(True)

        optimizer.zero_grad()
        gradnorm_loss.backward(retain_graph=True)
        optimizer.step()

        # Project onto valid weight simplex (non-negative, sums to 1)
        with torch.no_grad():
            self.loss_weights.data = torch.relu(self.loss_weights.data)
            w_sum = self.loss_weights.data.sum()
            if w_sum > 1e-12:
                self.loss_weights.data /= w_sum

        final_weights = self.loss_weights.detach()
        weighted = [final_weights[i] * losses[i] for i in range(self.n_tasks)]
        return sum(weighted)


# =============================================================================
# PART 5 — DYNAMIC REGULARIZATION SCHEDULING
# =============================================================================

class MetricScheduler:
    """
    Adjusts regularization strength (L2 weight decay, dropout) using a cubic schedule:

      Phase 1 (num_update/2): decays max_val -> min_val with cubic easing.
      Phase 2 (num_update/2): recovers min_val -> max_val with inverse cubic.

    Triggers only when val_metric exceeds threshold for patience consecutive epochs,
    with a cooldown period between updates.
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
        self.best = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.update_count = 0

    def step(self, metric: float, update_fn) -> None:
        if metric <= self.threshold:
            self.num_bad_epochs = 0
            return
        improved = (metric < self.best) if self.mode == "min" else (metric > self.best)
        if improved:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._update(update_fn)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _update(self, update_fn) -> None:
        self.update_count += 1
        if self.update_count > self.num_update:
            new_val = self.max_val
        else:
            half = self.num_update // 2
            if self.update_count <= half:
                t = self.update_count / half
                new_val = self.max_val - (self.max_val - self.min_val) * (t ** 3)
            else:
                t = (self.update_count - half) / (self.num_update - half)
                new_val = self.min_val + (self.max_val - self.min_val) * (
                    1.0 - (1.0 - t) ** 3
                )
        new_val = max(self.min_val, min(new_val, self.max_val))
        update_fn(new_val)
        if self.verbose:
            print(f"[MetricScheduler] Update #{self.update_count}, Value: {new_val:.4f}")


# =============================================================================
# PART 6 — TRAINING MANAGER (LR + L2 + DROPOUT SCHEDULING)
# =============================================================================

class TrainManager:
    """
    Centralizes optimizer setup and per-epoch dynamic updates:

      Learning rate:  warmup (cubic ramp-up) then ReduceLROnPlateau.
      L2 regularization: cubic schedule per layer group.
      Dropout probability: cubic schedule per task head.

    Each component can be independently enabled/disabled.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        warmup_epochs: int = 10,
        lr_input: Tuple[float, float] = (1e-4, 1e-4),
        lr_common: float = 5e-4,
        lr_output: Tuple[float, float] = (1e-3, 1e-3),
        l2_common: float = 1e-3,
        l2_output: Tuple[float, float] = (1e-3, 1e-3),
        dp_output: Tuple[float, float] = (0.5, 0.5),
        enable_schedule: Tuple[bool, bool, bool] = (True, True, True),
    ):
        self.config = config
        self.model = model
        self.warmup_epochs = warmup_epochs
        enable_lr, enable_l2, enable_dp = enable_schedule

        # Configure L2 and dropout values based on scheduling
        if enable_l2:
            common_l2, out1_l2, out2_l2 = 1e-2, l2_output[0], 1e-1
        else:
            common_l2, out1_l2, out2_l2 = l2_common, l2_output[0], l2_output[1]

        if enable_dp:
            dp_amp, dp_phase = dp_output[0], 0.8
        else:
            dp_amp, dp_phase = dp_output[0], dp_output[1]

        self.optimizer = getattr(torch.optim, config["optimizer_name"])([
            {"params": model.rmp_upper_branch.parameters(), "lr": lr_input[0], "weight_decay": 0},
            {"params": model.rmp_lower_branch.parameters(), "lr": lr_input[0], "weight_decay": 0},
            {"params": model.saddle_branch.parameters(),      "lr": lr_input[1], "weight_decay": 0},
            {"params": model.response_branch.parameters(),    "lr": lr_input[1], "weight_decay": 0},
            {
                "params": model.drift_branch.parameters()
                if hasattr(model, "drift_branch") else [],
                "lr": lr_input[1],
                "weight_decay": 0,
            },
            {
                "params": model.transformer_backbone.parameters(),
                "lr": lr_common,
                "weight_decay": common_l2,
            },
            {"params": model.amp_head.parameters(),  "lr": lr_output[0], "weight_decay": out1_l2},
            {"params": model.phase_head.parameters(), "lr": lr_output[1], "weight_decay": out2_l2},
        ])

        # Apply initial dropout rates to task heads
        for layer in model.amp_head.modules():
            if isinstance(layer, nn.Dropout):
                layer.p = dp_amp
        for layer in model.phase_head.modules():
            if isinstance(layer, nn.Dropout):
                layer.p = dp_phase

        # Build schedulers
        if not (enable_lr or enable_l2 or enable_dp):
            self.scheduler_warmup = None
            self.scheduler_plateau = None
            self.l2_sched_common = None
            self.l2_sched_amp = None
            self.l2_sched_phase = None
            self.dp_sched_amp = None
            self.dp_sched_phase = None
            return

        if enable_lr:
            self.scheduler_warmup = LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: max(
                    min(1.0 - (1.0 - (epoch + 1) / warmup_epochs) ** 3, 1.0), 0.01
                ),
            )
            self.scheduler_plateau = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.8,
                patience=3,
                min_lr=1e-5,
                verbose=True,
            )
        else:
            self.scheduler_warmup = self.scheduler_plateau = None

        mk_sched = lambda: MetricScheduler(
            mode="min",
            threshold=0.0001,
            num_update=int(warmup_epochs),
            patience=1,
            cooldown=0,
            min_val=0,
            verbose=True,
        )

        self.l2_sched_common = mk_sched() if enable_l2 else None
        self.l2_sched_amp    = mk_sched() if enable_l2 else None
        self.l2_sched_phase  = mk_sched() if enable_l2 else None
        self.dp_sched_amp   = mk_sched() if enable_dp else None
        self.dp_sched_phase  = mk_sched() if enable_dp else None

    def update_epoch(
        self,
        epoch: int,
        train_loss: Tuple[float, float],
        val_loss: Tuple[float, float],
    ) -> None:
        """Update learning rate, L2, and dropout after each epoch."""
        enable_lr = self.scheduler_warmup is not None
        if enable_lr:
            if epoch < self.warmup_epochs:
                self.scheduler_warmup.step()
            else:
                scalar = (val_loss[0] + val_loss[1]) / 2.0
                self.scheduler_plateau.step(scalar)

        scalar_train = (train_loss[0] + train_loss[1]) / 2.0
        scalar_val   = (val_loss[0] + val_loss[1]) / 2.0
        diff = scalar_val - scalar_train

        # --- L2 weight decay scheduling ---
        def update_l2(sched: Optional[MetricScheduler], params, loss_diff: float) -> None:
            if sched is None:
                return
            sched.step(loss_diff, lambda val: [
                g.update({"weight_decay": val})
                for g in self.optimizer.param_groups
                if any(p in params for p in g["params"])
            ])

        if self.l2_sched_common is not None:
            update_l2(
                self.l2_sched_common,
                set(self.model.transformer_backbone.parameters()),
                diff,
            )
        if self.l2_sched_amp is not None:
            update_l2(
                self.l2_sched_amp,
                set(self.model.amp_head.parameters()),
                val_loss[0] - train_loss[0],
            )
        if self.l2_sched_phase is not None:
            update_l2(
                self.l2_sched_phase,
                set(self.model.phase_head.parameters()),
                val_loss[1] - train_loss[1],
            )

        # --- Dropout probability scheduling ---
        def update_dp(sched: Optional[MetricScheduler], layers, loss_diff: float) -> None:
            if sched is None:
                return
            sched.step(loss_diff, lambda val: [
                setattr(l, "p", val) for l in layers if isinstance(l, nn.Dropout)
            ])

        if self.dp_sched_amp is not None:
            update_dp(
                self.dp_sched_amp,
                list(self.model.amp_head.modules()),
                val_loss[0] - train_loss[0],
            )
        if self.dp_sched_phase is not None:
            update_dp(
                self.dp_sched_phase,
                list(self.model.phase_head.modules()),
                val_loss[1] - train_loss[1],
            )


# =============================================================================
# PART 7 — EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Patience-based early stopping with best model checkpointing.

    Tracks validation loss; when no improvement exceeds `delta` for `patience`
    consecutive epochs, training halts. The best model state is saved to disk.
    """

    def __init__(
        self,
        patience: int = 10,
        delta: float = 0.0,
        save_path: str = "best_model.pth",
        min_epochs: int = 5,
        enable_early_stop: bool = True,
    ):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.min_epochs = min_epochs
        self.enable_early_stop = enable_early_stop
        self.best_score: Optional[float] = None
        self.counter = 0
        self.stopped = False

    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> None:
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self._save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0
        else:
            if epoch >= self.min_epochs:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    if self.enable_early_stop:
                        self.stopped = True
                    else:
                        print("Early stopping triggered but disabled — continuing...")

    def _save_checkpoint(
        self,
        val_loss: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> None:
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
            print(f"Old checkpoint removed: {self.save_path}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            self.save_path,
        )
        print(f"Validation loss improved. Best model saved at epoch {epoch}")


# =============================================================================
# PART 8 — UTILITY FUNCTIONS
# =============================================================================

def set_requires_grad_(params, requires_grad: bool = True) -> None:
    """Set requires_grad for a parameter or group of parameters."""
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


def freeze_params(*args) -> None:
    """Disable gradient computation for the given parameters."""
    set_requires_grad_(args, requires_grad=False)


def unfreeze_params(*args) -> None:
    """Enable gradient computation for the given parameters."""
    set_requires_grad_(args, requires_grad=True)


def save_to_json(data: Any, path: str) -> None:
    """Serialize data to a JSON file (handles tensors, numpy arrays, etc.)."""
    def convert(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        if isinstance(obj, set):
            return [convert(v) for v in obj]
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(path, "w") as f:
        json.dump(convert(data), f, indent=4)
    print(f"Saved to {path}")


def load_from_json(path: str) -> Any:
    """Load data from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


class InterruptManager:
    """
    Graceful Ctrl+C handling during Ray Tune runs.
    Intercepts SIGINT and raises KeyboardInterrupt so that Ray can collect
    the best result seen so far before shutting down.
    """

    def __init__(self, enable_interrupt: bool = True):
        self.enable_interrupt = enable_interrupt
        self.interrupted = False
        self._prev_handler: Any = None

    def _signal_handler(self, sig, frame) -> None:
        self.interrupted = True
        print("\nCtrl+C detected. Collecting best result so far...", flush=True)
        raise KeyboardInterrupt

    def _setup(self) -> None:
        if not self.enable_interrupt:
            return
        try:
            self._prev_handler = signal.signal(signal.SIGINT, self._signal_handler)
            os.environ.setdefault("TUNE_DISABLE_SIGINT_HANDLER", "1")
        except Exception as e:
            print(f"Warning: signal handler setup failed: {e}")

    def restore(self) -> None:
        if self._prev_handler is not None:
            try:
                signal.signal(signal.SIGINT, self._prev_handler)
            except Exception:
                pass


# =============================================================================
# PART 9 — TRAINING LOOP
# =============================================================================
# NOTE: The training loop expects DataLoaders created from a compatible Dataset.
#       The Dataset should yield tuples of (x_rmp, x_phys, y) per sample:
#         x_rmp  (batch, seq_len, 16)  — RMP coil signals
#         x_phys (batch, seq_len, 8/9) — physical observables
#         y      (batch, 2)            — [amplitude, phase] labels
#
# Example Dataset interface:
#     class YourDataset(Dataset):
#         def __getitem__(self, idx):
#             x_rmp, x_phys, y = ...  # load from your data source
#             return x_rmp, x_phys, y


def _get_shared_params(model: nn.Module) -> List[nn.Parameter]:
    """Extract shared (non-task-specific) parameters from the model."""
    m = ModelManager.get_module(model)
    params = (
        list(m.rmp_upper_branch.parameters())
        + list(m.rmp_lower_branch.parameters())
        + list(m.saddle_branch.parameters())
        + list(m.response_branch.parameters())
        + list(m.transformer_backbone.parameters())
    )
    if hasattr(m, "drift_branch"):
        params += list(m.drift_branch.parameters())
    return params


def train_model(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_metric: CustomLossMetric,
    device: Union[str, torch.device],
    mtl_strategy: Tuple[bool, bool] = (False, False),
    gradnorm_worker: Optional[GradNorm] = None,
    optimizer_gradnorm: Optional[torch.optim.Optimizer] = None,
) -> np.ndarray:
    """
    Single training epoch. Supports three training strategies:

      (False, False) — joint training with equal loss weights.
      (True,  False) — alternate freezing: update amp head, phase head, then shared.
      (False, True)  — GradNorm: dynamic loss weighting per Chen et al.
      (True,  True)  — alternate freezing + GradNorm for the shared block.

    Args:
        dataloader:       Training DataLoader yielding (x_rmp, x_phys, y) tuples.
        model:            The model to train.
        optimizer:        Main optimizer for model parameters.
        train_metric:     Metric for accumulating per-batch losses.
        device:           Target device for tensors.
        mtl_strategy:     (enable_alternate_freeze, enable_gradnorm).
        gradnorm_worker:  GradNorm instance (required if enable_gradnorm=True).
        optimizer_gradnorm: Optimizer for GradNorm weight parameters.

    Returns:
        Per-task mean losses [amp_loss, phase_loss] as a numpy array.
    """
    size = len(dataloader.dataset)
    freeze_on, gradnorm_on = mtl_strategy
    model.train()
    print(f"Total batches: {len(dataloader)}")
    print(f"MTL flags — freeze: {freeze_on}, GradNorm: {gradnorm_on}")

    shared_params = _get_shared_params(model)
    amp_params  = list(ModelManager.get_module(model).amp_head.parameters())
    phase_params = list(ModelManager.get_module(model).phase_head.parameters())

    for batch_idx, (x_rmp, x_phys, y) in enumerate(dataloader):
        x_rmp = x_rmp.to(device).unsqueeze(1)
        x_phys = x_phys.to(device).unsqueeze(1)
        y = y.to(device)

        with torch.backends.cudnn.flags(enabled=not gradnorm_on):
            pred = model(x_rmp, x_phys)
        loss_v = vector_loss(pred, y)
        loss_amp, loss_phase = loss_v[0], loss_v[1]
        train_metric.update(loss_v.detach().to(device))

        # --- Strategy dispatch ---
        if not freeze_on and not gradnorm_on:
            # Joint training with equal weights
            w_loss = (loss_amp + loss_phase) / 2.0
            optimizer.zero_grad()
            w_loss.backward()
            optimizer.step()

        elif freeze_on and not gradnorm_on:
            # Alternate freezing strategy
            # Phase 1: freeze shared + phase, update amplitude head
            freeze_params(shared_params, phase_params)
            unfreeze_params(amp_params)
            optimizer.zero_grad()
            loss_amp.backward(retain_graph=True)
            optimizer.step()

            # Phase 2: freeze shared + amplitude, update phase head
            unfreeze_params(shared_params, amp_params)
            with torch.backends.cudnn.flags(enabled=False):
                pred = model(x_rmp, x_phys)
            loss_phase = vector_loss(pred, y)[1]
            freeze_params(shared_params, amp_params)
            unfreeze_params(phase_params)
            optimizer.zero_grad()
            loss_phase.backward(retain_graph=True)
            optimizer.step()

            # Phase 3: freeze heads, update shared block
            unfreeze_params(shared_params, amp_params, phase_params)
            with torch.backends.cudnn.flags(enabled=False):
                pred = model(x_rmp, x_phys)
            loss_v = vector_loss(pred, y)
            loss_amp, loss_phase = loss_v[0], loss_v[1]
            w_loss = (loss_amp + loss_phase) / 2.0
            freeze_params(amp_params, phase_params)
            unfreeze_params(shared_params)
            optimizer.zero_grad()
            w_loss.backward()
            optimizer.step()
            unfreeze_params(shared_params, amp_params, phase_params)

        elif not freeze_on and gradnorm_on:
            # Joint training with GradNorm
            w_loss = gradnorm_worker(loss_v, shared_params, optimizer_gradnorm)
            optimizer.zero_grad()
            w_loss.backward()
            optimizer.step()

        else:
            # Alternate freezing + GradNorm
            freeze_params(shared_params, phase_params)
            unfreeze_params(amp_params)
            optimizer.zero_grad()
            loss_amp.backward()
            optimizer.step()

            unfreeze_params(shared_params, amp_params)
            with torch.backends.cudnn.flags(enabled=False):
                pred = model(x_rmp, x_phys)
            loss_phase = vector_loss(pred, y)[1]
            freeze_params(shared_params, amp_params)
            unfreeze_params(phase_params)
            optimizer.zero_grad()
            loss_phase.backward()
            optimizer.step()

            unfreeze_params(shared_params, amp_params, phase_params)
            with torch.backends.cudnn.flags(enabled=False):
                pred = model(x_rmp, x_phys)
            loss_v = vector_loss(pred, y)
            w_loss = gradnorm_worker(loss_v, shared_params, optimizer_gradnorm)
            freeze_params(amp_params, phase_params)
            unfreeze_params(shared_params)
            optimizer.zero_grad()
            w_loss.backward()
            optimizer.step()
            unfreeze_params(shared_params, amp_params, phase_params)

        if batch_idx % 40 == 0:
            print(
                f"Batch {batch_idx}: AmpLoss={float(loss_amp.item()):.6f}, "
                f"PhaseLoss={float(loss_phase.item()):.6f}  "
                f"[{(batch_idx + 1) * len(x_rmp)}/{size}]"
            )

    return train_metric.compute().cpu().numpy()


def evaluate_model(
    dataloader: DataLoader,
    model: nn.Module,
    metric: CustomLossMetric,
    device: Union[str, torch.device],
) -> np.ndarray:
    """
    Single evaluation pass over a dataloader, computing per-task losses.

    Args:
        dataloader: DataLoader yielding (x_rmp, x_phys, y) tuples.
        model:      Model to evaluate.
        metric:     Metric for accumulating per-batch losses.
        device:     Target device.

    Returns:
        Per-task mean losses [amp_loss, phase_loss] as a numpy array.
    """
    model.eval()
    with torch.no_grad():
        for x_rmp, x_phys, y in dataloader:
            x_rmp = x_rmp.to(device).unsqueeze(1)
            x_phys = x_phys.to(device).unsqueeze(1)
            y = y.to(device)
            pred = model(x_rmp, x_phys)
            loss_v = vector_loss(pred, y)
            metric.update(loss_v.detach().to(device))
    return metric.compute().cpu().numpy()


def execute_training_loop(
    config: Dict[str, Any],
    max_epochs: int,
    save_mode: str = "early_stop",
    break_cont: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
) -> None:
    """
    Main training loop covering model initialization, epoch-by-epoch training
    and evaluation, dynamic scheduling, checkpointing, and loss history export.

    NOTE: This function expects `config['data']` to contain 'train_vald' and 'test'
    paths, and `load_datasets(config)` to be available in the module scope
    (e.g., imported from your data pipeline). If `break_cont` is provided,
    training resumes from that checkpoint.

    Args:
        config:      Model / optimizer hyperparameters. See file header for key fields.
        max_epochs: Maximum training epochs.
        save_mode:  'ray_early_stop' (Ray Tune reporting) | 'early_stop' (standalone).
        break_cont: Checkpoint filename to resume from, or None.
        device:     torch device string.
    """
    if isinstance(device, str):
        device = torch.device(device)

    model = ModelManager.create(config, device, config.get("data_parallel", False))

    if break_cont and os.path.exists(break_cont):
        print("Loading checkpoint...")
        ModelManager.load(model, break_cont, map_location=device)

    # Metrics for train, val, and test sets
    train_detail_metric = CustomLossMetric().to(device)
    train_loss_metric   = CustomLossMetric().to(device)
    val_loss_metric     = CustomLossMetric().to(device)
    test_loss_metric    = CustomLossMetric().to(device)

    # Training manager with dynamic scheduling
    train_mgr = TrainManager(
        config=config,
        model=ModelManager.get_module(model),
        warmup_epochs=config["warmup_epochs"],
        lr_input=(0.1 * config["lr"], 0.1 * config["lr"]),
        lr_common=0.5 * config["lr"],
        lr_output=(config["lr"], 0.5 * config["lr"]),
        l2_common=0.5 * config["l2"],
        l2_output=(0.5 * config["l2"], 5.0 * config["l2"]),
        dp_output=(0.5 * config["dropout"], 1.5 * config["dropout"]),
        enable_schedule=tuple(config["scheduling_flags"]),
    )
    optimizer = train_mgr.optimizer

    # GradNorm worker
    use_gradnorm = config["mtl_strategy"][1]
    if use_gradnorm:
        gradnorm_worker = GradNorm(
            n_tasks=2,
            alpha=1.5,
            loss_weights=[0.5, 0.5],
            warm_up_batches=100,
        )
        optimizer_gradnorm = getattr(torch.optim, config["optimizer_name"])([
            {
                "params": gradnorm_worker.loss_weights,
                "lr": 0.1 * config["lr"],
                "weight_decay": 0,
            }
        ])
    else:
        gradnorm_worker = None
        optimizer_gradnorm = None

    # Early stopping
    early_stop = EarlyStopping(
        min_epochs=config["warmup_epochs"],
        patience=10,
        delta=0.0003,
        save_path="best_model_early.pth",
        enable_early_stop=config["enable_early_stop"],
    )

    # DataLoaders
    # NOTE: load_datasets is a user-provided function that returns
    # (train_dataset, val_dataset, test_dataset) from config.
    try:
        from data import load_datasets
    except ImportError:
        raise ImportError(
            "load_datasets function not found. Please provide a data.py module "
            "with load_datasets(config) -> (train_dataset, val_dataset, test_dataset)."
        )

    batch_size = int(config["batch_size"])
    dataset_train, dataset_val, dataset_test = load_datasets(config)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val   = DataLoader(dataset_val,   batch_size=batch_size, shuffle=False)
    loader_test  = DataLoader(dataset_test,  batch_size=batch_size, shuffle=False)

    # Print sample shapes for debugging
    for x_r, x_p, y in loader_train:
        print(f"x_rmp shape: {x_r.shape}, x_phys shape: {x_p.shape}, y shape: {y.shape}")
        break

    history_loss, history_valloss = [], []

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}\n{'=' * 40}")
        t0 = time.time()

        train_details = train_model(
            loader_train,
            model,
            optimizer,
            train_detail_metric,
            device,
            tuple(config["mtl_strategy"]),
            gradnorm_worker,
            optimizer_gradnorm,
        )
        train_loss = evaluate_model(loader_train, model, train_loss_metric, device)
        val_loss   = evaluate_model(loader_val,   model, val_loss_metric,   device)
        test_loss  = evaluate_model(loader_test,  model, test_loss_metric,  device)

        scalar_train = (train_loss[0] + train_loss[1]) / 2.0
        scalar_val   = (val_loss[0]   + val_loss[1])   / 2.0
        scalar_test  = (test_loss[0]  + test_loss[1])  / 2.0

        print(
            f"Train loss: {scalar_train:.6f}, "
            f"Val loss: {scalar_val:.6f}, "
            f"Test loss: {scalar_test:.6f}"
        )
        history_loss.append(train_loss)
        history_valloss.append(val_loss)

        train_loss_metric.reset()
        val_loss_metric.reset()
        test_loss_metric.reset()

        train_mgr.update_epoch(epoch + 1, train_loss, val_loss)
        print(f"Epoch time: {time.time() - t0:.2f}s")

        if save_mode == "ray_early_stop":
            # Ray Tune: save checkpoint and report metrics
            tempdir = tempfile.mkdtemp()
            try:
                ckp = os.path.join(tempdir, "checkpoint.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "scalar_val_loss": scalar_val,
                    },
                    ckp,
                )
                import ray.train

                ray.train.report(
                    {
                        "ray_metric_loss": scalar_test,
                        "ray_train_loss": train_loss,
                        "ray_vald_loss": val_loss,
                        "epoch": epoch + 1,
                        "batch_size": config["batch_size"],
                        "optimizer": config["optimizer_name"],
                        "learning_rate": config["lr"],
                        "weight_decay": config["l2"],
                    },
                    checkpoint=ray.train.Checkpoint.from_directory(tempdir),
                )
            finally:
                shutil.rmtree(tempdir)

            early_stop(scalar_val, model, optimizer, epoch + 1)
            if early_stop.stopped:
                print("Early stopping triggered.")
                break

        elif save_mode == "early_stop":
            early_stop(scalar_val, model, optimizer, epoch + 1)
            if early_stop.stopped:
                print("Early stopping triggered.")
                break

    # Export loss history to .mat file
    history_loss    = np.array(history_loss)
    history_valloss = np.array(history_valloss)
    if history_loss.ndim == 1:
        history_loss    = history_loss.reshape(-1, 1)
        history_valloss = history_valloss.reshape(-1, 1)

    try:
        import scipy.io as sio

        mat_path = "plotdata_LossAcc.mat"
        sio.savemat(mat_path, {"loss_acc_struc": {"train": history_loss, "val": history_valloss}})
        print(f"Saved loss history to {mat_path}")
    except ImportError:
        print("scipy not available — skipping .mat export.")

    print("Training complete.")


# =============================================================================
# PART 10 — HYPERPARAMETER TUNING (Ray Tune)
# =============================================================================
# NOTE: Requires the `ray[tune]` pip package.
#   pip install ray[tune] pyarrow
#
# Config fields referenced by this module:
#   sequence_length, norm_mode, dimen_input, dimen_output, use_drift_feature,
#   warmup_epochs, scheduling_flags, enable_early_stop, data_parallel,
#   epochs_retrain (max_epochs per trial), tuning.ray_samples (num_samples).
# =============================================================================

def get_top_k_models(
    result, metric: str, mode: str, k: int = 5
) -> List[Dict[str, Any]]:
    """Return top-k trials from a Ray Tune experiment sorted by metric."""
    df = result.results_df.sort_values(by=metric, ascending=(mode == "min"))
    top_k = df.head(k)
    top_models = []
    for _, row in top_k.iterrows():
        trial = next(
            (t for t in result.trials if t.trial_id == row.name), None
        )
        if trial is None:
            continue
        ckpt = result.get_best_checkpoint(trial, metric=metric, mode=mode)
        top_models.append(
            {
                "trial_id": trial.trial_id,
                "config": trial.config,
                "metric_value": trial.last_result.get(metric, None),
                "checkpoint_path": ckpt,
            }
        )
        print(f"Trial {trial.trial_id}: metric={top_models[-1]['metric_value']}")
    return top_models


def hyperparameter_tuning(
    num_samples: int,
    max_epochs: int,
    config_space: Dict[str, Any],
    local_dir: str,
    best_model_path: str,
) -> Optional[Dict[str, Any]]:
    """
    Run Ray Tune hyperparameter search with ASHA scheduler and save the best model.

    Args:
        num_samples:     Number of random configurations to try.
        max_epochs:      Max epochs per trial.
        config_space:    Ray Tune search space dict.
        local_dir:       Directory for Ray Tune results.
        best_model_path: Where to save the best model weights.

    Returns:
        Best configuration dict, or None if interrupted.
    """
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.report import CLIReporter

    ray.init(ignore_reinit_error=True)

    scheduler = ASHAScheduler(
        max_t=max_epochs,
        grace_period=50,
        reduction_factor=4,
    )
    reporter = CLIReporter(
        metric_columns=[
            "ray_metric_loss",
            "ray_train_loss",
            "ray_vald_loss",
            "epoch",
            "batch_size",
            "optimizer",
            "learning_rate",
        ]
    )

    interrupt_mgr = InterruptManager(enable_interrupt=True)
    interrupt_mgr._setup()

    stopper = {"time_total_s": 12 * 3600, "training_iteration": max_epochs}
    results = None

    try:
        results = tune.run(
            tune.with_parameters(
                execute_training_loop,
                max_epochs=max_epochs,
                save_mode="ray_early_stop",
                break_cont="best_model_top20.pth",
            ),
            resources_per_trial={"cpu": 4, "gpu": 0.5},
            max_concurrent_trials=16,
            reuse_actors=True,
            config=config_space,
            metric="ray_metric_loss",
            mode="min",
            scheduler=scheduler,
            progress_reporter=reporter,
            num_samples=num_samples,
            name="tune_cnn_itransformer",
            trial_dirname_creator=lambda t: f"trial_{t.trial_id}",
            storage_path=local_dir,
            resume="AUTO",
            stop=stopper,
            fail_fast=False,
            max_failures=3,
            raise_on_failed_trial=False,
        )
    except KeyboardInterrupt:
        print("Search interrupted by user.")
    finally:
        interrupt_mgr.restore()

    if results is None:
        print("No results — exiting.")
        return None

    top_5 = get_top_k_models(results, "ray_metric_loss", "min", k=5)
    best_trial = results.get_best_trial("ray_metric_loss", "min", "last")
    print(f"Best config: {best_trial.config}")

    best_ckpt = results.get_best_checkpoint(best_trial, "ray_metric_loss", "min")
    if best_ckpt:
        with best_ckpt.as_directory() as ckpt_dir:
            model = ModelManager.create(
                best_trial.config, torch.device("cuda" if torch.cuda.is_available() else "cpu"), False
            )
            ModelManager.load(
                model, os.path.join(ckpt_dir, "checkpoint.pt"), map_location="cpu"
            )
            ModelManager.save(model, best_model_path)
            print(f"Best model saved to {best_model_path}")

    # Save best configuration
    cfg_path = os.path.join(local_dir, "best_config.json")
    save_to_json(best_trial.config, cfg_path)
    return best_trial.config


# =============================================================================
# PART 11 — ENTRY POINT
# =============================================================================

def build_config_space(
    cfg: Dict[str, Any],
    num_samples: int,
    max_epochs: int,
    local_dir: str,
    best_model_path: str,
) -> Optional[Dict[str, Any]]:
    """
    Entry point for hyperparameter tuning.

    Builds a Ray Tune search space from fixed config values and variable
    search ranges, then calls hyperparameter_tuning().

    To use: import and call build_config_space(your_config_dict).
    """
    config_space = {
        "units_1": [64, 128, 256],
        "units_2": [64, 128, 256],
        "units_3": [64, 128, 256],
        "units_4": [64, 128, 256],
        "activation_1": ["relu", "tanh", "sigmoid", "selu"],
        "activation_2": ["relu", "tanh", "sigmoid", "selu"],
        "batch_size": [16, 32, 64],
        "optimizer_name": ["Adam", "AdamW"],
        "lr": (1e-4, 1e-2),
        "l2": (1e-5, 1e-2),
        "dropout": (0.0, 0.2),
    }

    from ray import tune

    config_space_ray = {
        "units_1": tune.choice(config_space["units_1"]),
        "units_2": tune.choice(config_space["units_2"]),
        "units_3": tune.choice(config_space["units_3"]),
        "units_4": tune.choice(config_space["units_4"]),
        "activation_1": tune.choice(config_space["activation_1"]),
        "activation_2": tune.choice(config_space["activation_2"]),
        "batch_size": tune.choice(config_space["batch_size"]),
        "optimizer_name": tune.choice(config_space["optimizer_name"]),
        "lr": tune.loguniform(config_space["lr"][0], config_space["lr"][1]),
        "l2": tune.loguniform(config_space["l2"][0], config_space["l2"][1]),
        "dropout": tune.uniform(config_space["dropout"][0], config_space["dropout"][1]),
    }

    # Merge fixed params from config
    fixed_keys = [
        "sequence_length",
        "norm_mode",
        "dimen_input",
        "dimen_output",
        "use_drift_feature",
        "warmup_epochs",
        "scheduling_flags",
        "enable_early_stop",
        "data_parallel",
    ]
    for k in fixed_keys:
        if k in cfg:
            config_space_ray[k] = cfg[k]

    best_config = hyperparameter_tuning(
        num_samples=num_samples,
        max_epochs=max_epochs,
        config_space=config_space_ray,
        local_dir=local_dir,
        best_model_path=best_model_path,
    )
    return best_config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CNN-iTransformer Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, default="tuning", choices=["tuning", "train"],
                        help="'tuning' to run Ray Tune, 'train' to run standalone")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda', 'cpu', 'mps', or None for auto")
    args = parser.parse_args()

    import yaml

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    if args.mode == "tuning":
        build_config_space(
            cfg=cfg,
            num_samples=cfg.get("tuning", {}).get("ray_samples", 100),
            max_epochs=cfg.get("epochs_retrain", 200),
            local_dir="ray_results",
            best_model_path="best_model_ray.pth",
        )
    else:
        execute_training_loop(
            config=cfg,
            max_epochs=cfg.get("epochs_retrain", 200),
            save_mode="early_stop",
            device=device,
        )
