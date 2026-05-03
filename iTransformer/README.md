# CNN-iTransformer: Plasma Response Prediction

**Author:** Jian Xu (jianxu@gmail.com) — Dalian University of Technology

## Overview

`cnn_itransformer.py` implements a multi-task neural network for predicting plasma response — specifically the **amplitude** and **phase** of Resonant Magnetic Perturbation (RMP) coil-induced plasma responses in a tokamak.

The model combines:

- **CNN branches** for spatial feature extraction from RMP coil signals
- **LSTM branches** for temporal encoding of plasma response and drift dynamics
- **iTransformer backbone** for cross-channel dependency modeling
- **Dual task-specific heads** for independent amplitude and phase predictions

## Architecture Pipeline

```
Input Tensors
  x_rmp  (N,1,8,16)   — RMP coil signals: upper 8 + lower 8 channels
  x_phys (N,1,8,8/9)  — Physical observables: 6 saddle coils + 2 response + (optional drift)

Stage 1 — CNN Feature Extraction
  rmp_upper_branch  Conv2d(1→32→64)  → (N,4,64)
  rmp_lower_branch  Conv2d(1→32→64)  → (N,4,64)
  saddle_branch     Conv2d(1→48→96)  → (N,4,96)

Stage 2 — LSTM Temporal Encoding
  response_branch  LSTM(2→18)  → (N,8,18)
  drift_branch     LSTM(1→14)  → (N,8,14)   [optional]

Stage 3 — Concatenation & Transformer Backbone
  Concat → (N,4,242/256)  →  iTransformer(num_variates=242/256, depth=6, heads=8)  → (N,4,128)

Stage 4 — Task Heads
  amp_head   Linear*3  → (N,1)
  phase_head Linear*3  → (N,1)

Output: concat → (N,2)  [amplitude, phase]
```

## Installation

```bash
pip install torch>=2.0
pip install i-transformer
pip install ray[tune] pyarrow   # for hyperparameter tuning
```

## Quick Start

### As a Library

```python
from cnn_itransformer import (
    PlasmaResponseNet, ModelManager, TrainManager, GradNorm,
    EarlyStopping, CustomLossMetric, MetricScheduler,
    vector_loss, amplitude_loss, phase_loss,
    hyperparameter_tuning, execute_training_loop,
)

# Create and use the model
model = ModelManager.create(config, device='cuda')
execute_training_loop(config, max_epochs=200, save_mode='early_stop', device='cuda')
```

### Standalone Training

```bash
python cnn_itransformer.py --config config.yaml --mode train --device cuda
```

### Hyperparameter Tuning (Ray Tune)

```bash
python cnn_itransformer.py --config config.yaml --mode tuning --device cuda
```

## Key Classes and Functions

### Loss Functions (Part 0)

| Function | Description |
|---|---|
| `amplitude_loss` | Enhanced MSE with smooth exponential weight — stronger gradients for small errors |
| `phase_loss` | Periodic cosine-based loss for [0, 1] normalized phase values, handles wrap-around at 0/1 |
| `vector_loss` | Combines amplitude and phase losses into a 2-element vector for per-task tracking |

### Model (Part 1-2)

| Class | Description |
|---|---|
| `PlasmaResponseNet` | Main multi-input multi-output network. Accepts RMP signals and physical observables, predicts amplitude and phase. |

### Training Utilities (Part 3-6)

| Class | Description |
|---|---|
| `ModelManager` | Model creation, device placement, DataParallel wrapping, checkpoint save/load |
| `CustomLossMetric` | Accumulates per-batch vector losses, returns per-task mean losses |
| `GradNorm` | Implements Chen et al. (ICLR 2018) GradNorm for dynamic multi-task loss weighting |
| `MetricScheduler` | Cubic schedule for L2 weight decay and dropout, triggered by validation metric degradation |
| `TrainManager` | Centralized optimizer setup with per-epoch dynamic scheduling of LR, L2, and dropout |
| `EarlyStopping` | Patience-based early stopping with best model checkpointing |

### Training Loop (Part 8-9)

| Function | Description |
|---|---|
| `train_model` | Single training epoch supporting joint, alternate-freezing, GradNorm, or combined MTL strategies |
| `evaluate_model` | Single evaluation pass returning per-task losses |
| `execute_training_loop` | Full training loop: initialization, epoch-by-epoch training/evaluation, checkpointing, loss history export |

### Hyperparameter Tuning (Part 10-11)

| Function | Description |
|---|---|
| `hyperparameter_tuning` | Ray Tune with ASHA scheduler, saves best model to disk |
| `get_top_k_models` | Returns top-k trials sorted by a given metric |
| `build_config_space` | Entry point for tuning — builds Ray Tune search space from fixed config and variable ranges |

## Configuration Fields

### Model Configuration

| Field | Type | Description |
|---|---|---|
| `sequence_length` | int | Length of input time window for sliding-window slicing |
| `use_drift_feature` | bool | Whether to include drift time as a feature channel |
| `dropout` | float | Dropout probability for task heads and transformer |
| `units_1..units_4` | int | Hidden layer sizes for amp_head and phase_head MLPs |
| `activation_1` | str | Activation for amplitude head (`relu` or `tanh` or `sigmoid` or `selu` ) |
| `activation_2` | str | Activation for phase head (`relu` or `tanh`  or `sigmoid` or `selu`) |

### Training Configuration

| Field | Type | Description |
|---|---|---|
| `lr` | float | Base learning rate |
| `l2` | float | Base L2 weight decay |
| `batch_size` | int | Mini-batch size |
| `optimizer_name` | str | Optimizer class (`Adam` or `AdamW`) |
| `warmup_epochs` | int | Warm-up epochs before ReduceLROnPlateau takes over |
| `scheduling_flags` | tuple | `(enable_lr, enable_l2, enable_dropout)` as bool tuple |
| `enable_early_stop` | bool | Whether to trigger early stopping |
| `data_parallel` | bool | Whether to wrap model with `nn.DataParallel` |
| `mtl_strategy` | tuple | `(enable_alternate_freeze, enable_gradnorm)` |
| `dimen_input` | int | Total number of input feature channels |
| `dimen_output` | int | Number of output targets (always 2: amplitude + phase) |

## Multi-Task Learning Strategies

The training loop (`train_model`) supports four strategies controlled by `mtl_strategy`:

1. **(False, False)** — Joint training with equal loss weights: `w_loss = (loss_amp + loss_phase) / 2`
2. **(True, False)** — Alternate freezing: update amplitude head, phase head, then shared block sequentially
3. **(False, True)** — GradNorm: dynamic loss weighting based on relative gradient norms
4. **(True, True)** — Alternate freezing + GradNorm applied to the shared block

## Dynamic Scheduling

`TrainManager.update_epoch` applies per-epoch adjustments after each epoch:

- **Learning rate**: Cubic warmup ramp-up, then ReduceLROnPlateau
- **L2 weight decay**: Cubic schedule (decay then recover) triggered by train-val loss divergence per layer group
- **Dropout probability**: Cubic schedule triggered by per-task train-val loss divergence

`MetricScheduler` controls L2 and dropout updates — it only triggers when the validation metric exceeds a threshold for `patience` consecutive epochs, with a `cooldown` period between updates.

## Data Format

The `execute_training_loop` expects `load_datasets(config)` to return three `Dataset` objects `(train_dataset, val_dataset, test_dataset)`. Each dataset should yield tuples `(x_rmp, x_phys, y)` where:

- `x_rmp`  — shape `(batch, seq_len, 16)`, RMP coil signals
- `x_phys` — shape `(batch, seq_len, 8)` or `(batch, seq_len, 9)` with drift
- `y`      — shape `(batch, 2)`, `[amplitude, phase]` labels

## External Dependencies

- PyTorch >= 2.0
- [iTransformer](https://github.com/yyupeter/iTransformer) (`pip install i-transformer`)
- Ray Tune (`pip install ray[tune] pyarrow`) — for hyperparameter search
