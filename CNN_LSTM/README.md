# CNN-LSTM Multi-Task Model

A PyTorch implementation of a CNN-LSTM hybrid neural network for multi-output sequence prediction. Processes multi-channel time-series inputs through parallel CNN/LSTM feature extractors, fuses them in a shared LSTM, and produces two task-specific outputs (e.g., amplitude and phase).

**Author:** Jian Xu (jianxu@gmail.com) — Dalian University of Technology

---

## Architecture

```
Input:  x_a [B, seq_len, 16] + x_b [B, seq_len, 8]
        y_true [B, 2]

  x_a  ──► CNN Branch A (Conv2d×2 → 64ch) ──┐
  x_b  ──► CNN Branch B (Conv2d×2 → 96ch) ──┤
  x_b[:,:,:2]  ──► LSTM Branch C (2→18) ────┤
  x_b[:,:,-1]  ──► LSTM Branch D (1→14) ────┤
                                                ├─► Shared LSTM (→32ch) ──►
          Output Head 1              Output Head 2
              │                           │
           [B, 1]                      [B, 1]
```

### Components

| Component | Type | Role |
|---|---|---|
| `cnn_branch_a` | `nn.Sequential` | 2-layer 2D CNN, output 64 channels |
| `cnn_branch_b` | `nn.Sequential` | 2-layer 2D CNN, output 96 channels |
| `lstm_branch_c` | `nn.LSTM` | 2-layer LSTM, input=2, hidden=18 |
| `lstm_branch_d` | `nn.LSTM` | 2-layer LSTM, input=1, hidden=14 |
| `shared_lstm` | `nn.LSTM` | 2-layer LSTM, hidden=32 |
| `head_output_1` | `nn.Sequential` | Private MLP for Output 1 |
| `head_output_2` | `nn.Sequential` | Private MLP for Output 2 |

### Losses

- **Output 1:** Enhanced MSE — smooth transition from linear (small errors) to quadratic (large errors) on [0,1] normalized values.
- **Output 2:** Cosine-based periodic loss — maps [0,1] to [0, 2π], combines cosine + sine enhancement terms.

---

## Installation

```bash
pip install torch torchmetrics "ray[tune]" numpy
```

- Python >= 3.8, PyTorch >= 1.10, torchmetrics, ray[tune], NumPy.

---

## Data Interface

Data must be normalized to **[0, 1]** before feeding to the model.

Your `DataLoader` should yield `(x_a, x_b, y_true)` tuples:

| Tensor | Shape | Description |
|---|---|---|
| `x_a` | `[batch, seq_len, 16]` | Signal group A → CNN Branch A |
| `x_b` | `[batch, seq_len, 8]` | Signal group B, internally split into CNN + two LSTM branches |
| `y_true` | `[batch, 2]` | Ground truth, col 0 → Output 1, col 1 → Output 2 |

> **Note:** Inside `forward()`, `x_b` is sliced as: `[:,:,:2]` → LSTM Branch C, `[:,:,-1:]` → LSTM Branch D, the remainder → CNN Branch B. Adjust the slicing in `forward()` if your channel layout differs.

Before running, set the signal counts in Section 12 of `model.py`:

```python
num_signals_a = 16   # width of x_a
num_signals_b = 8    # width of x_b
```

---

## Usage

### 1. Prepare Data

```python
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, ...):
        # Load and normalize your data to [0, 1]
        pass

    def __getitem__(self, idx):
        return x_a, x_b, y   # shapes: [seq_len,16], [seq_len,8], [2]

    def __len__(self):
        return len(...)

train_loader = DataLoader(MyDataset(...), batch_size=32, shuffle=True)
val_loader   = DataLoader(MyDataset(...), batch_size=32)
test_loader  = DataLoader(MyDataset(...), batch_size=32)
```

### 2. Quick Training

> **Usage example — adapt to your own dataset and config.**

```python
from model import training_loop, MultiOutputModel, ModelManager

config = {
    "dropout": 0.25,
    "units_1": 80, "units_2": 64,
    "units_3": 80, "units_4": 64,
    "activation_1": "relu",
    "activation_2": "relu",
    "optimizer_name": "Adam",
    "lr": 5e-4,
    "l2": 1e-3,
    "drift_enabled": True,
}

training_loop(
    config=config,
    max_epochs=200,
    dataloader_train=train_loader,
    dataloader_vald=val_loader,
    dataloader_test=test_loader,
    multitask_config=(True, False),   # (alternate_freezing, use_gradnorm)
    save_mode="early_stop",
)
```

### 3. Hyperparameter Tuning with Ray Tune

> **Usage example — adjust search space to your task.**

```python
from model import hyperparameter_tuning

config_space = {
    "units_1": tune.choice([64, 80, 96]),
    "units_2": tune.choice([48, 64, 80]),
    "units_3": tune.choice([64, 80, 96]),
    "units_4": tune.choice([48, 64, 80]),
    "activation_1": tune.choice(["relu", "tanh", "sigmoid", "selu"]),
    "activation_2": tune.choice(["relu", "tanh", "sigmoid", "selu"]),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "optimizer_name": tune.choice(["Adam", "AdamW"]),
    "lr": tune.loguniform(1e-4, 1e-2),
    "l2": tune.loguniform(1e-4, 5e-2),
    "dropout": tune.uniform(0.1, 0.5),
    "drift_enabled": tune.choice([True, False]),
}

best_config = hyperparameter_tuning(
    num_samples=500,
    max_epochs=100,
    config_space=config_space,
    metric="ray_metric_loss",
    mode="min",
    name="my_experiment",
    local_dir="ray_results",
    best_model_path="best_model_ray.pth",
)
```

### 4. Model Save / Load

```python
from model import MultiOutputModel, ModelManager

# Save
ModelManager.save(model, "my_model.pth")

# Load
new_model = MultiOutputModel(config)
ModelManager.load(new_model, "my_model.pth", map_location="cpu")
```

---

## Multi-Task Strategies

Control via `multitask_config = (alternate_freezing, use_gradnorm)`:

| Config | Strategy |
|---|---|
| `(False, False)` | Vanilla joint update, equal 50/50 weights |
| `(True, False)` | Alternating freeze: train Head1 → Head2 → shared layers per batch |
| `(False, True)` | GradNorm: dynamic loss weighting on shared layers |
| `(True, True)` | Alternating freeze + GradNorm combined |

GradNorm implements Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks" (ICML 2018).

---

## Key Constants (Section 12)

| Constant | Default | Description |
|---|---|---|
| `num_signals_a` | `16` | Width of CNN Branch A input |
| `num_signals_b` | `8` | Width of CNN Branch B input |
| `warmup_epochs` | `30` | Warmup epochs before LR plateau / L2 / dropout schedules activate |
| `epochs_tuning` | `200` | Max epochs per trial |
| `enable_schedule` | `(True, True, True)` | Enable LR / L2 / Dropout scheduling |
| `enable_earlystop` | `True` | Enable early stopping |
| `multitask_freeze_gradnorm` | `(True, False)` | Default multi-task strategy |
| `training_mode` | `"TA"` | `"TA"` = Ray Tune; `"BC"` = resume from checkpoint |

---

## Entry Points

Run directly:

```bash
python model.py
```

Set `training_mode = "TA"` to run Ray Tune hyperparameter search. Set `training_mode = "BC"` to resume from the best checkpoint and DataLoader placeholders in `main()`.
