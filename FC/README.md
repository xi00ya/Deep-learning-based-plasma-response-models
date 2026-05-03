# Plasma Response Prediction Model

A multi-output fully-connected neural network (MLP) for predicting plasma response **amplitude** and **phase** in tokamak reactors, driven by Resonant Magnetic Perturbation (RMP) control signals.

**Author:** Jian Xu (jianxu@gmail.com) — Dalian University of Technology

## Overview

This project implements a configurable MLP model with two key innovations for plasma control:

1. **Multi-output regression** with dedicated loss functions for each target type:
   - **Amplitude**: MSE (Mean Squared Error) loss for continuous values
   - **Phase**: Cosine-based loss \((1 - \cos(\theta_{true} - \theta_{pred}))^2\), which properly handles the angular/periodic nature of phase data

2. **Automated hyperparameter tuning** via KerasTuner (RandomSearch), searching over architecture depth, layer width, activation functions, dropout rates, normalization modes, and batch sizes.


## Installation

```bash
pip install tensorflow keras-tuner numpy scipy scikit-learn
```

Tested with Python 3.9+, TensorFlow 2.14+, KerasTuner 1.3+.

## Quick Start

### 1. Prepare Your Data

The model expects NumPy arrays with the following shapes:

```python
# Training set
x_train  # shape: [N_train, INPUT_DIM]   -- input features
y_train  # shape: [N_train, 2]           -- labels: [amplitude, phase]

# Validation set
x_vald   # shape: [N_vald, INPUT_DIM]
y_vald   # shape: [N_vald, 2]

# Reference dataset (for normalization statistics)
x_ref    # shape: [N_all, INPUT_DIM]     -- typically train + vald combined
y_ref    # shape: [N_all, 2]
```

> **Note on `INPUT_DIM`**: Currently set to `22` in the code (line 62). Adjust this constant and the input layer shape to match your feature set.

### 2. Integrate Data

Open `model_training.py` and uncomment/replace the data import in `main()`:

```python
def main():
    from your_data_pipeline import (
        x_pdtrain as x_train, y_pdtrain as y_train,
        x_pdvald as x_vald, y_pdvald as y_vald,
        x_pd as x_ref, y_pd as y_ref,
        data_process,
    )
    # ... rest of main()
```

### 3. Run

```bash
python model_training.py
```

## Configuration

All tunable hyperparameters are defined as module-level constants in the **Configuration** section of `model_training.py`.

### Fixed Architecture Parameters

| Constant | Default | Description |
|---|---|---|
| `INPUT_DIM` | `22` | Number of input features |
| `OUTPUT_DIM` | `2` | Number of outputs (amplitude + phase) |
| `WEIGHT_INIT_STDDEV` | `0.05` | Std dev for weight initialization |
| `L2_REG_STRENGTH` | `1e-4` | L2 regularization coefficient |

### Training Parameters

| Constant | Default | Description |
|---|---|---|
| `TUNER_EPOCHS` | `35` | Max epochs per trial during hyperparameter search |
| `RETRAIN_EPOCHS` | `35` | Max epochs for final model retraining |
| `MAX_TRIALS` | `100` | Number of RandomSearch trials |
| `EARLY_STOP_PATIENCE` | `5` | Epochs to wait for improvement before early stopping |
| `LEARNING_RATE` | `1e-3` | Adam optimizer learning rate |
| `MIX_LOSS_WEIGHT` | `1.0` | Weight of phase loss in combined loss |

### Hyperparameter Search Space

During RandomSearch, the following dimensions are tuned:

| Parameter | Range | Step |
|---|---|---|
| `num_layers` | 1 - 4 | 1 |
| `units_layer_N` | 24 - 196 | 8 |
| `activation_layer_N` | relu / sigmoid / tanh / selu | -- |
| `dropout_layer_N` | 0.0 - 0.5 | 0.1 |
| `units_amp_branch` | 24 - 196 | 8 |
| `activation_amp_branch` | relu / sigmoid / tanh / selu | -- |
| `dropout_amp_branch` | 0.0 - 0.5 | 0.1 |
| `units_phase_branch` | 24 - 196 | 8 |
| `activation_phase_branch` | relu / sigmoid / tanh / selu | -- |
| `dropout_phase_branch` | 0.0 - 0.5 | 0.1 |
| `norm_mode` | MinMax / Standard | -- |
| `batch_size` | 32 - 6432 | 64 |

## Architecture

```
Input (INPUT_DIM features)
    │
    ▼
Shared Hidden Layers (1-4 Dense layers, each with: Dense + Dropout)
    │
    ├──► Amplitude Branch: Dense + Dropout + Dense(linear) ──────► amp_output
    │
    └──► Phase Branch:     Dense + Dropout + Dense(linear) ─────► phase_output
```

Each shared hidden layer and both output branches independently tune: number of units, activation function, and dropout rate.


## Output Files

After training, the following files are generated in the script directory:

| File | Description |
|---|---|
| `best_model_configure/` | Saved best model (Keras `save()` format) |
| `tuner_bestHp.mat` | Best hyperparameters (MATLAB struct) |
| `plotdata_LossAcc.mat` | Training history: `[history_loss, history_vald_loss]` |
| `topo_mlp.eps` | Model architecture diagram |
| `tuner_results/` | KerasTuner search logs |
| `tb_logs/` | TensorBoard event files |

## Extending

### Add New Output Targets

1. Define a new loss function (e.g., following the cosine pattern for another angular variable)
2. Add a new output branch in `MLPHyperModel.build()`, mirroring the amplitude/phase branch pattern
3. Update `combined_loss()` and `loss_vector()` to include the new branch
4. Increase `OUTPUT_DIM` to match total number of outputs

### Use a Different Tuner

Replace `keras_tuner.RandomSearch` with `BayesianOptimization` or `Hyperband` in `main()`:

```python
from keras_tuner import BayesianOptimization
tuner = keras_tuner.BayesianOptimization(
    hypermodel=MLPHyperModel(),
    objective=keras_tuner.Objective("my_metric", "min"),
    max_trials=MAX_TRIALS,
    directory=os.path.join(_SCRIPT_DIR, "tuner_results"),
    project_name="plasma_response_search",
)
```

### Load a Trained Model

```python
import tensorflow as tf
model = tf.keras.models.load_model("best_model_configure", compile=False)
predictions = model.predict(x_new)  # Returns [amp_preds, phase_preds]
```

## License

MIT
