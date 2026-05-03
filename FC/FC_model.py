"""
Plasma Response Prediction Model
================================

Author: [Jian Xu] <[jianxu@gmail.com]>
        [Dalian University of Technology]

================================

A multi-output fully-connected neural network for predicting plasma response
amplitude and phase in tokamak reactors, using Resonant Magnetic Perturbation
(RMP) control signals.

Key Features:
- Multi-output regression: amplitude (continuous) + phase (angular)
- Hyperparameter tuning via KerasTuner (RandomSearch)
- Custom loss functions: MSE for amplitude, cosine-based loss for phase
- Configurable architecture: 1-4 hidden layers with tunable units/activations
- Early stopping and model checkpointing

Input Configuration (expected from data pipeline):
    - x_train, y_train: Training features and labels (shape: [N, input_dim], [N, 2])
    - x_vald, y_vald: Validation features and labels (same shape)
    - x_pd, y_pd: Full dataset used for normalization statistics
    - dimen_input: Number of input features
    - dimen_output: Number of output targets (fixed: 2 for [amplitude, phase])

Hyperparameter Search Space:
    - num_layer: 1-4 hidden layers
    - units_N: Dense units per layer (24-196, step=8)
    - activation_N: relu / sigmoid / tanh / selu
    - dropout_N: Dropout rate per layer (0-0.5)
    - norm_mode: MinMax / Standard normalization
    - batch_size: 32-6432 (step=32)

Output:
    - Best hyperparameters saved to: tuner_bestHp.mat
    - Best model saved to: best_model_configure/
    - Training history saved to: plotdata_LossAcc.mat
"""

import os
import time
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner
import scipy.io as sio


# =============================================================================
# Configuration
# =============================================================================

# Resolve script directory for relative path resolution
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
tf.keras.backend.set_floatx("float64")
np.set_printoptions(threshold=np.inf)

# -------------------------------------------------------------------------
# Model Architecture Hyperparameters (Fixed)
# -------------------------------------------------------------------------
INPUT_DIM = 22           # Number of input features
OUTPUT_DIM = 2           # Number of output targets: [amplitude, phase]
WEIGHT_INIT_STDDEV = 0.05  # Standard deviation for weight initialization
L2_REG_STRENGTH = 1e-4   # L2 regularization strength

# -------------------------------------------------------------------------
# Training Hyperparameters
# -------------------------------------------------------------------------
TUNER_EPOCHS = 200        # Max epochs per trial during hyperparameter search
RETRAIN_EPOCHS = 200      # Max epochs for final model retraining
MAX_TRIALS = 4000         # Number of random search trials
EARLY_STOP_PATIENCE = 5  # Epochs to wait before early stopping
ARCH_DPI = 120           # DPI for saved model architecture plot

# -------------------------------------------------------------------------
# Loss Function Weights
# -------------------------------------------------------------------------
# Combined loss: amplitude_loss + mix_weight * phase_loss
MIX_LOSS_WEIGHT = 1.0    # Weight for phase loss term
REL_EPSILON = 1e-6       # Small constant to avoid division by zero

# -------------------------------------------------------------------------
# Optimizer Settings
# -------------------------------------------------------------------------
LEARNING_RATE = 1e-3     # Adam learning rate
ADAM_BETA1 = 0.9         # Adam beta_1
ADAM_BETA2 = 0.999       # Adam beta_2

# -------------------------------------------------------------------------
# Data Pipeline Interface
# -------------------------------------------------------------------------
# The following variables should be provided by the data preprocessing module.
# Placeholder values are used here to document the expected interface.
# Uncomment and fill with actual data when integrating with your pipeline.

# from your_data_module import (
#     x_pdtrain, y_pdtrain,    # Training set: features [N_train, INPUT_DIM], labels [N_train, OUTPUT_DIM]
#     x_pdvald, y_pdvald,      # Validation set: features [N_vald, INPUT_DIM], labels [N_vald, OUTPUT_DIM]
#     x_pd, y_pd,              # Full dataset for normalization: [N_all, INPUT_DIM], [N_all, OUTPUT_DIM]
#     data_process,            # Function: data_process(x, y, mul, mode, x_ref, y_ref) -> (x_norm, y_norm)
# )


# =============================================================================
# Loss Functions
# =============================================================================

def amplitude_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Mean Squared Error (MSE) loss for amplitude prediction.

    Args:
        y_true: Ground truth amplitude values, shape [batch, 1]
        y_pred: Predicted amplitude values, shape [batch, 1]

    Returns:
        Scalar MSE loss
    """
    error = tf.math.square(tf.math.abs(y_true - y_pred))
    mean_error = tf.math.reduce_mean(error, axis=0)
    return tf.math.reduce_mean(mean_error, axis=0)


def phase_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Cosine-based loss for phase prediction.

    Handles the angular nature of phase data using: (1 - cos(angle_diff))^2.
    This is more suitable than MSE for periodic/angular quantities.

    Args:
        y_true: Ground truth phase values (in radians), shape [batch, 1]
        y_pred: Predicted phase values (in radians), shape [batch, 1]

    Returns:
        Scalar cosine-based loss
    """
    angle_diff = y_true - y_pred
    error = tf.math.square(1.0 - tf.cos(angle_diff))
    mean_error = tf.math.reduce_mean(error, axis=0)
    return tf.math.reduce_mean(mean_error, axis=0)


def combined_loss(y_true: tf.Tensor, y_pred: list) -> tf.Tensor:
    """
    Combined loss for multi-output regression.

    Loss = amplitude_loss + mix_weight * phase_loss

    Args:
        y_true: Ground truth labels [batch, 2], columns: [amplitude, phase]
        y_pred: List of two prediction tensors [amp_pred, phase_pred]

    Returns:
        Scalar combined loss
    """
    amp_true = tf.reshape(y_true[:, 0], (-1, 1))
    phase_true = tf.reshape(y_true[:, 1], (-1, 1))
    amp_pred = tf.reshape(y_pred[0], (-1, 1))
    phase_pred = tf.reshape(tf.concat(y_pred, axis=1)[:, 1], (-1, 1))

    amp_loss_val = amplitude_loss(amp_true, amp_pred)
    phase_loss_val = phase_loss(phase_true, phase_pred)

    return amp_loss_val + MIX_LOSS_WEIGHT * phase_loss_val


def loss_vector(y_true: tf.Tensor, y_pred: list) -> tf.Tensor:
    """
    Vector-form loss for logging and visualization.

    Returns losses as a vector [amplitude_loss, phase_loss] instead of
    the scalar combined loss. Useful for monitoring individual component
    losses during training.

    Args:
        y_true: Ground truth labels [batch, 2]
        y_pred: List of prediction tensors

    Returns:
        Loss vector [amplitude_loss, phase_loss], shape [1, 2]
    """
    amp_true = tf.reshape(y_true[:, 0], (-1, 1))
    phase_true = tf.reshape(y_true[:, 1], (-1, 1))
    amp_pred = tf.reshape(y_pred[0], (-1, 1))
    phase_pred = tf.reshape(tf.concat(y_pred, axis=1)[:, 1], (-1, 1))

    amp_loss_val = amplitude_loss(amp_true, amp_pred)
    phase_loss_val = phase_loss(phase_true, phase_pred)

    return tf.convert_to_tensor([amp_loss_val, phase_loss_val])


# =============================================================================
# Model Definition
# =============================================================================

class MLPHyperModel(keras_tuner.HyperModel):
    """
    HyperModel for multi-output MLP with tunable architecture.

    Architecture:
        Input
          |
        Hidden Layers (1-4 configurable Dense layers)
          |
        ├──-> Amplitude Branch (Dense -> Dropout -> Dense[linear])
        |
        └──-> Phase Branch (Dense -> Dropout -> Dense[linear])

    Each hidden layer supports independent tuning of:
        - Number of units
        - Activation function
        - Dropout rate
    """

    def build(self, hp: keras_tuner.HyperParameters) -> keras.Model:
        """
        Build the MLP model with hyperparameter choices.

        Args:
            hp: Hyperparameter search space

        Returns:
            Compiled Keras Model
        """
        inputs = keras.Input(shape=(INPUT_DIM,), name="input_features")

        # Shared hidden layers
        x = inputs
        num_layers = hp.Int("num_layers", min_value=1, max_value=4)
        for i in range(num_layers):
            units = hp.Int(f"units_layer_{i}", min_value=24, max_value=196, step=8)
            activation = hp.Choice(f"activation_layer_{i}", ["relu", "sigmoid", "tanh", "selu"])
            dropout_rate = hp.Float(f"dropout_layer_{i}", min_value=0.0, max_value=0.5, step=0.1)

            x = keras.layers.Dense(
                units=units,
                use_bias=True,
                activation=activation,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=WEIGHT_INIT_STDDEV),
                bias_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=WEIGHT_INIT_STDDEV),
                kernel_regularizer=tf.keras.regularizers.l2(L2_REG_STRENGTH),
                name=f"shared_layer_{i}"
            )(x)
            x = keras.layers.Dropout(rate=dropout_rate, name=f"dropout_layer_{i}")(x)

        # Amplitude prediction branch
        amp_units = hp.Int("units_amp_branch", min_value=24, max_value=196, step=8)
        amp_activation = hp.Choice("activation_amp_branch", ["relu", "sigmoid", "tanh", "selu"])
        amp_dropout = hp.Float("dropout_amp_branch", min_value=0.0, max_value=0.5, step=0.1)

        amp_hidden = keras.layers.Dense(
            units=amp_units,
            use_bias=True,
            activation=amp_activation,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=WEIGHT_INIT_STDDEV),
            bias_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=WEIGHT_INIT_STDDEV),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG_STRENGTH),
            name="amp_hidden"
        )(x)
        amp_hidden = keras.layers.Dropout(rate=amp_dropout, name="dropout_amp_branch")(amp_hidden)
        amp_output = keras.layers.Dense(
            units=1,
            use_bias=True,
            activation="linear",
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=WEIGHT_INIT_STDDEV),
            bias_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=WEIGHT_INIT_STDDEV),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG_STRENGTH),
            name="amp_output"
        )(amp_hidden)

        # Phase prediction branch
        phase_units = hp.Int("units_phase_branch", min_value=24, max_value=196, step=8)
        phase_activation = hp.Choice("activation_phase_branch", ["relu", "sigmoid", "tanh", "selu"])
        phase_dropout = hp.Float("dropout_phase_branch", min_value=0.0, max_value=0.5, step=0.1)

        phase_hidden = keras.layers.Dense(
            units=phase_units,
            use_bias=True,
            activation=phase_activation,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=WEIGHT_INIT_STDDEV),
            bias_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=WEIGHT_INIT_STDDEV),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG_STRENGTH),
            name="phase_hidden"
        )(x)
        phase_hidden = keras.layers.Dropout(rate=phase_dropout, name="dropout_phase_branch")(phase_hidden)
        phase_output = keras.layers.Dense(
            units=1,
            use_bias=True,
            activation="linear",
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=WEIGHT_INIT_STDDEV),
            bias_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=WEIGHT_INIT_STDDEV),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG_STRENGTH),
            name="phase_output"
        )(phase_hidden)

        model = keras.Model(
            inputs=inputs,
            outputs=[amp_output, phase_output],
            name="plasma_response_mlp"
        )
        return model

    def fit(
        self,
        hp: keras_tuner.HyperParameters,
        model: keras.Model,
        epochs: int,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_vald: np.ndarray,
        y_vald: np.ndarray,
        callbacks: list = None,
        **kwargs
    ) -> float:
        """
        Custom training loop for the hypermodel.

        Normalizes data internally using the provided reference dataset,
        then runs a full training loop with validation monitoring.

        Args:
            hp: Hyperparameters for this trial
            model: Model instance to train
            epochs: Number of training epochs
            x_train: Training features [N_train, INPUT_DIM]
            y_train: Training labels [N_train, OUTPUT_DIM]
            x_vald: Validation features [N_vald, INPUT_DIM]
            y_vald: Validation labels [N_vald, OUTPUT_DIM]
            callbacks: Keras callbacks for epoch events

        Returns:
            Best validation loss achieved
        """
        norm_mode = hp.Choice("norm_mode", ["MinMax", "Standard"])
        batch_size = 2 * hp.Int("batch_size", min_value=32, max_value=6432, step=32)

        # Normalize data using reference statistics from full dataset
        x_train_norm, y_train_norm = data_process(
            deepcopy(x_train), deepcopy(y_train), 1.0, norm_mode, x_train, y_train
        )
        x_vald_norm, y_vald_norm = data_process(
            deepcopy(x_vald), deepcopy(y_vald), 1.0, norm_mode, x_train, y_train
        )

        # Convert to tensors
        x_train_norm = tf.cast(x_train_norm, tf.float64)
        y_train_norm = tf.cast(y_train_norm, tf.float64)
        x_vald_norm = tf.cast(x_vald_norm, tf.float64)
        y_vald_norm = tf.cast(y_vald_norm, tf.float64)

        # Create tf.data.Dataset pipelines
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train_norm, y_train_norm))
        vald_dataset = tf.data.Dataset.from_tensor_slices((x_vald_norm, y_vald_norm))

        train_dataset = train_dataset.shuffle(
            buffer_size=len(x_train_norm), reshuffle_each_iteration=True
        ).batch(batch_size)
        vald_dataset = vald_dataset.batch(batch_size)

        # Optimizer and metrics
        optimizer = keras.optimizers.Adam(
            learning_rate=LEARNING_RATE, beta_1=ADAM_BETA1, beta_2=ADAM_BETA2
        )
        train_loss_metric = keras.metrics.MeanTensor()
        vald_loss_metric = keras.metrics.MeanTensor()

        # Attach model to callbacks
        if callbacks:
            for callback in callbacks:
                callback.model = model

        # Training step using tf.function for graph optimization
        @tf.function
        def train_step(features: tf.Tensor, labels: tf.Tensor):
            with tf.GradientTape() as tape:
                predictions = model(features, training=True)
                loss = combined_loss(labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            predictions_eval = model(features)
            batch_loss = loss_vector(labels, predictions_eval)
            train_loss_metric.update_state(batch_loss)

            return batch_loss, model.trainable_variables, gradients

        # Validation step
        @tf.function
        def vald_step(features: tf.Tensor, labels: tf.Tensor):
            predictions = model(features, training=False)
            batch_loss = loss_vector(labels, predictions)
            vald_loss_metric.update_state(batch_loss)
            return batch_loss

        # Training loop
        best_vald_loss = float("inf")
        global history_loss, history_vald_loss

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            for batch_features, batch_labels in train_dataset:
                batch_loss, batch_vars, batch_grads = train_step(batch_features, batch_labels)

            # Validate
            for batch_features, batch_labels in vald_dataset:
                _ = vald_step(batch_features, batch_labels)

            # Collect epoch metrics
            epoch_train_loss = train_loss_metric.result().numpy()
            epoch_vald_loss = vald_loss_metric.result().numpy()

            # Report to tuner callbacks
            if callbacks:
                for callback in callbacks:
                    callback.on_epoch_end(
                        epoch,
                        logs={"my_metric": epoch_vald_loss[0] + MIX_LOSS_WEIGHT * epoch_vald_loss[1]}
                    )

            train_loss_metric.reset_states()
            vald_loss_metric.reset_states()

            # Track history for plotting
            history_loss.append(epoch_train_loss)
            history_vald_loss.append(epoch_vald_loss)

            # Compute scalar losses for logging
            scalar_train_loss = epoch_train_loss[0] + MIX_LOSS_WEIGHT * epoch_train_loss[1]
            scalar_vald_loss = epoch_vald_loss[0] + MIX_LOSS_WEIGHT * epoch_vald_loss[1]

            print(f"Epoch {epoch:05d} | train_loss: {scalar_train_loss:.10f} | "
                  f"vald_loss: {scalar_vald_loss:.10f} | time: {time.time() - epoch_start:.2f}s")

            # Save best model based on validation loss
            if scalar_vald_loss < best_vald_loss:
                checkpoint_path = os.path.join(_SCRIPT_DIR, "best_model_configure")
                print(f"Epoch {epoch:05d}: Improved from {best_vald_loss:.10f} to "
                      f"{scalar_vald_loss:.10f}, saving model...")
                best_vald_loss = scalar_vald_loss
                model.save(checkpoint_path, overwrite=True)

        return best_vald_loss


# =============================================================================
# Data Processing Utilities
# =============================================================================

def data_process(
    x: np.ndarray,
    y: np.ndarray,
    mul: float,
    mode: str,
    x_ref: np.ndarray,
    y_ref: np.ndarray
) -> tuple:
    """
    Normalize features and scale labels.

    Args:
        x: Features to normalize
        y: Labels to scale
        mul: Multiplier for labels (e.g., for loss weighting)
        mode: Normalization mode -- "MinMax" or "Standard"
        x_ref: Reference features for computing normalization stats
        y_ref: Reference labels (unused currently, reserved for future)

    Returns:
        Tuple of (normalized_x, scaled_y)
    """
    from sklearn import preprocessing

    if mode == "MinMax":
        scaler = preprocessing.MinMaxScaler().fit(x_ref)
        x = scaler.transform(x)
    elif mode == "Standard":
        scaler = preprocessing.StandardScaler().fit(x_ref)
        x = scaler.transform(x)

    y = y * mul
    return deepcopy(x), deepcopy(y)


# =============================================================================
# Utility Functions
# =============================================================================

def save_struct_to_mat(filepath: str, struct_name: str, *arrays) -> None:
    """
    Save multiple numpy arrays to a MATLAB .mat file under a struct.

    Args:
        filepath: Output .mat file path
        struct_name: Name of the MATLAB struct to create
        *arrays: Variable number of numpy arrays to save
    """
    import inspect

    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"Existing file removed: {filepath}")

    data_dict = {struct_name: {}}
    frame = inspect.currentframe()
    try:
        for arr in arrays:
            var_name = None
            caller_locals = frame.f_back.f_locals
            for name, value in caller_locals.items():
                if value is arr:
                    var_name = name
                    break
            if var_name is None:
                var_name = "array"
            data_dict[struct_name][var_name] = arr
    finally:
        del frame

    sio.savemat(filepath, data_dict)
    print(f"Saved to '{filepath}' under struct '{struct_name}'")


# =============================================================================
# Main Training Pipeline
# =============================================================================

def main():
    """
    Full training pipeline: hyperparameter search -> retrain -> save results.

    Data Requirements:
        x_pdtrain, y_pdtrain: Training set (features [N_train, INPUT_DIM], labels [N_train, 2])
        x_pdvald, y_pdvald: Validation set (features [N_vald, INPUT_DIM], labels [N_vald, 2])
        x_pd, y_pd: Full dataset for normalization (features [N_all, INPUT_DIM], labels [N_all, 2])
        data_process: Normalization function with signature:
            data_process(x, y, mul, mode, x_ref, y_ref) -> (x_norm, y_norm)
    """
    # -------------------------------------------------------------------------
    # Placeholder data (replace with actual data from your pipeline)
    # -------------------------------------------------------------------------
    # Uncomment the following lines and ensure your data module provides these:
    #
    # from your_data_module import x_pdtrain, y_pdtrain, x_pdvald, y_pdvald, x_pd, y_pd, data_process
    #
    # x_pdtrain, y_pdtrain  # Training features and labels
    # x_pdvald, y_pdvald    # Validation features and labels
    # x_pd, y_pd            # Reference dataset for normalization
    # data_process          # Normalization function

    print("=" * 60)
    print("PLASMA RESPONSE PREDICTION MODEL - TRAINING PIPELINE")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Initialize Hyperparameter Tuner
    # -------------------------------------------------------------------------
    tuner = keras_tuner.RandomSearch(
        hypermodel=MLPHyperModel(),
        objective=keras_tuner.Objective("my_metric", "min"),
        max_trials=MAX_TRIALS,
        directory=os.path.join(_SCRIPT_DIR, "tuner_results"),
        project_name="plasma_response_search",
        overwrite=True,
    )
    tuner.search_space_summary()

    # -------------------------------------------------------------------------
    # Run Hyperparameter Search
    # -------------------------------------------------------------------------
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="my_metric", mode="min",
            patience=EARLY_STOP_PATIENCE, verbose=1, restore_best_weights=True
        ),
        keras.callbacks.TensorBoard(os.path.join(_SCRIPT_DIR, "tb_logs", "tuner_search")),
    ]

    print("\n--- Starting Hyperparameter Search ---\n")
    tuner.search(
        epochs=TUNER_EPOCHS,
        x_pdtrain=x_pdtrain,
        y_pdtrain=y_pdtrain,
        x_pdvald=x_pdvald,
        y_pdvald=y_pdvald,
        callbacks=callbacks,
    )
    tuner.results_summary()

    # Retrieve best hyperparameters and model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: {best_hps.values}")

    best_model = tuner.get_best_models()[0]
    print("Best model summary:")
    best_model.summary()

    # -------------------------------------------------------------------------
    # Retrain with Best Hyperparameters
    # -------------------------------------------------------------------------
    print("\n--- Retraining with Best Hyperparameters ---\n")

    hypermodel = MLPHyperModel()
    best_hp = tuner.get_best_hyperparameters()[0]
    model = hypermodel.build(best_hp)

    print("Retrained model summary:")
    model.summary()

    # Save model architecture diagram
    topo_path = os.path.join(_SCRIPT_DIR, "topo_mlp.eps")
    if os.path.exists(topo_path):
        os.remove(topo_path)
    tf.keras.utils.plot_model(
        model, to_file=topo_path, show_shapes=True,
        show_layer_names=True, show_layer_activations=False,
        rankdir="TB", dpi=ARCH_DPI, expand_nested=True
    )

    callbacks_retrain = [
        keras.callbacks.EarlyStopping(
            monitor="my_metric", mode="min",
            patience=EARLY_STOP_PATIENCE, verbose=1, restore_best_weights=True
        ),
        keras.callbacks.TensorBoard(os.path.join(_SCRIPT_DIR, "tb_logs", "retrain")),
    ]

    best_vald_loss = hypermodel.fit(
        best_hp, model,
        epochs=RETRAIN_EPOCHS,
        x_train=x_pdtrain,
        y_train=y_pdtrain,
        x_vald=x_pdvald,
        y_vald=y_pdvald,
        callbacks=callbacks_retrain,
    )
    print(f"\nRetraining complete. Best validation loss: {best_vald_loss:.10f}")

    # -------------------------------------------------------------------------
    # Save Results
    # -------------------------------------------------------------------------
    history_loss = np.array(history_loss)
    history_vald_loss = np.array(history_vald_loss)

    if history_loss.ndim == 1:
        history_loss = history_loss.reshape(-1, 1)
        history_vald_loss = history_vald_loss.reshape(-1, 1)

    # Save training history
    loss_acc_path = os.path.join(_SCRIPT_DIR, "plotdata_LossAcc.mat")
    if os.path.exists(loss_acc_path):
        os.remove(loss_acc_path)
    save_struct_to_mat(loss_acc_path, "training_history", history_loss, history_vald_loss)

    # Save best hyperparameters
    hp_path = os.path.join(_SCRIPT_DIR, "tuner_bestHp.mat")
    if os.path.exists(hp_path):
        os.remove(hp_path)
    save_struct_to_mat(hp_path, "best_hyperparameters", best_hps.values)

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Best model saved to: {os.path.join(_SCRIPT_DIR, 'best_model_configure')}")
    print(f"Training history saved to: {loss_acc_path}")
    print(f"Hyperparameters saved to: {hp_path}")


if __name__ == "__main__":
    main()
