"""
model.py

Creates Fully Connected Neural Networks (FCNN)
for Assignment 3.

Architectures are defined in config.py.
"""

import tensorflow as tf

from config import (
    INPUT_DIM,
    NUM_CLASSES,
    ARCHITECTURES,
    HIDDEN_ACTIVATION,
    OUTPUT_ACTIVATION,
    LOSS_FUNCTION,
)


# =============================================================================
# Build FCNN
# =============================================================================

def build_model(hidden_layers):
    """
    Build a fully connected neural network.

    Parameters
    ----------
    hidden_layers : list
        Example:
        [256,128,64]
    """

    model = tf.keras.Sequential(name="FCNN")

    # Input layer
    model.add(tf.keras.layers.Input(shape=(INPUT_DIM,)))

    # Hidden layers
    for units in hidden_layers:

        model.add(
            tf.keras.layers.Dense(
                units=units,
                activation=HIDDEN_ACTIVATION,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
            )
        )

    # Output layer
    model.add(
        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation=OUTPUT_ACTIVATION,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )
    )

    return model


# =============================================================================
# Compile Model
# =============================================================================

def compile_model(model, optimizer):
    """
    Compile model with selected optimizer.
    """

    model.compile(
        optimizer=optimizer,
        loss=LOSS_FUNCTION,
        metrics=["accuracy"],
    )

    return model


# =============================================================================
# Create Model
# =============================================================================

def create_model(architecture_name, optimizer):
    """
    Create and compile one architecture.

    Parameters
    ----------
    architecture_name : str

        Arch_3HL
        Arch_4HL
        Arch_5HL

    optimizer : keras optimizer
    """

    if architecture_name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture_name}")

    hidden_layers = ARCHITECTURES[architecture_name]

    model = build_model(hidden_layers)

    model = compile_model(model, optimizer)

    return model


# =============================================================================
# Model Information
# =============================================================================

def print_model_summary(architecture_name):
    """
    Print summary of one architecture.
    """

    model = build_model(ARCHITECTURES[architecture_name])

    print("\n")
    print("=" * 70)
    print(f"Architecture : {architecture_name}")
    print("=" * 70)

    model.summary()

    print("=" * 70)


# =============================================================================
# Save Initial Weights
# =============================================================================

def get_initial_weights(architecture_name):
    """
    Build a fresh model and return its
    initial weights.

    These weights are reused before training
    every optimizer so that all optimizers start
    from identical initialization.
    """

    model = build_model(
        ARCHITECTURES[architecture_name]
    )

    return model.get_weights()


# =============================================================================
# Reset Model
# =============================================================================

def reset_model_weights(model, initial_weights):
    """
    Reset model to identical initial weights.
    """

    model.set_weights(initial_weights)


# =============================================================================
# Parameter Count
# =============================================================================

def count_parameters(architecture_name):
    """
    Return number of trainable parameters.
    """

    model = build_model(
        ARCHITECTURES[architecture_name]
    )

    return model.count_params()


# =============================================================================
# Display All Architectures
# =============================================================================

def print_all_architectures():
    """
    Print architecture details.
    """

    print("\n")
    print("=" * 80)
    print("AVAILABLE ARCHITECTURES")
    print("=" * 80)

    for name, layers in ARCHITECTURES.items():

        print(f"\n{name}")

        print("-" * 40)

        print(f"Hidden Layers : {len(layers)}")

        print(f"Units         : {layers}")

        print(f"Parameters    : {count_parameters(name):,}")

    print("=" * 80)

# =============================================================================
# Create Untrained Model
# =============================================================================

def create_untrained_model(architecture_name):
    """
    Create a model WITHOUT compiling.

    Used by resume.py to load saved weights.
    """

    if architecture_name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture_name}")

    return build_model(
        ARCHITECTURES[architecture_name]
    )
