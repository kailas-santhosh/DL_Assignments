"""
config.py

Global configuration file for Assignment 3.
Modify experiment settings only in this file.
"""

from pathlib import Path
import tensorflow as tf

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Data"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

RESULTS_DIR = PROJECT_ROOT / "Results"

PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
LOGS_DIR = RESULTS_DIR / "logs"
HISTORIES_DIR = RESULTS_DIR / "histories"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"

# =============================================================================
# Dataset
# =============================================================================

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_CHANNELS = 1

INPUT_DIM = 784

NUM_CLASSES = 5

CLASS_NAMES = ["0", "1", "5", "7", "8"]

# =============================================================================
# Reproducibility
# =============================================================================

RANDOM_SEED = 42

# =============================================================================
# Architectures
# =============================================================================

ARCHITECTURES = {

    "3HL_128-64-32": [128, 64, 32],
    "3HL_256-128-64": [256, 128, 64],

    "4HL_256-128-64-32": [256, 128, 64, 32],
    "4HL_512-256-128-64": [512, 256, 128, 64],

    "5HL_256-128-64-32-16": [256, 128, 64, 32, 16],
    "5HL_512-256-128-64-32": [512, 256, 128, 64, 32],
}

# =============================================================================
# Activation
# =============================================================================

HIDDEN_ACTIVATION = "relu"
OUTPUT_ACTIVATION = "softmax"

# =============================================================================
# Optimizer Hyperparameters
# =============================================================================

LEARNING_RATE = 0.001

MOMENTUM = 0.9

RMSPROP_RHO = 0.99
RMSPROP_EPSILON = 1e-8

ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1e-8

# =============================================================================
# Batch Sizes
# =============================================================================

SGD_BATCH_SIZE = 1
MOMENTUM_BATCH_SIZE = 1
NAG_BATCH_SIZE = 1
ADAM_BATCH_SIZE = 1

# Full-batch optimizers (updated after dataset loading)
FULL_BATCH_SIZE = None

# =============================================================================
# Training
# =============================================================================

LOSS_FUNCTION = tf.keras.losses.CategoricalCrossentropy()

STOPPING_THRESHOLD = 1e-4

# Safety limit only.
# Training MUST stop using STOPPING_THRESHOLD,
# not because of this value.
MAX_EPOCHS = 1_000_000

VERBOSE = 1

# =============================================================================
# Optimizers to Evaluate
# =============================================================================

OPTIMIZERS = [
    "SGD",	
    "BatchGD",
    "Momentum",
    "NAG",
    "AdaGrad",
    "RMSProp",
    "Adam",
]

# =============================================================================
# Plot Settings
# =============================================================================

FIGSIZE = (12,6)

DPI = 400

LINE_WIDTH = 2.0

FONT_SIZE = 12

# =============================================================================
# Output Files
# =============================================================================

EPOCH_TABLE = TABLES_DIR / "epochs_summary.csv"

ACCURACY_TABLE = TABLES_DIR / "accuracy_summary.csv"

BEST_MODEL_TABLE = TABLES_DIR / "best_architecture.csv"

TRAINING_LOG = LOGS_DIR / "training_log.csv"

# =============================================================================
# Create Output Directories
# =============================================================================

OUTPUT_DIRS = [
    RESULTS_DIR,
    PLOTS_DIR,
    TABLES_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    HISTORIES_DIR,
    CONFUSION_DIR,
]

for directory in OUTPUT_DIRS:
    directory.mkdir(parents=True, exist_ok=True)
