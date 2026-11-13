
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment identity
# ---------------------------------------------------------------------------
GROUP = 24
SEED = 42

# ---------------------------------------------------------------------------
# Class selection: Group 24
# ---------------------------------------------------------------------------
CLASSES = {
    "handwriting": ["a", "bA", "dA", "lA", "tA"],
    "cv": ["bhA", "ba", "tA", "kaa", "tI"],
}

# ---------------------------------------------------------------------------
# Hyperparameter configurations
# ---------------------------------------------------------------------------
CONFIGS = [
    {"model": "RNN",  "hidden_size": 32, "num_layers": 1},
    {"model": "RNN",  "hidden_size": 64, "num_layers": 1},
    {"model": "RNN",  "hidden_size": 64, "num_layers": 2},
    {"model": "LSTM", "hidden_size": 32, "num_layers": 1},
    {"model": "LSTM", "hidden_size": 64, "num_layers": 1},
    {"model": "LSTM", "hidden_size": 64, "num_layers": 2},
]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
MAX_EPOCHS = 3000
CONVERGENCE_THRESHOLD = 1e-4

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VALIDATION_FRACTION = 0.15

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
CHECK_DUPLICATES = True
CHECK_FILE_DIMENSIONS = True
VERBOSE = True

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "Data"
CV_DATA_ROOT = DATA_ROOT / "CV_Data"
HW_DATA_ROOT = DATA_ROOT / "Handwriting_Data"

RESULTS_ROOT = PROJECT_ROOT / "Results"
PLOTS_ROOT = PROJECT_ROOT / "Report" / "Plots"

RESULTS_HW_ROOT = RESULTS_ROOT / "Handwriting"
RESULTS_CV_ROOT = RESULTS_ROOT / "CV"

PLOTS_HW_ROOT = PLOTS_ROOT / "Handwriting"
PLOTS_CV_ROOT = PLOTS_ROOT / "CV"

HW_MAPPING_FILE = HW_DATA_ROOT / "Mapping_Handwritten.txt"
CV_MAPPING_FILE = CV_DATA_ROOT / "Mapping_CV.txt"
