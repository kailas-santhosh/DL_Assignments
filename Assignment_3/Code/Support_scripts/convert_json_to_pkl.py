"""
convert_json_to_pkl.py

Converts the BatchGD JSON history into the same
history dictionary format used by the other .pkl files.
"""

import json
import pickle
from pathlib import Path

# ---------------------------------------------------------
# Input JSON
# ---------------------------------------------------------

json_file = Path(
    "/home/kali/Desktop/Ass_3_Trial_Run/Assignment_3/Final/FINAL/final/Results/histories/"
    "5HL_256-128-64-32-16_BatchGD.json"
)

# ---------------------------------------------------------
# Output PKL
# ---------------------------------------------------------

pkl_file = Path(
    "/home/kali/Desktop/Deep_Learning/Assignment/Assignment_3/Final/final/Results/histories/"
    "5HL_256-128-64-32-16_BatchGD.pkl"
)

# ---------------------------------------------------------
# Read JSON
# ---------------------------------------------------------

with open(json_file, "r") as f:
    data = json.load(f)

loss_history = data["loss_history"]
epochs = len(loss_history)

# ---------------------------------------------------------
# Create history dictionary
# ---------------------------------------------------------

history = {
    "loss": loss_history,

    # These are not used for your bar plots,
    # but are added for compatibility with Keras history.
    "accuracy": [data["final_train_acc"] / 100.0] * epochs,
    "val_accuracy": [data["final_val_acc"] / 100.0] * epochs,
}

# ---------------------------------------------------------
# Save PKL
# ---------------------------------------------------------

with open(pkl_file, "wb") as f:
    pickle.dump(history, f)

print()
print("=" * 60)
print("JSON successfully converted to PKL")
print("=" * 60)
print(f"Input : {json_file}")
print(f"Output: {pkl_file}")
print(f"Epochs: {epochs}")
print("=" * 60)
