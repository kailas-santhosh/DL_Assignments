# Programming Assignment 4 — PCA and Autoencoders

**Course:** CS601T — Deep Learning  
**Semester:** August–November 2026  
**Assignment Date:** September 30, 2026  
**Submission Deadline:** October 11, 2026, 10:00 PM

## 1. Overview

The objective of this assignment is to understand dimensionality reduction, autoencoders, compressed representations, and their applications to image reconstruction and classification.

The assignment uses the same group-specific subset of the MNIST handwritten digit dataset as Assignment 3.

Each group is assigned five classes. Every image has a resolution of 28 × 28 pixels and is flattened into a 784-dimensional vector.

The assignment investigates PCA-based dimensionality reduction, standard autoencoders, denoising autoencoders, and classification using learned compressed representations.

## 2. Dataset

### MNIST Subset

- Five classes per group.
- Image dimensions: 28 × 28 pixels.
- Input dimension: 784.
- Separate training, validation, and test datasets.
- Same group-specific dataset as Assignment 3.

This assignment uses Dataset 2.

[Download Dataset 2](../Datasets/Data_2.zip)

## 3. Tasks and Implementation

### Task 1: Dimensionality Reduction Using PCA

Apply Principal Component Analysis (PCA) to reduce the original 784-dimensional input into:

- 32 dimensions.
- 64 dimensions.
- 128 dimensions.
- 256 dimensions.

Requirements:

- Obtain PCA projection directions (eigenvectors) using training data only.
- Calculate the mean vector using training data.
- Subtract the training mean from training, validation, and test data.
- Project all three datasets using the training-derived eigenvectors.

Use the reduced representations as inputs to FCNN-based multiclass classification models.

Expected results:

1. Validation accuracy for different FCNN architectures at each reduced dimension.
2. Select the best FCNN architecture using validation accuracy.
3. Present test accuracy and confusion matrix for the selected architecture.
4. Identify the best PCA representation based on test accuracy.
5. Compare performance with Assignment 3.

### Task 2: Autoencoders for Image Reconstruction

Build autoencoders with:

- One hidden layer.
- Three hidden layers.

For the three-hidden-layer architecture:

- First hidden layer: 400 neurons.
- Middle bottleneck layer: 32, 64, 128, or 256 neurons.
- Third hidden layer: 400 neurons.

Use:

- Adam optimizer.
- Linear activation for the bottleneck layer.
- A consistent sigmoid-family activation (logistic or tanh) for the remaining hidden layers.

Expected results:

1. Average reconstruction error on training, validation, and test datasets.
2. Original and reconstructed images for one image from each class.
3. Reconstruction visualizations for each architecture and dataset split.

### Task 3: Classification Using One-Hidden-Layer Autoencoder Representations

Extract compressed representations from the bottleneck layers of the one-hidden-layer autoencoders.

Generate compressed representations for:

- Training data.
- Validation data.
- Test data.

For every compressed representation:

1. Train FCNN-based multiclass classification models.
2. Evaluate validation accuracy for different FCNN architectures.
3. Select the best architecture using validation accuracy.
4. Report test accuracy and confusion matrix.
5. Identify the best compressed representation.
6. Compare performance with Assignment 3 and Task 1.

### Task 4: Classification Using Three-Hidden-Layer Autoencoder Representations

Extract compressed representations from the bottleneck layers of the three-hidden-layer autoencoders.

Repeat the classification experiments using:

- Training representations.
- Validation representations.
- Test representations.

For each reduced representation:

1. Train FCNN-based classification models.
2. Evaluate different FCNN architectures on validation data.
3. Select the best architecture.
4. Report test accuracy and confusion matrix.
5. Identify the best compressed representation.
6. Compare results with Assignment 3, Task 1, and Task 3.

### Task 5: Denoising Autoencoders

Build one-hidden-layer denoising autoencoders using:

- 20% input noise.
- 40% input noise.

Use the bottleneck dimension selected based on the best test accuracy from Task 3.

Expected results:

1. Average reconstruction error on training, validation, and test datasets.
2. Original and reconstructed images for each class and dataset split.
3. Extract compressed representations from the denoising autoencoders.
4. Use the compressed representations as inputs to FCNN classifiers.
5. Evaluate validation and test classification accuracy using the best architecture selected in Task 3 for the corresponding representation.

### Task 6: Weight Visualization

Visualize the input patterns that maximally activate neurons in the compressed representations.

Requirements:

1. For the best one-hidden-layer autoencoder, visualize the input patterns associated with maximum activation of each bottleneck neuron.
2. Repeat the visualization for both denoising autoencoders.
3. Compare the learned representations and activation patterns.

## 4. Evaluation and Analysis

The assignment requires comparisons across:

- PCA-based representations.
- Standard autoencoder representations.
- Three-hidden-layer autoencoder representations.
- Denoising autoencoder representations.
- Original FCNN classification results from Assignment 3.

Include reconstruction errors, classification accuracy, confusion matrices, reconstructed images, activation visualizations, and observations.

## 5. Implementation Requirements

- Implement PCA and autoencoder experiments.
- Deep learning APIs such as PyTorch, TensorFlow, and Keras are permitted.
- Use Adam for autoencoder training.
- Use the training set to determine PCA projection directions and mean.
- Use validation accuracy for architecture selection.
- Use test data for final evaluation.

## 6. Technologies

- Python 3
- PyTorch / TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

## 7. Execution

Navigate to the assignment code directory:

```bash
cd Assignment_4/Code
```

Install the required dependencies according to your implementation:

```bash
pip install torch numpy matplotlib scikit-learn
```

Run:

```bash
python main.py
```

Adjust the execution command according to the actual entry-point filename.

## 8. Submission Requirements

The original academic submission requires:

**Code folder:**
`Group<number>_Assignment4_code`

**Code ZIP file:**
`Group<number>_Assignment4_code.zip`

**Report PDF:**
`Group<number>_Assignment4_report.pdf`

The report must include all experimental results, reconstruction visualizations, classification comparisons, weight visualizations, and observations.

Upload the code ZIP and PDF report to Moodle.

## 9. Repository Structure

```text
Assignment_4/
├── Code/
├── Problem_Statement.pdf
└── README.md
```

---

**Course:** CS601T — Deep Learning  
**Assignment:** Programming Assignment 4  
**Semester:** August–November 2026
