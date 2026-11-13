# Programming Assignment 3 — Optimization Algorithms for FCNN

**Course:** CS601T — Deep Learning  
**Semester:** August–November 2026  
**Assignment Date:** September 2026  
**Submission Deadline:** September 15, 2026, 10:00 PM

## 1. Overview

The objective of this assignment is to investigate and compare different optimization algorithms used for training Fully Connected Neural Networks (FCNNs).

The assignment focuses on understanding how different optimizers affect convergence speed, training error, and classification performance.

A subset of the MNIST handwritten digit dataset is provided. Each group is assigned five classes, with separate training, validation, and test datasets.

Each image has a resolution of 28 × 28 pixels and must be flattened into a 784-dimensional input vector.

## 2. Dataset

### MNIST Subset

- Five digit classes per group.
- Image dimensions: 28 × 28 pixels.
- Flattened input dimension: 784.
- Separate training, validation, and test datasets.
- Group-specific class allocation.

The same group-specific dataset must be used throughout the assignment.

This assignment uses Dataset 2.

[Download Dataset 2](../Datasets/Data_2.zip)

Refer to the central dataset directory for the MNIST subset and group-specific data files.

## 3. Model Architecture

Develop Fully Connected Neural Networks with:

- Minimum: 3 hidden layers.
- Maximum: 5 hidden layers.

Experiment with different numbers of neurons in each hidden layer to study the effect of network architecture on convergence and classification performance.

### Loss Function

Use cross-entropy loss.

### Optimization Algorithms

Train every architecture using the following seven optimization algorithms:

1. Stochastic Gradient Descent (SGD)
   - Batch size = 1.

2. Batch Gradient Descent (Vanilla Gradient Descent)
   - Batch size = Total number of training examples.

3. SGD with Momentum (Generalized Delta Rule)
   - Batch size = 1.

4. SGD with Nesterov Accelerated Gradient (NAG)
   - Batch size = 1.

5. AdaGrad
   - Batch size = Total number of training examples.

6. RMSProp
   - Batch size = Total number of training examples.

7. Adam
   - Batch size = 1.

## 4. Hyperparameters

Use the following prescribed hyperparameters:

| Parameter | Value |
|---|---|
| Learning rate | 0.001 |
| Momentum (Generalized Delta Rule) | 0.9 |
| Momentum (NAG) | 0.9 |
| RMSProp beta | 0.99 |
| RMSProp epsilon | 1e-8 |
| Adam beta1 | 0.9 |
| Adam beta2 | 0.999 |
| Adam epsilon | 1e-8 |
| Convergence threshold | 1e-4 |

### Convergence Criteria

Training must stop when the absolute difference between the average errors of successive epochs falls below 10⁻⁴.

A fixed number of iterations or epochs must not be used as the convergence criterion.

### Weight Initialization

Use the same initial random weight values for every optimizer when comparing optimizers within the same architecture.

This ensures a fair comparison of optimization algorithms.

## 5. Experimental Requirements

### 5.1 Convergence Analysis

- Record the number of epochs required for convergence for each optimizer and architecture.
- Tabulate and compare convergence epochs.

### 5.2 Training Error Visualization

Plot average training error against epochs for every architecture.

Superimpose the error curves from all seven optimizers for each architecture.

### 5.3 Performance Evaluation

Report:

- Training accuracy for every optimizer and architecture.
- Validation accuracy for every optimizer and architecture.

### 5.4 Best Architecture Selection

Select the best architecture based on validation accuracy.

For the selected architecture, report:

- Training accuracy.
- Training confusion matrix.
- Test accuracy.
- Test confusion matrix.

## 6. Implementation Requirements

- Implement FCNN models with at least three and at most five hidden layers.
- Compare all seven optimization algorithms.
- Use the same initial weights when comparing optimizers within an architecture.
- PyTorch, TensorFlow, Keras, or other deep learning APIs are permitted.
- Include observations about convergence and classification performance.

## 7. Technologies

- Python 3
- PyTorch / TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

## 8. Execution

Navigate to the assignment code directory:

```bash
cd Assignment_3/Code
```

Install the required dependencies according to the implementation:

```bash
pip install torch numpy matplotlib scikit-learn
```

Run the implementation:

```bash
python main.py
```

Adjust the command according to the actual entry-point filename.

## 9. Submission Requirements

The original academic submission requires:

**Code folder:**
`Group<number>_Assignment3_code`

**Code ZIP file:**
`Group<number>_Assignment3_code.zip`

**Report PDF:**
`Group<number>_Assignment3_report.pdf`

The report must include optimizer comparisons, convergence analysis, accuracy results, confusion matrices, plots, and observations.

Upload the code ZIP and PDF report to Moodle.

## 10. Repository Structure

```text
Assignment_3/
├── Code/
├── Problem_Statement.pdf
└── README.md
```

---

**Course:** CS601T — Deep Learning  
**Assignment:** Programming Assignment 3  
**Semester:** August–November 2026
