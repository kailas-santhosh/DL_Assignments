# Programming Assignment 2 — Fully Connected Neural Networks

**Course:** CS601T — Deep Learning  
**Semester:** August–November 2026  
**Assignment Date:** August 2026  
**Submission Deadline:** September 7, 2026, 10:00 PM

## 1. Overview

The objective of this assignment is to understand Fully Connected Neural Networks (FCNNs) and their application to classification and regression problems.

The assignment extends the single-neuron Perceptron models implemented in Assignment 1 by introducing hidden layers, different network architectures, and stochastic gradient descent-based backpropagation.

The performance of different network architectures is evaluated using a validation dataset, and the best architecture is selected for final testing.

## 2. Classification Tasks

### Datasets

The same classification datasets used in Assignment 1 are considered:

- **Dataset 1 — Linearly Separable Data:** Three classes of two-dimensional linearly separable data, with 500 samples per class.
- **Dataset 2 — Nonlinearly Separable Data:** Two-dimensional data containing two or three nonlinearly separable classes.

### Data Splitting

Each class is divided into:

- Training: 60%
- Validation: 20%
- Testing: 20%

### Model and Implementation

Develop FCNN models with the following architectures:

- **Dataset 1:** One hidden layer.
- **Dataset 2:** Two hidden layers.

Experiment with different numbers of hidden neurons to investigate the effect of model complexity.

Implementation requirements:

- Implement stochastic gradient descent (SGD) for backpropagation.
- Use squared error as the instantaneous loss function.
- Evaluate different network architectures using validation performance.
- Select the best architecture based on validation results.

### Expected Results

1. Plot average training error against epochs for the best architecture.
2. Plot decision regions superimposed with training samples for the best architecture.
3. Present validation results for all architectures:
   - Confusion matrix.
   - Classification accuracy.
   - Precision for each class and mean precision.
   - Recall for each class and mean recall.
   - F-measure for each class and mean F-measure.
4. Present test-set confusion matrix and classification metrics for the best architecture.
5. Visualize the outputs of individual hidden and output neurons for training, validation, and test datasets.
6. Compare FCNN performance with the single-neuron Perceptron models from Assignment 1.
7. Discuss observations and conclusions.

## 3. Regression Tasks

### Datasets

The same regression datasets used in Assignment 1 are considered:

- **Dataset 1:** Univariate regression with one-dimensional input.
- **Dataset 2:** Bivariate regression with two-dimensional input.

### Data Splitting

- Training: 60%
- Validation: 20%
- Testing: 20%

### Model and Implementation

Develop FCNN models using:

- **Dataset 1:** One hidden layer.
- **Dataset 2:** One and two hidden layers.

Experiment with different numbers of hidden neurons.

Use SGD-based backpropagation with squared error as the instantaneous loss function.

### Expected Results

1. Plot average training error against epochs for the best architecture.
2. Calculate RMSE and percentage RMSE for different architectures on training and validation data.
3. Calculate RMSE and percentage RMSE on test data for the best architecture.
4. Plot predicted outputs superimposed on target outputs for training, validation, and test datasets.
5. Generate scatter plots of target outputs versus model outputs for all three datasets.
6. Visualize outputs of hidden and output neurons for the best architecture.
7. Compare performance with the Perceptron models from Assignment 1.
8. Discuss observations and conclusions.

## 4. Implementation Requirements

- Implement FCNN models from scratch.
- Python, MATLAB, or another programming language may be used.
- Do not use TensorFlow, PyTorch, Keras, or existing neural network and gradient descent implementations.
- Implement forward propagation, backpropagation, and SGD yourself.

## Dataset

This assignment uses Dataset 1.

[Download Dataset 1](../Datasets/Data_1.zip)

Refer to the central dataset directory and the group-specific dataset allocation.

## 6. Technologies

- Python 3
- NumPy
- Matplotlib

## 7. Execution

Navigate to the assignment code directory:

```bash
cd Assignment_2/Code
```

Install the required dependencies:

```bash
pip install numpy matplotlib
```

Run the implementation:

```bash
python main.py
```

Adjust the execution command according to the actual entry-point filename.

## 8. Submission Requirements

The original academic submission requires:

**Code folder:**
`Group<number>_Assignment2_code`

**Code ZIP file:**
`Group<number>_Assignment2_code.zip`

**Report PDF:**
`Group<number>_Assignment2_report.pdf`

The report must contain experimental results, architecture comparisons, visualizations, observations, and conclusions.

Upload the code ZIP and PDF report to Moodle.

## 9. Repository Structure

```text
Assignment_2/
├── Code/
├── Problem_Statement.pdf
└── README.md
```

---

**Course:** CS601T — Deep Learning  
**Assignment:** Programming Assignment 2  
**Semester:** August–November 2026
