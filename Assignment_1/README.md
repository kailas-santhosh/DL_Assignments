# Programming Assignment 1 — Perceptron

**Course:** CS601T — Deep Learning  
**Semester:** August–November 2026  
**Assignment Date:** August 14, 2026  
**Submission Deadline:** August 23, 2026, 10:00 PM

## 1. Overview

The objective of this assignment is to understand and implement the Perceptron learning algorithm from scratch using gradient descent.

The assignment covers both classification and regression problems. Different activation functions are explored to understand their effect on learning and model performance.

## 2. Classification Tasks

### Datasets

Two datasets are provided for classification:

- **Dataset 1 — Linearly Separable Data:** Three classes of two-dimensional linearly separable data, containing 500 data points per class.
- **Dataset 2 — Nonlinearly Separable Data:** Two-dimensional data containing two or three nonlinearly separable classes. The number of examples in each class and their ordering are specified at the beginning of each file.

### Data Splitting

Each class is divided into:
- Training: 70%
- Testing: 30%

### Model and Implementation

Implement a Perceptron with the following activation functions:

1. Logistic (Sigmoid)
2. Hyperbolic Tangent (Tanh)

Requirements:
- Implement the gradient descent learning algorithm from scratch.
- Use the one-against-one approach for three-class classification.
- Train and evaluate the models separately for both classification datasets.

### Expected Results

1. Plot average training error against epochs.
2. Plot decision regions superimposed with training data:
   - Decision regions between each pair of classes.
   - Combined decision regions for all classes.
3. Present test-set evaluation metrics:
   - Confusion matrix.
   - Classification accuracy.
   - Precision for every class and mean precision.
   - Recall for every class and mean recall.
   - F-measure for every class and mean F-measure.
4. Discuss observations from the plots and evaluation results.

## 3. Regression Tasks

### Datasets

- **Dataset 1:** Univariate regression with one-dimensional input.
- **Dataset 2:** Bivariate regression with two-dimensional input.

### Data Splitting

- Training: 70%
- Testing: 30%

### Model and Implementation

Implement a Perceptron with a linear activation function using gradient descent.

### Expected Results

1. Plot average training error against epochs.
2. Calculate RMSE and percentage RMSE on training and test datasets.
3. Plot model predictions superimposed on target outputs:
   - Dataset 1: Input \(x\) versus target/model output \(y\).
   - Dataset 2: Inputs \(x_1,x_2\) versus target/model output \(y\).
4. Generate scatter plots of target outputs versus model outputs for training and test data.
5. Discuss the observed results and model performance.

## 4. Implementation Requirements

- Implement the Perceptron from scratch.
- Python, MATLAB, or another programming language may be used.
- Do not use existing Perceptron, neural network, or gradient descent libraries.
- Implement the learning algorithm and relevant mathematical operations yourself.

## Dataset

This assignment uses Dataset 1.

[Download Dataset 1](../Datasets/Data_1.zip)

Refer to the dataset directory for the relevant files assigned to your group.

## 6. Technologies

- Python 3
- NumPy
- Matplotlib

## 7. Execution

Navigate to the assignment code directory:

```bash
cd Assignment_1/Code
```

Install the required dependencies:

```bash
pip install numpy matplotlib
```

Run the implementation:

```bash
python main.py
```

Adjust the execution command according to the actual entry-point filename in the Code directory.

## 8. Submission Requirements

The original academic submission requires:

**Code folder:**
`Group<number>_Assignment1_code`

**Code ZIP file:**
`Group<number>_Assignment1_code.zip`

**Report PDF:**
`Group<number>_Assignment1_report.pdf`

The report must include experimental results, plots, observations, and conclusions.

**Note:** The original submission instructions require uploading the code ZIP and PDF report to Moodle.

## 9. Repository Structure

```text
Assignment_1/
├── Code/
├── Problem_Statement.pdf
└── README.md
```

---

**Course:** CS601T — Deep Learning  
**Assignment:** Programming Assignment 1  
**Semester:** August–November 2026
