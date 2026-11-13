# Programming Assignment 6 — RNN and LSTM for Sequence Classification

**Course:** CS601T — Deep Learning  
**Semester:** August–November 2026  
**Assignment Date:** October 30, 2026  
**Submission Deadline:** November 12, 2026, 10:00 PM

## 1. Overview

The objective of this assignment is to understand Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks for sequential data classification.

The assignment explores sequence modeling using two different types of sequential datasets:

1. Handwritten character recognition using two-dimensional coordinate sequences.
2. Consonant-Vowel (CV) segment classification using Mel-Frequency Cepstral Coefficient (MFCC) features.

Different network architectures and hyperparameter configurations are investigated to identify the best-performing RNN and LSTM models.

## 2. Datasets

### Dataset 1: Handwritten Character Recognition

This dataset contains a subset of handwritten characters from Kannada/Telugu scripts.

Characteristics:

- Five character classes per group.
- Each character is represented as a sequence of two-dimensional points.
- Each point contains an x and y coordinate.
- Each data file represents one sample.
- Each sequence corresponds to one pen stroke, from pen-down to pen-up.
- Training and test samples are provided separately.
- Test samples are located inside the respective `dev` directories.

### Data Format

Each data file contains an array of numerical elements:

- The first element indicates the number of sequential points.
- Remaining elements represent x and y coordinates in pairs.

The coordinates must be normalized independently for each sample to the range [0, 1].

Refer to `Mapping_Handwritten.txt` to identify the five character classes assigned to the group.

### Dataset 2: Consonant-Vowel (CV) Segment Classification

This dataset contains subsets of CV segments extracted from conversational Hindi speech.

Characteristics:

- Five CV segment classes per group.
- Each data file represents one sample.
- Features are extracted frame-wise from speech segments.
- Each frame contains a 39-dimensional MFCC feature vector.
- The number of frames varies depending on the duration of the speech segment.
- Training and test samples are provided separately.

### Data Format

Each row in a data file represents one 39-dimensional MFCC feature vector.

Each sample is therefore represented as a variable-length sequence of 39-dimensional feature vectors.

Refer to `Mapping_CV.txt` to identify the five CV segment classes assigned to the group.

### Dataset Location

Both datasets are maintained in the central repository dataset directory.

This assignment uses Dataset 4, containing the handwritten
character and Consonant-Vowel (CV) segment datasets.

- [Download Dataset 4](../Datasets/Data_4.zip)
- [Dataset Documentation](../Datasets/README.md)

Refer to the dataset archive descriptions and group-specific mapping files to identify the correct data for each experiment.

## 3. Model Architectures

Develop two types of recurrent neural network models for both datasets:

1. Recurrent Neural Network (RNN).
2. Long Short-Term Memory (LSTM).

Experiment with different hyperparameter configurations, including:

- Number of hidden layers.
- Number of neurons in each hidden layer.
- Number of RNN/LSTM layers.
- Network architecture and model complexity.

The objective is to identify the best RNN and LSTM configurations for each dataset.

## 4. Training and Convergence

Train the RNN and LSTM models using the respective sequential datasets.

### Convergence Criteria

Use the difference between average errors of successive epochs as the stopping criterion.

Training must stop when:

\[
|E_t-E_{t-1}| < 10^{-4}
\]

where:

- \(E_t\) is the average training error at the current epoch.
- \(E_{t-1}\) is the average training error at the previous epoch.

A fixed number of epochs must not be used as the convergence criterion.

## 5. Experimental Requirements

### 5.1 Training Error Visualization

For every hyperparameter configuration:

- Plot average training error against epochs.
- Compare convergence behavior across different architectures.
- Observe the effect of hidden layers and hidden neurons on training performance.

### 5.2 Performance Evaluation

For every hyperparameter configuration, report:

- Training accuracy.
- Test accuracy.

Evaluate RNN and LSTM models separately for both datasets.

### 5.3 Best Model Selection

Identify the best RNN and LSTM models based on their performance.

For each selected model, report:

- Training accuracy.
- Test accuracy.
- Confusion matrix.

Compare the classification performance of RNN and LSTM models.

### 5.4 Handwritten Character Visualization

For the handwritten character dataset:

- Select at least five images/samples from each assigned character class.
- Normalize the x and y coordinates to the range [0, 1].
- Visualize the normalized handwritten character sequences.
- Include the visualizations in the final report.

## 6. Implementation Requirements

- Implement RNN and LSTM models using Python or another programming language.
- Deep learning APIs such as PyTorch, TensorFlow, and Keras are permitted.
- Support variable-length sequential inputs.
- Normalize handwritten character coordinates independently for each sample.
- Process MFCC features as sequences of 39-dimensional vectors.
- Experiment with different network architectures and hyperparameters.
- Use the prescribed convergence threshold of 10⁻⁴.

## 7. Technologies

- Python 3
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## 8. Execution

Navigate to the assignment code directory:

```bash
cd Assignment_6/Code
```

Install the required dependencies:

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

Run the implementation:

```bash
python main.py
```

Adjust the command according to the actual entry-point filename and ensure that both datasets are extracted and their paths are configured correctly.

## 9. Submission Requirements

The original academic submission requires:

**Code folder:**
`Group<number>_Assignment6_code`

**Code ZIP file:**
`Group<number>_Assignment6_code.zip`

**Report PDF:**
`Group<number>_Assignment6_report.pdf`

The report must include:

- Training error plots for different hyperparameter configurations.
- Training and test accuracy for all configurations.
- Best RNN and LSTM model results.
- Confusion matrices.
- At least five normalized handwritten character samples from every assigned class.
- Comparisons and observations.

Upload the code ZIP and PDF report to Moodle.

## 10. Repository Structure

```text
Assignment_6/
├── Code/
├── Problem_Statement.pdf
└── README.md
```

---

**Course:** CS601T — Deep Learning  
**Assignment:** Programming Assignment 6  
**Semester:** August–November 2026
