# Programming Assignment 5 — Convolutional Neural Networks

**Course:** CS601T — Deep Learning  
**Semester:** August–November 2026  
**Assignment Date:** October 12, 2026  
**Submission Deadline:** October 31, 2026, 10:00 PM

## 1. Overview

The objective of this assignment is to understand the functional components of Convolutional Neural Networks (CNNs) and their application to image classification.

The assignment is divided into two parts:

- **Part A:** CNN implementation and feature visualization using a subset of the MNIST handwritten digit dataset.
- **Part B:** CNN architectures and transfer learning using a subset of the Caltech-101 dataset.

The assignment explores convolution, activation functions, pooling, feature extraction, backpropagation, transfer learning, guided backpropagation, and Grad-CAM visualization.

# PART A — CNN Using MNIST

## 2. Dataset

Use the same group-specific subset of the MNIST dataset as Assignments 3 and 4.

- Five classes per group.
- Image dimensions: 28 × 28 pixels.
- Separate training, validation, and test datasets.

This assignment uses Dataset 3, containing the data required
for Part A (MNIST) and Part B (Caltech-101).

[Download Dataset 3](../Datasets/Data_3.zip)

## 3. Task 1: Convolution Operation

Select one image from each of the five classes.

Requirements:

1. Initialize a 5 × 5 convolution filter using Kaiming initialization.
2. Apply convolution with:
   - Stride = 1.
   - Padding = 0.
3. Generate the resulting feature maps.
4. Calculate the expected feature map dimensions mathematically.
5. Verify the theoretical dimensions against the implemented output.

Document the selected images, initialized filter values, resulting feature maps, and observations.

## 4. Task 2: Convolutional and Pooling Layers

Implement a convolutional layer with ReLU activation followed by max pooling.

### First Convolutional Layer

- Filters: 6.
- Kernel size: 5 × 5.
- Stride: 1.
- Padding: 0.
- Max pooling: 2 × 2.
- Pooling stride: 2.

### Second Convolutional Layer

- Filters: 16.
- Kernel size: 5 × 5.
- Stride: 1.
- Padding: 0.
- Max pooling: 2 × 2.
- Pooling stride: 2.

Use Kaiming initialization for convolutional filters.

Pass the five selected images through both layers and:

1. Print feature maps after each layer.
2. Calculate and verify feature map dimensions mathematically.
3. Visualize any three filters from the first convolutional layer.
4. Visualize any three filters from the second convolutional layer.
5. Display corresponding feature maps.

## 5. Task 3: CNN for MNIST Classification

Develop a CNN with the following architecture:

| Layer | Configuration |
|---|---|
| Input | 28 × 28 grayscale image |
| Conv1 | 6 filters, 5 × 5, stride 1, padding 0 |
| ReLU + MaxPool1 | 2 × 2, stride 2 |
| Conv2 | 16 filters, 5 × 5, stride 1, padding 0 |
| ReLU + MaxPool2 | 2 × 2, stride 2 |
| Flatten | Flatten feature maps |
| Fully Connected 1 | 64 neurons, ReLU |
| Fully Connected 2 | 5 neurons, Softmax |

Initialization:

- Convolutional filters: Kaiming initialization.
- Fully connected layers: Xavier initialization.

Train using stochastic gradient descent and backpropagation.

### Expected Results

1. Training, validation, and test classification accuracy.
2. Confusion matrices.
3. Visualize 10 filters from the first convolutional layer.
4. Visualize 10 filters from the second convolutional layer.
5. Display corresponding feature maps for the selected images.
6. Identify the five most activated neurons in the last convolutional layer for each selected image.
7. Trace the activations back to their corresponding input image patches.
8. Visualize bounding boxes around the image patches responsible for maximum activation.

# PART B — CNN Using Caltech-101

## 6. Dataset

Use the group-specific subset of the Caltech-101 dataset.

- Five classes per group.
- Color images.
- Separate training, validation, and test datasets.
- Resize all images to 224 × 224 pixels.

**Dataset location:** [Central Datasets Directory](../Datasets/)

## 7. Task 1: CNN Architecture Comparison

Develop three CNN architectures.

### Architecture 1 — Four-Layer Architecture

| Layer | Configuration |
|---|---|
| Input | 224 × 224 × 3 |
| Conv1 | 8 filters, 11 × 11, stride 4, padding 0 |
| MaxPool1 | 3 × 3, stride 2 |
| Conv2 | 16 filters, 5 × 5, stride 1, padding 0 |
| MaxPool2 | 3 × 3, stride 2 |
| Flatten | Flatten feature maps |
| Fully Connected 1 | 128 neurons, ReLU |
| Output | 5 neurons, Softmax |

### Architecture 2 — Five-Layer Architecture

The first two convolutional layers are identical to Architecture 1.

Additional layer:

- Conv3: 32 filters, 3 × 3, stride 1, padding 0.
- MaxPool3: 3 × 3, stride 2.

Follow with flattening, a 128-neuron ReLU fully connected layer, and a five-neuron Softmax output layer.

### Architecture 3 — Six-Layer Architecture

The first two convolutional layers are identical to Architecture 1.

Additional layers:

- Conv3: 32 filters, 3 × 3, stride 1, padding 0, without pooling.
- Conv4: 64 filters, 3 × 3, stride 1, padding 0.
- MaxPool4: 3 × 3, stride 2.

Follow with flattening, a 128-neuron ReLU fully connected layer, and a five-neuron Softmax output layer.

### Expected Results

1. Training and validation accuracy for all three architectures.
2. Confusion matrices for the evaluated architectures.
3. Mathematical calculation and description of feature map dimensions at each layer.
4. Select the best architecture using validation accuracy.
5. Report test accuracy and confusion matrix for the selected architecture.
6. Visualize all eight feature maps from the first convolutional layer and eight selected feature maps from the remaining convolutional layers.
7. Identify maximally activated neurons in the last convolutional layer.
8. Trace their activations back to the corresponding image patches and visualize those patches.

## 8. Task 2: Transfer Learning Using VGG19

Use the pretrained VGG19 network with ImageNet weights through TensorFlow Keras.

Requirements:

- Modify the classification layer to produce five output classes.
- Retrain only the classification layer.
- Keep the pretrained feature extraction layers frozen.

### Expected Results

1. Training, validation, and test accuracy.
2. Confusion matrices.
3. Compare test performance with the best CNN architecture from Part B, Task 1.
4. Identify maximally activated neurons in the last convolutional layer and visualize the corresponding image patches.
5. Apply guided backpropagation to visualize the influence of input pixels on selected neurons.
6. Document the selected image, neuron positions, gradient images, and observations.
7. Apply Grad-CAM to visualize class-discriminative localization maps.
8. Generate heatmaps for each of the five selected images against all five classes, resulting in 25 visualizations.

## 9. Implementation Requirements

- CNN architectures may be implemented using PyTorch, TensorFlow, or Keras.
- Use Kaiming initialization for convolutional filters in Part A.
- Use Xavier initialization for fully connected layers in Part A.
- Use the prescribed convolution and pooling configurations.
- Use pretrained VGG19 for transfer learning.
- Include feature maps, activation visualizations, guided backpropagation, and Grad-CAM results.

## 10. Technologies

- Python 3
- PyTorch
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn

## 11. Execution

Navigate to the assignment code directory:

```bash
cd Assignment_5/Code
```

Install the required dependencies according to your implementation:

```bash
pip install torch tensorflow numpy matplotlib opencv-python scikit-learn
```

Run the implementation:

```bash
python main.py
```

Adjust the command according to the actual entry-point filename and ensure the required datasets are available.

## 12. Submission Requirements

The original academic submission requires:

**Code folder:**
`Group<number>_Assignment5_code`

**Code ZIP file:**
`Group<number>_Assignment5_code.zip`

**Report PDF:**
`Group<number>_Assignment5_report.pdf`

The report must include architectural descriptions, feature maps, classification results, confusion matrices, activation visualizations, guided backpropagation results, Grad-CAM heatmaps, and observations.

Upload the code ZIP and PDF report to Moodle.

## 13. Repository Structure

```text
Assignment_5/
├── Code/
├── Problem_Statement.pdf
└── README.md
```

---

**Course:** CS601T — Deep Learning  
**Assignment:** Programming Assignment 5  
**Semester:** August–November 2026
