
# Project Overview

This repository contains two Jupyter Notebooks for computer vision tasks: 

1. **Convolutional Neural Networks - Build Model** - 
2. **Image Processing Using Opencv** - 

---

## 1. Convolutional Neural Networks - Build Model
**Description:**  
This notebook demonstrates the construction and training of a Convolutional Neural Network (CNN) using PyTorch. It is designed for image classification tasks using the CIFAR-10 dataset.

### Features:
- **Data Loading and Preprocessing:** Utilizes CIFAR-10 dataset with normalization and augmentation.
- **Model Architecture:** Custom CNN with multiple convolutional layers, max-pooling, and fully connected layers.
- **Training and Validation:** Tracks loss and accuracy, with visualization of training progress.
- **Evaluation:** Test accuracy per class and overall model performance.

### Requirements:
- `torch`, `torchvision`, `matplotlib`, `numpy`

### Usage:
1. Install the required libraries:  
    ```bash
    pip install torch torchvision matplotlib numpy
    ```
2. Open and run the notebook:
    ```bash
    jupyter notebook build_cnn.ipynb
    ```

---

## 2. Image Processing Using Opencv
**Description:**  
This notebook focuses on image filtering techniques, including convolution operations, edge detection, and applying various image filters.

### Features:
- **Image Loading and Display:** Loads sample images for demonstration.
- **Image Filtering Techniques:** Applies edge detection, sharpening, and smoothing filters.
- **Custom Kernel Implementation:** Demonstrates custom convolution kernels for advanced filtering.
- **Visualization:** Displays original and filtered images for comparison.

### Requirements:
- `opencv-python`, `matplotlib`, `numpy`

### Usage:
1. Install the required libraries:  
    ```bash
    pip install opencv-python matplotlib numpy
    ```
2. Open and run the notebook:
    ```bash
    jupyter notebook image_filtering.ipynb
    ```

---

## Notes and Troubleshooting:
- Ensure you have GPU support (CUDA) installed if using GPU acceleration for CNN training.
- Adjust the batch size in `build_cnn.ipynb` if encountering memory issues.
- If images do not display correctly, ensure proper un-normalization before using `imshow()`.

## License:
This project is open-source and available under the MIT License.

## Author:
**Abhishek Birajdar** - [GitHub Profile](https://github.com/AbhishekBirajdar/CSCI611_Spring25_Birajdar_Abhishek)

---

## Acknowledgments:
- The CIFAR-10 dataset is provided by [Krizhevsky et al.](https://www.cs.toronto.edu/~kriz/cifar.html)
- PyTorch and OpenCV communities for their amazing tools and resources.

---

## Why a README?
A README file describes how to use your code, providing necessary information on installation, execution, and functionality. It helps users understand the purpose of the project, its features, and how to get started quickly.

