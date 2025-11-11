# MNIST-Digit-Classification-using-Neural-Networks
Here is a clean, professional **README.md** file tailored exactly to your implementation — no extra features, only what your code contains.

---

# **MNIST Digit Classification using Neural Networks (From Scratch)**

This project implements a **handwritten digit classifier** for the MNIST dataset by building a complete **Artificial Neural Network (ANN) from scratch using NumPy**.
No deep-learning frameworks (TensorFlow/Keras/PyTorch) were used.

---

## 🚀 **Project Overview**

This implementation covers the full machine learning pipeline:

* Loading MNIST using **fetch_openml**
* **Normalization** using Min-Max scaling
* **Train/Validation/Test split**
* **One-hot encoding** for labels
* Building a multi-layer neural network:

  * Input: 784 neurons
  * Hidden Layer 1: 128 neurons (ReLU)
  * Hidden Layer 2: 64 neurons (ReLU)
  * Output Layer: 10 neurons (Softmax)
* Custom implementations of:

  * He weight initialization
  * Forward propagation
  * Cross-entropy loss
  * Backpropagation
  * Mini-batch gradient descent
* Evaluation on test set using:

  * Confusion Matrix
  * Accuracy
  * Precision
  * F1 Score

All plots (training curves, confusion matrix, sample predictions) are generated with **Matplotlib**.

---

## 📂 **Directory Structure**

```
|-- CS_Project_Dharmraj.ipynb
|-- README.md   (this file)
```

---

## 🧠 **Neural Network Architecture**

```
Input Layer      : 784 neurons (28×28 image)
Hidden Layer 1   : 128 neurons + ReLU
Hidden Layer 2   : 64 neurons + ReLU
Output Layer     : 10 neurons + Softmax
```

Parameter initialization uses **He Initialization** to improve training stability.

---

## 🔧 **Training Details**

* **Epochs:** 15
* **Batch Size:** 128
* **Learning Rate:** 0.1
* **Optimizer:** Mini-Batch Gradient Descent

During training, the code logs:

* Training loss
* Training accuracy
* Validation loss
* Validation accuracy

Both **loss and accuracy curves** are plotted after training.

---

## 📊 **Evaluation Metrics**

After training, the model is evaluated on the **10,000-image test set** using:

* **Confusion Matrix**
* **Accuracy Score**
* **Precision Score (weighted)**
* **F1 Score (weighted)**
* Per-class Precision & F1
* Visualization of sample predictions (correct vs incorrect)

---

## 🖼️ **Visualizations Included**

* Random MNIST samples
* Training loss curve
* Training accuracy curve
* Confusion matrix heatmap
* Sample test predictions with confidence scores

---

## ▶️ **How to Run**

1. Upload or open the notebook in **Google Colab** or any Python environment.
2. Install required dependencies (usually pre-installed in Colab):

```bash
pip install numpy matplotlib scikit-learn
```

3. Run all cells in `CS_Project_Dharmraj.ipynb`.

---

## 📚 **Dependencies**

* **NumPy**
* **Matplotlib**
* **scikit-learn**



---

## 🎯 **Key Skills Demonstrated**

* Neural networks implemented mathematically from scratch
* Backpropagation, gradients, and optimization
* Data preprocessing and model evaluation
* Working with high-dimensional datasets
* Model visualization and performance diagnostics

