# Recurrent Neural Network (RNN) on MNIST Dataset

## Overview
This project demonstrates the use of a **Recurrent Neural Network (RNN)** to classify handwritten digits from the **MNIST dataset**. Unlike traditional feedforward or convolutional models, the MNIST images are treated as **sequences** and processed using recurrent layers.

All implementation and analysis are provided in the Jupyter Notebook:

**`mnist rnn.ipynb`**

---

## Dataset: MNIST
MNIST is a standard benchmark dataset for handwritten digit recognition.

**Dataset Details:**
- 70,000 grayscale images
- Image size: 28 × 28 pixels
- 10 digit classes (0–9)
- 60,000 training images
- 10,000 testing images

---

## Objective
The objectives of this project are to:
- Apply Recurrent Neural Networks to image classification
- Understand how images can be modeled as sequential data
- Evaluate RNN performance on visual datasets
- Compare RNN-based learning with CNN and ANN approaches

---

## Technologies Used
- Python  
- Jupyter Notebook  
- NumPy  
- Matplotlib  
- TensorFlow / Keras (or PyTorch, depending on implementation)

---

## Project Workflow
1. **Import Libraries**
   - Load deep learning, numerical, and visualization libraries.

2. **Load Dataset**
   - Load the MNIST dataset using built-in dataset loaders.

3. **Data Preprocessing**
   - Normalize pixel values to range [0, 1]
   - Reshape images into sequences (e.g., 28 timesteps × 28 features)
   - Prepare labels for classification

4. **RNN Model Architecture**
   - Input layer representing image sequences
   - Recurrent layer (SimpleRNN / LSTM / GRU)
   - Dense fully connected layer
   - Output layer with Softmax activation for 10 classes

5. **Model Compilation**
   - Optimizer: Adam
   - Loss function: Sparse Categorical Crossentropy
   - Evaluation metric: Accuracy

6. **Model Training**
   - Train the RNN using training data
   - Validate performance using test data
   - Track loss and accuracy across epochs

7. **Evaluation**
   - Evaluate model performance on test dataset
   - Plot training and validation accuracy and loss

8. **Prediction**
   - Predict digit classes for unseen samples
   - Visualize predicted vs actual labels

---

## Results
- The RNN successfully learns sequential patterns from image rows
- Performance is lower than CNN but demonstrates effective sequence modeling
- LSTM/GRU variants improve learning stability over basic RNNs

---

## How to Run the Notebook
1. Download or clone the project files
2. Install required dependencies
3. Launch Jupyter Notebook
4. Run `mnist rnn.ipynb`

```bash
pip install numpy matplotlib tensorflow
jupyter notebook
