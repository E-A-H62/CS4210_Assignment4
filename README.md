# Handwritten Digit Classification Using CNN

## Description

This project implements a Convolutional Neural Network (CNN) to classify grayscale images of handwritten digits (0–9), each resized to 32x32 pixels. Using TensorFlow and Keras, the CNN is trained on a dataset of digit images and evaluated on a separate test set. The workflow involves:

- Loading and preprocessing the image data
- Building a CNN architecture
- Training the model on labeled data
- Evaluating accuracy on unseen test data

---

## Model Architecture

The CNN used in this project follows this architecture:

1. **Conv2D Layer** – 32 filters, 3×3 kernel, ReLU activation  
2. **MaxPooling2D** – 2×2 pooling  
3. **Flatten Layer** – converts 2D features into a 1D vector  
4. **Dense Layer** – 64 neurons, ReLU activation  
5. **Output Layer** – 10 neurons (for digits 0–9), softmax activation  

---

## Dataset Structure

The dataset consists of grayscale `.png` images organized into two folders:

```

digit\_dataset/
├── train/
│   ├── 0\_image1.png
│   ├── 1\_image2.png
│   └── ...
└── test/
├── 5\_image1.png
├── 8\_image2.png
└── ...

```

Each filename starts with the digit label (`0`–`9`), which is extracted and used as the image’s class during training.

---

## Features

- Custom image loading and preprocessing using `PIL` and `NumPy`
- CNN model building using `TensorFlow/Keras`
- Automatic normalization and reshaping of image data
- Training using SGD optimizer and cross-entropy loss
- Prints test set accuracy after training

---

## Example Output

```

Epoch 10/10
100/100 \[==============================] - 1s 6ms/step - loss: 0.0587 - accuracy: 0.9840 - val\_loss: 0.0925 - val\_accuracy: 0.9730
313/313 \[==============================] - 1s 2ms/step - loss: 0.0925 - accuracy: 0.9730
Test accuracy: 0.97

````

---

## Requirements

- Python 3.x
- TensorFlow (>=2.x)
- NumPy
- Pillow (PIL)

### Install Dependencies

```bash
pip install tensorflow numpy pillow
````

---

## How to Run

1. Ensure your folder structure matches the expected dataset layout.
2. Run the training script:

```bash
python cnn.py
```
