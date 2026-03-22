# DIGIAI 🧠✍️

A real-time handwritten digit recognition desktop app built with **PyGame** and **PyTorch**.

DIGIAI allows you to draw digits (**0–9**) directly on a canvas, and a trained **Convolutional Neural Network (CNN)** instantly predicts the number in real time using a model trained on the **MNIST dataset**.

This project combines:

- **Deep Learning**
- **Computer Vision preprocessing**
- **Interactive desktop UI**
- **Real-time inference**

---

## 📌 Overview

DIGIAI is a simple but powerful machine learning project that demonstrates how to connect:

- a **trained neural network model**
- with a **live drawing interface**
- and perform **real-time predictions** as the user draws.

The app continuously preprocesses the drawing to match MNIST-style inputs, then runs inference every few milliseconds to update the prediction dynamically.

---

## ✨ Features

- 🎨 Draw digits freely with the mouse
- 🤖 Real-time handwritten digit prediction
- 📊 Confidence percentage displayed with each prediction
- 🧠 Custom CNN trained on MNIST
- ⚡ Live inference while drawing
- 🧹 Clear canvas instantly
- 📜 Built-in console logging for preprocessing and predictions
- 🖥️ Lightweight desktop app using PyGame
- 🧪 Great beginner-to-intermediate AI/ML portfolio project

---

## 🛠️ Tech Stack

- **Python 3**
- **PyTorch**
- **Torchvision**
- **PyGame**
- **NumPy**
- **Pillow (PIL)**

---

## 🧠 Model Architecture

The project uses a custom CNN called `DigitNet`.

### Network Structure

- **Conv2D**: `1 → 32`, kernel size `3x3`
- **ReLU**
- **MaxPool2D**
- **Conv2D**: `32 → 64`, kernel size `3x3`
- **ReLU**
- **MaxPool2D**
- **Flatten**
- **Linear**: `64 * 5 * 5 → 128`
- **ReLU**
- **Linear**: `128 → 10`

### Output

- Predicts one digit from **0 to 9**
- Uses **Softmax** to calculate class probabilities
- Displays:
  - predicted digit
  - confidence percentage

---

## 📂 Project Structure

```bash id="digiai-tree-001"
DIGIAI/
│
├── pygame_app.py        # Main PyGame application (draw + predict)
├── train_model0.py      # CNN training script using MNIST
├── digit_cnn.pth        # Saved trained model weights (generated after training)
├── data/                # MNIST dataset folder (auto-downloaded)
├── requirements.txt     # rquired libraries for this project
└── README.md
```
