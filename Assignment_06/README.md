# Assignment 6: Traffic Sign Classification using CNN & GTSRB Dataset

**Student:** Chaitanya Pahilwan  
**Track:** Tata Technologies TechPulse Applied AI/ML Laboratory  
**Topic:** Deep Learning, Convolutional Neural Networks (CNN) & Computer Vision

---

## 1. Project Overview & Objective
This assignment develops and evaluates a Convolutional Neural Network (CNN) built with **PyTorch** to classify real-world traffic signs across 43 distinct categories from the **German Traffic Sign Recognition Benchmark (GTSRB)**.

---

## 2. Dataset & Balanced Sampling Strategy
- **Source:** Official GTSRB via `torchvision.datasets.GTSRB`.
- **Categories:** 43 classes (Speed limits, Stop, Yield, Warnings, Prohibitions, Directions).
- **Hardware Optimization:** Balanced subset sampler (`USE_SUBSET = True`) allocating 100 samples per class for training (4,300 total training images) and 25 samples per class for validation (1,075 images) to allow efficient local execution while ensuring rigorous class coverage.
- **Image Pipeline:** Resized to 32×32 pixels, converted to PyTorch Tensors, and normalized via ImageNet channel statistics ($\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225]$).

---

## 3. Network Architecture
A modular deep convolutional architecture:
- `Conv2d(3, 32, kernel_size=3, padding=1)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2, 2)`
- `Conv2d(32, 64, kernel_size=3, padding=1)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2, 2)`
- `Conv2d(64, 128, kernel_size=3, padding=1)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2, 2)`
- `Dropout(0.3)` + `Linear(128 * 4 * 4, 256)` + `ReLU` + `Dropout(0.4)` + `Linear(256, 43)`

---

## 4. Evaluation Highlights
- **Loss Function:** `nn.CrossEntropyLoss()`
- **Optimizer:** Adam (`lr=0.001`, `weight_decay=1e-4`)
- **Visual Diagnostics:** Training vs. validation loss/accuracy curves, multi-class confusion matrix, and visual prediction overlays on test samples.

---

## 5. Execution Instructions
```bash
cd Assignments/Assignment_06
jupyter notebook TTL_Assignment_06.ipynb
```
