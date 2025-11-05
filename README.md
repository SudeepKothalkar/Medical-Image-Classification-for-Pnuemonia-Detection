# 🏥 Medical Image Classification Project - Complete Setup Guide

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Dataset Setup](#dataset-setup)
4. [Training Models](#training-models)
5. [Running the Web Application](#running-the-web-application)
6. [Project Structure](#project-structure)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This project implements an AI-powered pneumonia detection system using chest X-ray images. It includes:
- **3 Deep Learning Models**: ResNet50, VGG16, EfficientNetB0
- **Transfer Learning**: Pre-trained on ImageNet
- **Web Application**: Interactive interface for doctors
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, AUC-ROC

---

## 🔧 Installation

### Step 1: Clone or Create Project Directory

```bash
mkdir pneumonia_detection
cd pneumonia_detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

Create a `requirements.txt` file:

```txt
# Deep Learning
tensorflow==2.0.2
keras==2.14.0

# Data Science
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0

# Computer Vision
opencv-python==4.8.0.76
Pillow==10.0.0

# Web Application
streamlit==1.51.0

# Utilities
tqdm==4.67.1
```

Install packages:

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import streamlit; print('Streamlit installed successfully')"
```

---

## 📊 Dataset Setup

### Download Dataset from Kaggle

1. **Visit**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

2. **Download** the dataset (1.15 GB)

3. **Extract** to your project directory

### Expected Directory Structure

```
pneumonia_detection/
│
├── chest_xray/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── models/          # Created automatically
├── train_model.py   # Training script
├── app.py          # Web application
└── requirements.txt
```

### Alternative: Using Kaggle API

```bash
# Install Kaggle API
pip install kaggle

# Setup credentials (create kaggle.json from Kaggle account settings)
# Place in: ~/.kaggle/kaggle.json (Linux/Mac) or C:\Users\<user>\.kaggle\kaggle.json (Windows)

# Download dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Extract
unzip chest-xray-pneumonia.zip -d chest_xray/
```

### Dataset Statistics

- **Training Images**: ~5,216
  - Normal: 1,341
  - Pneumonia: 3,875

- **Validation Images**: 16
  - Normal: 8
  - Pneumonia: 8

- **Test Images**: 624
  - Normal: 234
  - Pneumonia: 390

---

## 🚀 Training Models

### Quick Start Training

```bash
python train_model.py
```

### What Happens During Training

1. **Data Loading**: Loads and preprocesses images
2. **Data Augmentation**: Applies transformations to training data
3. **Model Training**: Trains all 3 models sequentially
4. **Evaluation**: Tests on test set
5. **Saves**: Best models in `models/` directory

### Training Output

You'll see:
- Real-time training progress
- Validation metrics after each epoch
- Training history plots
- Confusion matrices
- ROC curves
- Model comparison table

### Expected Training Time

| Model | GPU Time | CPU Time |
|-------|----------|----------|
| ResNet50 | ~15 min | ~2 hours |
| VGG16 | ~20 min | ~3 hours |
| EfficientNetB0 | ~12 min | ~1.5 hours |

*Times vary based on hardware*

### Customizing Training

Edit the `Config` class in `train_model.py`:

```python
class Config:
    IMG_HEIGHT = 224        # Image dimensions
    IMG_WIDTH = 224
    BATCH_SIZE = 32         # Adjust based on GPU memory
    EPOCHS = 30             # Number of training epochs
    LEARNING_RATE = 0.0001  # Learning rate
```

---

## 🌐 Running the Web Application

### Step 1: Ensure Models Are Trained

Make sure you have trained models in the `models/` directory:
```
models/
├── ResNet50_best.h5
├── VGG16_best.h5
└── EfficientNetB0_best.h5
```

### Step 2: Launch Streamlit App

```bash
streamlit run app.py
```

### Step 3: Access the Application

The app will automatically open in your browser at:
```
http://localhost:8501
```

### Using the Web Application

1. **Select Model**: Choose from ResNet50, VGG16, or EfficientNetB0
2. **Upload Image**: Click "Browse files" and upload a chest X-ray
3. **Analyze**: Click "Analyze Image" button
4. **View Results**:
   - Diagnosis (Normal/Pneumonia)
   - Confidence score
   - Prediction probabilities
5. **History**: View all previous predictions in the History tab

---

## 📁 Project Structure

```
pneumonia_detection/
│
├── chest_xray/                 # Dataset directory
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/                     # Saved models
│   ├── ResNet50_best.h5
│   ├── VGG16_best.h5
│   └── EfficientNetB0_best.h5
│
├── train_model.py             # Training pipeline
├── app.py                     # Web application
├── requirements.txt           # Dependencies
│
├── *.png                      # Generated plots
└── model_comparison.csv       # Results comparison
```

---

## 🔍 Understanding the Results

### Evaluation Metrics

1. **Accuracy**: Percentage of correct predictions
   - Target: >90%

2. **Precision**: Of predicted pneumonia cases, how many are correct
   - Important to avoid false alarms

3. **Recall (Sensitivity)**: Of actual pneumonia cases, how many are detected
   - Critical in medical diagnosis - we don't want to miss cases

4. **AUC-ROC**: Model's ability to distinguish between classes
   - 0.5 = random, 1.0 = perfect
   - Target: >0.95

### Confusion Matrix

```
              Predicted
              Normal  Pneumonia
Actual Normal    TN      FP
     Pneumonia   FN      TP
```

- **TN (True Negative)**: Correctly identified healthy
- **TP (True Positive)**: Correctly identified pneumonia
- **FN (False Negative)**: Missed pneumonia (dangerous!)
- **FP (False Positive)**: False alarm (concerning but less critical)

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory Error

**Problem**: GPU/RAM runs out of memory during training

**Solutions**:
```python
# Reduce batch size
BATCH_SIZE = 16  # or even 8

# Use mixed precision training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

#### 2. Model Files Not Found

**Problem**: Web app can't find trained models

**Solution**:
```bash
# Check if models exist
ls models/

# If not, train models first
python train_model.py
```

#### 3. Dataset Not Found

**Problem**: Training script can't find dataset

**Solution**:
```python
# Update paths in Config class
DATASET_PATH = 'path/to/your/chest_xray'
```

#### 4. CUDA/GPU Issues

**Problem**: TensorFlow not detecting GPU

**Solutions**:
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install CUDA-compatible TensorFlow
pip install tensorflow-gpu==2.14.0
```

#### 5. Streamlit Port Already in Use

**Problem**: Port 8501 is occupied

**Solution**:
```bash
# Use different port
streamlit run app.py --server.port 8502
```

---

## 🎓 Tips for Best Performance

### Data Preprocessing
- Ensure images are clear and properly centered
- Use consistent image quality
- Remove any text overlays or markers

### Training
- Start with small EPOCHS (5-10) to test setup
- Monitor validation loss to avoid overfitting
- Use early stopping to save time

### Deployment
- Test with various image qualities
- Keep model files backed up
- Monitor prediction confidence scores

### Medical Context
- Always verify predictions with medical professionals
- Use as a screening tool, not final diagnosis
- Document all predictions for audit trails

---

## 📚 Additional Resources

### Learning Materials
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Transfer Learning Guide](https://keras.io/guides/transfer_learning/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Dataset Information
- [Original Paper](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)
- [Kaggle Competition](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Research Papers
- ResNet: "Deep Residual Learning for Image Recognition"
- VGG: "Very Deep Convolutional Networks"
- EfficientNet: "EfficientNet: Rethinking Model Scaling"

---

## 🤝 Contributing

To extend this project:

1. **Add More Models**: Implement DenseNet, Inception, etc.
2. **Improve UI**: Enhance Streamlit interface
3. **Add Features**: 
   - Batch processing
   - Report generation
   - DICOM support
4. **Optimize**: Model quantization, pruning

---

## ⚠️ Important Disclaimer

**This is an educational/research project and should NOT be used for actual medical diagnosis without:**
- Validation by medical professionals
- Regulatory approval (FDA, CE marking, etc.)
- Clinical trials
- Proper quality assurance systems

Always consult qualified healthcare professionals for medical decisions.

---

## 📞 Support

For issues:
1. Check troubleshooting section
2. Verify installation steps
3. Check TensorFlow/Keras compatibility
4. Review error logs carefully

---

## 🎉 Next Steps

1. ✅ Install dependencies
2. ✅ Download dataset
3. ✅ Train models
4. ✅ Run web application
5. ✅ Test with sample images
6. 🚀 Deploy to production (with proper medical validation)

**Happy coding! 🚀**
