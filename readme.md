# 🔍 Fake News Detector - Machine Learning Project

A comprehensive machine learning project that detects fake news articles using both traditional ML algorithms and advanced deep learning models (BERT).


## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements and compares multiple machine learning approaches for detecting fake news:
- **5 Traditional ML Models**: Logistic Regression, SVM, Naive Bayes, Random Forest, Gradient Boosting
- **1 Deep Learning Model**: BERT (Bidirectional Encoder Representations from Transformers)

The system analyzes news article text and predicts whether it's fake or real news with high accuracy.

## ✨ Features

- ✅ **Multiple ML Models** - Compare 6 different algorithms
- ✅ **BERT Integration** - State-of-the-art NLP model
- ✅ **Comprehensive Metrics** - Accuracy, Precision, Recall, F1-Score
- ✅ **Visualizations** - Confusion matrices and comparison charts
- ✅ **Google Colab Support** - Ready to run in the cloud
- ✅ **CSV Data Loading** - Easy integration with your datasets
- ✅ **Model Persistence** - Save and load trained models

## 🛠️ Installation

### Option 1: Local Installation (Python 3.11 Required)

**⚠️ Important**: PyTorch doesn't fully support Python 3.13 yet. Use Python 3.11.

```bash
# Create conda environment with Python 3.11
conda create -n fake_news python=3.11 -y
conda activate fake_news

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Google Colab (Recommended for Beginners)

1. Upload the notebook to Google Colab
2. Upload your CSV files to Google Drive
3. Run all cells - dependencies are pre-installed!

## 📊 Dataset

### Required CSV Format

Your dataset should have two CSV files:
- `Fake.csv` - Contains fake news articles
- `True.csv` - Contains real news articles

**Expected columns**: `title`, `text`, `subject`, `date` (or similar)

### Example Data Structure

```csv
title,text,subject,date
"Breaking News Title","Article content here...",politics,2023-01-15
```

## 🚀 Usage

### 1. Prepare Your Data

Update the file paths in the script:

```python
FAKE_NEWS_FILE = 'path/to/Fake.csv'
TRUE_NEWS_FILE = 'path/to/True.csv'
```

### 2. Run the Script

```bash
python Fake_True_News.ipynb
```

### 3. For Google Colab

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Update paths
FAKE_NEWS_FILE = '/content/drive/MyDrive/data/Fake.csv'
TRUE_NEWS_FILE = '/content/drive/MyDrive/data/True.csv'
```

### 4. View Results

The script will:
- Train all 6 models
- Display metrics for each model
- Generate visualizations
- Save results as PNG file
- Show the best performing model

## 🤖 Models

### Traditional Machine Learning Models

| Model | Description | Training Time |
|-------|-------------|---------------|
| **Logistic Regression** | Linear classification model | Fast (~1-2 min) |
| **Support Vector Machine** | Maximum margin classifier | Medium (~2-3 min) |
| **Naive Bayes** | Probabilistic classifier | Very Fast (~30 sec) |
| **Random Forest** | Ensemble of decision trees | Medium (~3-5 min) |
| **Gradient Boosting** | Boosted ensemble model | Slow (~5-10 min) |

### Deep Learning Model

| Model | Description | Training Time |
|-------|-------------|---------------|
| **BERT** | Transformer-based model | Slow (~10-20 min) |

## 📈 Results

### Expected Performance

Typical accuracy ranges (depends on dataset quality):

- **Logistic Regression**: 88-92%
- **SVM**: 89-93%
- **Naive Bayes**: 85-89%
- **Random Forest**: 90-94%
- **Gradient Boosting**: 91-95%
- **BERT**: 93-97% ⭐

### Output Files

```
output.png    # Comprehensive visualizations
```

## 📁 Project Structure

```
fake-news-detector/
│
├── Fake_True_News.ipynb        # Main script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/                          # Data folder
│   ├── Fake.csv                   # Fake news dataset
│   └── True.csv                   # Real news dataset
│
├── results/                       # Output folder
│   └── fake_news_detection_results.png
```

## 🔧 Troubleshooting

### Common Issues

#### 1. PyTorch Import Error (Python 3.13)

```bash
# Error: Symbol not found in torch library
# Solution: Use Python 3.11
conda create -n fake_news python=3.11 -y
conda activate fake_news
pip install -r requirements.txt
```

#### 2. CUDA Out of Memory

```python
# Reduce batch size in the script
per_device_train_batch_size=8  # Change to 4 or 2
```

#### 3. CSV File Not Found

```python
# Check file paths
import os
print(os.path.exists('Fake.csv'))  # Should return True
```

#### 4. Slow Training

```python
# Use smaller dataset sample
sample_size = 5000  # Instead of full dataset
df = df.sample(n=sample_size, random_state=42)
```

### Performance Tips

1. **Use GPU**: If available, BERT will train 10x faster
2. **Reduce Dataset**: Use 5,000-10,000 samples for testing
3. **Lower Epochs**: Set `num_train_epochs=1` for quick testing
4. **Smaller Batch Size**: If memory issues occur

## 🎓 Learning Resources

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)


## 👨‍💻 Author

Created with ❤️ by Giorgi Paksashvili to detect fake news and promote information accuracy.

## 🙏 Acknowledgments

- Hugging Face for the transformers library
- Scikit-learn for ML algorithms
- PyTorch team for the deep learning framework
- The open-source community


## 📊 Quick Start Example

```python
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your CSV files
# - Fake.csv
# - True.csv

# 3. Run the script
python fake_news_detector.py

# 4. Check results
# - View console output for metrics
# - Open fake_news_detection_results.png for visualizations
```

## 🔬 Advanced Usage

### Use Trained Model for Predictions

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load saved model
model = AutoModelForSequenceClassification.from_pretrained('./saved_bert_model')
tokenizer = AutoTokenizer.from_pretrained('./saved_bert_model')

# Make prediction
text = "Your news article here..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1)

print("Fake" if prediction == 0 else "Real")
```

---