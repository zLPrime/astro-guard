# Astronomical Object Classification Pipeline

Hierarchical machine learning system for classifying variable stars using time series data.

## 📌 Project Overview

The system implements:
- **Automated processing** of raw data from S3 storage
- **Feature extraction** (17 FATS statistical characteristics)
- **Deep learning** on time series (1D CNN)
- **Hierarchical classification** (2 levels: main classes → subclasses)
- **Threshold-based logic** with automatic fallback to "Other" classes

## 🛠 Technology Stack
- **Storage**: Yandex Cloud S3 (S3-compatible)
- **ML Framework**: TensorFlow/Keras, LightGBM
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Integration**: Boto3 for S3 operations

## 📦 Installation
git clone https://github.com/your-repo/astronomy-classification.git
cd astronomy-classification