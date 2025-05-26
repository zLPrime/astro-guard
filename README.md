# Astronomical Object Classification Pipeline

Machine learning system for classifying variable stars using time series data.

## 📌 Project Overview

The system implements:
- **Automated processing** of raw data from S3 storage
- **Feature extraction** (TSFresh statistical characteristics)
- **Deep learning** on time series (1D CNN)

## 🛠 Technology Stack
- **Storage**: Yandex Cloud S3 (S3-compatible)
- **ML Framework**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Integration**: Boto3 for S3 operations

## 📦 Installation
git clone https://github.com/your-repo/astronomy-classification.git