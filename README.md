# Smart Pin - Acoustic Sensing for Smart Home Control

This project implements various machine learning and deep learning models for smart home control using acoustic sensing with a pin on the chest.

User data are excluded in this Github commit. 

## Models Implemented

1. Traditional Machine Learning Models:
   - K-Nearest Neighbors (KNN)
   - Logistic Regression (LR)
   - Random Forest (RF)
   - Support Vector Machine (SVM)
   - Stacking Ensemble

2. Deep Learning Models:
   - LSTM
   - CNN
   - Deep Neural Network (DNN)

## Project Structure

```
.
├── process_data.py          # Data preprocessing script
├── train_knn.py            # KNN model training
├── train_lr.py             # Logistic Regression training
├── train_rf.py             # Random Forest training
├── train_svm.py            # SVM model training
├── train_stacking.py       # Stacking ensemble training
├── train_lstm.py           # LSTM model training
├── train_cnn.py            # CNN model training
├── train_dl.py             # MLP training
└── requirements.txt        # Project dependencies
```

## Features

- Data preprocessing and feature extraction
- Hyperparameter tuning using GridSearchCV
- Model evaluation with confusion matrices and ROC curves
- Cross-validation for robust model assessment

  
## Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Preprocess data:
```bash
python process_data.py
```

3. Train models:
```bash
python train_knn.py    # Train KNN model
python train_lr.py     # Train Logistic Regression
python train_rf.py     # Train Random Forest
python train_svm.py    # Train SVM
python train_stacking.py  # Train Stacking Ensemble
python train_lstm.py   # Train LSTM
python train_cnn.py    # Train CNN
python train_dl.py     # Train MLP
```

## Model Evaluation

Each model training script generates:
- Confusion matrix
- Classification report
- ROC curves (where applicable)
- Model-specific visualizations
