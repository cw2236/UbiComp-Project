# Smart Pin - Acoustic Sensing for Smart Home Control

This project implements various machine learning and deep learning models for smart home control using acoustic sensing with a pin on the chest.

User data are excluded in this Github commit. 

## Data Processing 1 (self-implemented)

The data processing pipeline (`process_data.py`) includes the following steps:

1. **Data Loading**:
   - Loads FMCW radar profiles from .npy files
   - Reads action labels from .txt files
   - Organizes data by user sessions

2. **Time Alignment**:
   - Extracts action timestamps from audio files
   - Aligns radar data with action labels
   - Creates synchronized data segments

3. **Feature Extraction**:
   - Implements sliding window approach (window size: 15 frames, stride: 7 frames)
   - Extracts features from each window:
     - Statistical features (mean, std, min, max)
     - Frequency domain features
     - Time domain features
   - Generates feature vectors for each action segment

4. **Dataset Splitting**:
   - Splits data by user groups to prevent data leakage
   - Last user in each group assigned to test set
   - Remaining users assigned to training set
   - Maintains balanced class distribution

5. **Data Augmentation**:
   - Applies random noise
   - Time warping
   - Amplitude scaling
   - Generates synthetic samples for minority classes

6. **Output**:
   - Saves processed features and labels as .npy files
   - Generates data statistics and visualizations
   - Creates train/test split indices

## Data Processing 2 (provided by our mentor)

The data processing pipeline (`data-loading.ipynb`) includes the following steps:

1. **Data Loading**:
   - Loads pre-processed feature and label arrays from `.npy` files:
     - `features.npy`: 3D array containing feature vectors per sample
     - `labels.npy`: 1D array of corresponding class labels

2. **Label Distribution Visualization**:
   - Plots the distribution of action labels using `matplotlib`
   - Highlights potential class imbalance
   - Saves the resulting figure as an image for reference

3. **Dimensionality Inspection**:
   - Prints shapes of the feature and label arrays
   - Ensures compatibility with model input requirements

4. **Sample Verification**:
   - Displays a sample feature array and its associated label
   - Allows manual inspection of data correctness

5. **Output**:
   - Generates and saves a class distribution plot
   - Verifies the structure of the loaded data
   - Prepares the dataset for training and evaluation

     
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
├── process_data.py         # Data preprocessing 1 script
├── data-loading.ipynb      # Data preprocessing 2 notebook 
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

3. Preprocess data:
```
Run chunks in data-loading.ipynb after setting config in dataset folder generated in the data collection and data preparation process
```
5. Train models:
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
