# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load processed data"""
    X_train = np.load('processed_data/X_train.npy')
    X_test = np.load('processed_data/X_test.npy')
    y_train = np.load('processed_data/y_train.npy')
    y_test = np.load('processed_data/y_test.npy')
    label_names = np.load('processed_data/label_encoder.npy', allow_pickle=True)
    return X_train, X_test, y_train, y_test, label_names

def train_and_evaluate():
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test, label_names = load_data()
    
    # Data standardization
    print("Standardizing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN model
    print("Training KNN model...")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = knn.predict(X_test_scaled)
    
    # Evaluate model
    print("\n=== Model Evaluation Report ===")
    print(classification_report(y_test, y_pred, target_names=label_names))
    
    # Calculate and plot confusion matrix
    print("\nPlotting confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    # Create output directory
    os.makedirs('model_results', exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('model_results/confusion_matrix.png')
    
    # Save model performance metrics
    train_score = knn.score(X_train_scaled, y_train)
    test_score = knn.score(X_test_scaled, y_test)
    print(f"\nTraining accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")

if __name__ == "__main__":
    train_and_evaluate() 