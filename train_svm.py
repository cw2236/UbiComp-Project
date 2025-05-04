import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def load_data():
    print("Loading data...\n")
    X_train = np.load('processed_data/X_train.npy')
    y_train = np.load('processed_data/y_train.npy')
    X_test = np.load('processed_data/X_test.npy')
    y_test = np.load('processed_data/y_test.npy')
    
    # Load label encoder
    label_encoder = np.load('processed_data/label_encoder.npy', allow_pickle=True).item()
    
    return X_train, y_train, X_test, y_test, label_encoder

def preprocess_data(X_train, X_test):
    print("Preprocessing data...\n")
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"Number of features after dimensionality reduction: {X_train_pca.shape[1]}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum()*100:.2f}%\n")
    
    return X_train_pca, X_test_pca

def perform_grid_search(X_train, y_train):
    """Perform grid search to find optimal SVM parameters"""
    print("\nPerforming grid search for optimal parameters...")
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'degree': [2, 3, 4],  # for poly kernel
        'class_weight': ['balanced', None]
    }
    
    # Initialize SVM classifier
    svm = SVC(random_state=42, probability=True)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=5,  # 5-fold cross validation
        scoring='accuracy',
        n_jobs=-1,  # Use all available cores
        verbose=2
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Print results
    print("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_and_evaluate(X_train, y_train, X_test, y_test, label_encoder):
    """Train SVM model and evaluate its performance"""
    print("\nTraining SVM model...")
    
    # Perform grid search to find optimal parameters
    best_svm = perform_grid_search(X_train, y_train)
    
    # Train model with best parameters
    best_svm.fit(X_train, y_train)
    
    # Make predictions
    y_pred = best_svm.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Create output directory if it doesn't exist
    os.makedirs('svm_results', exist_ok=True)
    
    # Save confusion matrix plot
    plt.savefig('svm_results/confusion_matrix.png')
    plt.close()
    
    # Plot decision boundaries for first two features
    if X_test.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        X_2d = X_test[:, :2]
        y_2d = y_test
        
        # Create mesh grid
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        # Predict on mesh grid
        Z = best_svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, alpha=0.8)
        plt.title('Decision Boundary (First Two Features)')
        plt.xlabel('First Feature')
        plt.ylabel('Second Feature')
        plt.savefig('svm_results/decision_boundary.png')
        plt.close()
    
    return best_svm

def main():
    # Load data
    X_train, y_train, X_test, y_test, label_encoder = load_data()
    
    # Preprocess data
    X_train_pca, X_test_pca = preprocess_data(X_train, X_test)
    
    # Train and evaluate model
    best_svm = train_and_evaluate(X_train_pca, y_train, X_test_pca, y_test, label_encoder)
    
    print("\nTraining completed. Results saved in 'svm_results' directory.")

if __name__ == "__main__":
    main() 