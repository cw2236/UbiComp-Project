import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import GridSearchCV

def load_data():
    """Load training and testing data, and label encoder"""
    # Load data
    X_train = np.load('processed_data/X_train.npy')
    y_train = np.load('processed_data/y_train.npy')
    X_test = np.load('processed_data/X_test.npy')
    y_test = np.load('processed_data/y_test.npy')
    
    # Load label encoder
    label_encoder = np.load('processed_data/label_encoder.npy', allow_pickle=True).item()
    
    return X_train, y_train, X_test, y_test, label_encoder

def select_features(X_train, y_train, X_test, n_features=300):
    """Select top features based on Random Forest feature importance"""
    # Train a Random Forest model to get feature importance
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # Get feature importance
    importances = rf.feature_importances_
    
    # Get indices of top n_features
    top_indices = np.argsort(importances)[-n_features:]
    
    # Select top features
    X_train_selected = X_train[:, top_indices]
    X_test_selected = X_test[:, top_indices]
    
    print(f"Selected top {n_features} features based on Random Forest importance")
    
    return X_train_selected, X_test_selected, top_indices

def preprocess_data(X_train, X_test):
    """Data preprocessing: standardization and PCA dimensionality reduction"""
    print("\nPreprocessing data...")
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA dimensionality reduction
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"Number of features after dimensionality reduction: {X_train_pca.shape[1]}")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    return X_train_pca, X_test_pca

def perform_grid_search(X_train, y_train):
    """Perform grid search to find optimal Random Forest parameters"""
    print("\nPerforming grid search for optimal parameters...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }
    
    # Initialize Random Forest classifier
    rf = RandomForestClassifier(random_state=42)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
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
    """Train Random Forest model and evaluate its performance"""
    print("\nTraining Random Forest model...")
    
    # Perform grid search to find optimal parameters
    best_rf = perform_grid_search(X_train, y_train)
    
    # Train model with best parameters
    best_rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = best_rf.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
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
    os.makedirs('rf_results', exist_ok=True)
    
    # Save confusion matrix plot
    plt.savefig('rf_results/confusion_matrix.png')
    plt.close()
    
    # Plot feature importance
    feature_importance = best_rf.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.savefig('rf_results/feature_importance.png')
    plt.close()
    
    return best_rf

def main():
    # Load data
    X_train, y_train, X_test, y_test, label_encoder = load_data()
    
    # Preprocess data
    X_train_pca, X_test_pca = preprocess_data(X_train, X_test)
    
    # Train and evaluate model
    best_rf = train_and_evaluate(X_train_pca, y_train, X_test_pca, y_test, label_encoder)
    
    print("\nTraining completed. Results saved in 'rf_results' directory.")

if __name__ == "__main__":
    main() 