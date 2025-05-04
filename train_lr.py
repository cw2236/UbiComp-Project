import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load training and testing data"""
    X_train = np.load('processed_data/X_train.npy')
    y_train = np.load('processed_data/y_train.npy')
    X_test = np.load('processed_data/X_test.npy')
    y_test = np.load('processed_data/y_test.npy')
    label_encoder = np.load('processed_data/label_encoder.npy', allow_pickle=True).item()
    return X_train, y_train, X_test, y_test, label_encoder

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
    """Perform grid search to find optimal Logistic Regression parameters"""
    print("\nPerforming grid search for optimal parameters...")
    
    # Define parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['saga'],  # saga supports all penalties
        'l1_ratio': [0.2, 0.5, 0.8],  # for elasticnet
        'max_iter': [1000],
        'class_weight': ['balanced', None]
    }
    
    # Initialize Logistic Regression classifier
    lr = LogisticRegression(random_state=42)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=lr,
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
    """Train Logistic Regression model and evaluate its performance"""
    print("\nTraining Logistic Regression model...")
    
    # Perform grid search to find optimal parameters
    best_lr = perform_grid_search(X_train, y_train)
    
    # Train model with best parameters
    best_lr.fit(X_train, y_train)
    
    # Make predictions
    y_pred = best_lr.predict(X_test)
    
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
    os.makedirs('lr_results', exist_ok=True)
    
    # Save confusion matrix plot
    plt.savefig('lr_results/confusion_matrix.png')
    plt.close()
    
    # Plot feature coefficients
    if hasattr(best_lr, 'coef_'):
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(best_lr.coef_[0])), best_lr.coef_[0])
        plt.title('Feature Coefficients')
        plt.xlabel('Feature Index')
        plt.ylabel('Coefficient')
        plt.savefig('lr_results/feature_coefficients.png')
        plt.close()
    
    return best_lr

def main():
    # Load data
    X_train, y_train, X_test, y_test, label_encoder = load_data()
    
    # Preprocess data
    X_train_pca, X_test_pca = preprocess_data(X_train, X_test)
    
    # Train and evaluate model
    best_lr = train_and_evaluate(X_train_pca, y_train, X_test_pca, y_test, label_encoder)
    
    print("\nTraining completed. Results saved in 'lr_results' directory.")

if __name__ == "__main__":
    main() 