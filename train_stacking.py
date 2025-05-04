import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import cross_val_predict, KFold
from scipy import stats
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class StackingEnsemble:
    def __init__(self, base_models, meta_classifier, n_classes):
        self.base_models = base_models
        self.meta_classifier = meta_classifier
        self.n_classes = n_classes
    
    def predict(self, X):
        meta_features = np.zeros((X.shape[0], len(self.base_models), self.n_classes))
        for i, model in enumerate(self.base_models.values()):
            meta_features[:, i, :] = model.predict_proba(X)
        meta_features = meta_features.reshape(X.shape[0], -1)
        return self.meta_classifier.predict(meta_features)
    
    def predict_proba(self, X):
        meta_features = np.zeros((X.shape[0], len(self.base_models), self.n_classes))
        for i, model in enumerate(self.base_models.values()):
            meta_features[:, i, :] = model.predict_proba(X)
        meta_features = meta_features.reshape(X.shape[0], -1)
        return self.meta_classifier.predict_proba(meta_features)

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

def select_features(X, y, n_features=200):
    """Select top features based on Random Forest feature importance"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    print(f"Selected top {n_features} features based on Random Forest importance")
    return indices

def preprocess_data(X_train, X_test):
    """Data preprocessing: standardization"""
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def get_base_models():
    """Create and return base models for stacking"""
    models = {
        'rf': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            learning_rate=0.1
        ),
        'et': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    }
    return models

def train_stacking(X, y, X_test, y_test, feature_indices):
    # Select features
    X_selected = X[:, feature_indices]
    X_test_selected = X_test[:, feature_indices]
    
    # Get number of classes
    n_classes = len(np.unique(y))
    class_names = np.unique(y)
    
    # Initialize base models
    base_models = {
        'rf': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, 
                                   min_samples_leaf=2, max_features='sqrt', class_weight='balanced'),
        'gb': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                       min_samples_split=5, min_samples_leaf=2, subsample=0.8),
        'et': ExtraTreesClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
                                 min_samples_leaf=2, max_features='sqrt', class_weight='balanced')
    }
    
    # Initialize meta-learner
    meta_learner = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
                                        min_samples_leaf=2, max_features='sqrt', class_weight='balanced')
    
    # Initialize stacking ensemble
    stacking = StackingEnsemble(base_models, meta_learner, n_classes)
    
    # Train base models and generate meta-features
    print("\nTraining base models...")
    for name, model in base_models.items():
        print(f"\nTraining {name}...")
        model.fit(X_selected, y)
    
    # Generate meta-features using cross-validation
    print("\nGenerating meta-features...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize arrays for meta-features
    meta_features_train = np.zeros((X_selected.shape[0], len(base_models), n_classes))
    meta_features_test = np.zeros((X_test_selected.shape[0], len(base_models), n_classes))
    
    # Generate meta-features for training set
    for i, (train_idx, val_idx) in enumerate(kf.split(X_selected)):
        X_train_fold = X_selected[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X_selected[val_idx]
        
        for j, (name, model) in enumerate(base_models.items()):
            # Train on training fold
            model.fit(X_train_fold, y_train_fold)
            # Predict probabilities on validation fold
            meta_features_train[val_idx, j, :] = model.predict_proba(X_val_fold)
    
    # Generate meta-features for test set
    for j, (name, model) in enumerate(base_models.items()):
        meta_features_test[:, j, :] = model.predict_proba(X_test_selected)
    
    # Reshape meta-features
    meta_features_train = meta_features_train.reshape(X_selected.shape[0], -1)
    meta_features_test = meta_features_test.reshape(X_test_selected.shape[0], -1)
    
    # Train meta-learner
    print("\nTraining meta-learner...")
    meta_learner.fit(meta_features_train, y)
    
    # Make predictions
    y_pred = meta_learner.predict(meta_features_test)
    y_proba = meta_learner.predict_proba(meta_features_test)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create output directory if it doesn't exist
    os.makedirs('model_evaluation', exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('model_evaluation/stacking_confusion_matrix.png')
    plt.close()
    
    # Plot ROC curves for stacking
    plt.figure(figsize=(10, 8))
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Stacking Ensemble')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('model_evaluation/stacking_roc_curves.png')
    plt.close()
    
    return stacking

def main():
    # Load data
    X_train, y_train, X_test, y_test, label_encoder = load_data()
    
    # Select top features
    feature_indices = select_features(X_train, y_train)
    
    # Train stacking model
    train_stacking(X_train, y_train, X_test, y_test, feature_indices)

if __name__ == "__main__":
    main() 