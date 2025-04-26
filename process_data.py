# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_ground_truth(file_path):
    """Load ground truth data"""
    df = pd.read_csv(file_path, header=None, 
                    names=['id', 'start_time', 'end_time', 'action'])
    return df

def load_sensor_data(file_path):
    """Load sensor data"""
    data = np.load(file_path)
    return data

def augment_data(X, n_augment=5):
    """Data augmentation using random noise"""
    augmented_data = []
    for sample in X:
        augmented_data.append(sample)  # Original sample
        for _ in range(n_augment):
            # Add random noise
            noise = np.random.normal(0, 0.1, sample.shape)
            augmented_data.append(sample + noise)
    return np.array(augmented_data)

def process_data(ground_truth_path, sensor_data_path, output_dir='processed_data'):
    """Process data and generate training and test sets"""
    # Load data
    ground_truth = load_ground_truth(ground_truth_path)
    sensor_data = load_sensor_data(sensor_data_path)
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate number of samples per action
    samples_per_action = sensor_data.shape[0] // len(ground_truth)
    
    # Reshape data to match number of labels
    X = np.array([sensor_data[i:i+samples_per_action].mean(axis=0) 
                 for i in range(0, sensor_data.shape[0], samples_per_action)])
    
    # Encode action labels
    label_encoder = LabelEncoder()
    ground_truth['action_encoded'] = label_encoder.fit_transform(ground_truth['action'])
    y = ground_truth['action_encoded'].values
    
    # Data augmentation
    X_augmented = augment_data(X)
    y_augmented = np.repeat(y, X_augmented.shape[0] // len(y))
    
    print("Original data shape:", X.shape)
    print("Augmented data shape:", X_augmented.shape)
    print("Augmented labels shape:", y_augmented.shape)
    
    # Save label encoder
    np.save(os.path.join(output_dir, 'label_encoder.npy'), label_encoder.classes_)
    
    # Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_augmented, y_augmented, test_size=0.2, random_state=42, stratify=y_augmented
    )
    
    # Save processed data
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    print("Data processing completed and saved to directory:", output_dir)
    print("Training set size:", X_train.shape)
    print("Test set size:", X_test.shape)
    print("Action classes:", label_encoder.classes_)

if __name__ == "__main__":
    ground_truth_path = "record_20250420_151910_752866_records.txt"
    sensor_data_path = "audio000_fmcw_16bit_diff_profiles.npy"
    process_data(ground_truth_path, sensor_data_path) 