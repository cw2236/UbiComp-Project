# -*- coding: utf-8 -*-
import numpy as np
import os

def check_data(file_path):
    """Check basic information of data files"""
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return
    
    if file_path.endswith('label_encoder.npy'):
        data = np.load(file_path, allow_pickle=True)
        print(f"\n=== {file_path} ===")
        print(f"Label classes: {data}")
    else:
        data = np.load(file_path)
        print(f"\n=== {file_path} ===")
        print(f"Data type: {data.dtype}")
        print(f"Data shape: {data.shape}")
        if len(data.shape) == 1:
            print(f"First 5 values: {data[:5]}")
        else:
            print(f"Shape of first 5 samples: {data[:5].shape}")
            print(f"Statistics of first sample:")
            print(f"  Min: {data[0].min():.4f}")
            print(f"  Max: {data[0].max():.4f}")
            print(f"  Mean: {data[0].mean():.4f}")
            print(f"  Std: {data[0].std():.4f}")

if __name__ == "__main__":
    files = [
        "processed_data/X_train.npy",
        "processed_data/X_test.npy",
        "processed_data/y_train.npy",
        "processed_data/y_test.npy",
        "processed_data/label_encoder.npy"
    ]
    
    for file in files:
        check_data(file) 