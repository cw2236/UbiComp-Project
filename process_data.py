# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import re

def load_ground_truth(file_path):
    """Load ground truth data from CSV file
    
    Args:
        file_path (str): Path to the CSV file containing ground truth data
        Format: id,start_time,end_time,action (comma-separated)
    
    Returns:
        pd.DataFrame: DataFrame containing ground truth data
    """
    df = pd.read_csv(file_path, header=None, 
                    names=['id', 'start_time', 'end_time', 'action'])
    return df

def load_sensor_data(file_path):
    """Load sensor data"""
    data = np.load(file_path)
    return data

def extract_action_data(sensor_data, frame_times, start_time, end_time):
    """Extract sensor data for a specific action based on start and end times
    
    Args:
        sensor_data: The full sensor data array
        frame_times: Array of frame timestamps
        start_time: Action start time from ground truth
        end_time: Action end time from ground truth
    
    Returns:
        np.array: Extracted and processed sensor data for the action
    """
    # Find indices where frame times fall within the action interval
    mask = (frame_times >= start_time) & (frame_times <= end_time)
    action_indices = np.where(mask)[0]
    
    if len(action_indices) == 0:
        print("Warning: No frames found between {} and {}".format(start_time, end_time))
        return None
    
    print("Found {} frames between {} and {}".format(len(action_indices), start_time, end_time))
    
    # Extract the corresponding sensor data
    action_data = sensor_data[action_indices]
    
    # Take mean across time dimension to get fixed-size feature vector
    mean_data = np.mean(action_data, axis=0)
    print("Extracted data shape: {}, Mean data shape: {}".format(action_data.shape, mean_data.shape))
    
    return mean_data

def find_files(user_dir):
    """Find all necessary files in user directory"""
    # Find sensor data file
    sensor_files = glob.glob(os.path.join(user_dir, "*fmcw_16bit_diff_profiles.npy"))
    if not sensor_files:
        raise FileNotFoundError("No sensor data file found in {}".format(user_dir))
    sensor_file = sensor_files[0]
    
    # Find ground truth file
    gt_files = glob.glob(os.path.join(user_dir, "record_*_records.txt"))
    if not gt_files:
        raise FileNotFoundError("No ground truth file found in {}".format(user_dir))
    gt_file = gt_files[0]
    
    # Find frame time file (same prefix as ground truth file)
    gt_prefix = gt_file[:-12]  # Remove '_records.txt'
    frame_time_file = "{}_frame_time.txt".format(gt_prefix)
    if not os.path.exists(frame_time_file):
        raise FileNotFoundError("No frame time file found: {}".format(frame_time_file))
    
    return sensor_file, gt_file, frame_time_file

def create_sliding_windows(data, window_size=10, stride=5):
    """Create time series features using sliding windows
    
    Args:
        data: Array with shape (n_frames, n_features)
        window_size: Window size
        stride: Sliding step size
        
    Returns:
        windows: Array with shape (n_windows, window_size, n_features)
    """
    n_frames, n_features = data.shape
    n_windows = ((n_frames - window_size) // stride) + 1
    windows = np.zeros((n_windows, window_size, n_features))
    
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        windows[i] = data[start_idx:end_idx]
    
    return windows

def extract_features(window):
    """Extract statistical features from window data
    
    Args:
        window: Array with shape (window_size, n_features)
        
    Returns:
        features: Array containing statistical features
    """
    # Basic statistical features
    mean_features = np.mean(window, axis=0)
    std_features = np.std(window, axis=0)
    max_features = np.max(window, axis=0)
    min_features = np.min(window, axis=0)
    
    # Advanced statistical features
    median_features = np.median(window, axis=0)
    skew_features = np.zeros_like(mean_features)
    kurt_features = np.zeros_like(mean_features)
    for i in range(window.shape[1]):
        skew_features[i] = np.nan_to_num(np.mean(((window[:, i] - mean_features[i]) / std_features[i]) ** 3)) if std_features[i] != 0 else 0
        kurt_features[i] = np.nan_to_num(np.mean(((window[:, i] - mean_features[i]) / std_features[i]) ** 4)) if std_features[i] != 0 else 0
    
    # Time domain features
    zero_crossings = np.sum(np.diff(np.signbit(window), axis=0), axis=0)
    peak_to_peak = max_features - min_features
    rms_features = np.sqrt(np.mean(np.square(window), axis=0))
    
    # Combine all features
    features = np.concatenate([
        mean_features,      # Mean
        std_features,      # Standard deviation
        max_features,      # Maximum
        min_features,      # Minimum
        median_features,   # Median
        skew_features,     # Skewness
        kurt_features,     # Kurtosis
        zero_crossings,    # Zero crossing count
        peak_to_peak,      # Peak-to-peak value
        rms_features       # Root mean square
    ])
    
    return features

def process_user_data(user_folder, min_feature_dim, window_size=50, stride=5):
    """Process data for a single user using sliding windows to extract features
    
    Args:
        user_folder: Path to user data folder
        min_feature_dim: Minimum feature dimension
        window_size: Window size, default 50
        stride: Sliding step size, default 5
        
    Returns:
        tuple: (features, labels) or None (if processing fails)
    """
    # Get user ID
    user_id = os.path.basename(user_folder)
    print("\nProcessing data for user {}...".format(user_id))
    
    # Adjust window size based on user ID
    user_num = int(''.join(filter(str.isdigit, user_id)))
    if user_num <= 7:  # Data for users 1-7 is shorter
        window_size = 15  # Use smaller window
        stride = 2
    else:  # Data for users 8 and above is longer
        window_size = 20  # Use larger window
        stride = 3
    
    print("Using window size: {}, stride: {}".format(window_size, stride))
    
    # Find all necessary files
    try:
        sensor_file, gt_file, frame_time_file = find_files(user_folder)
    except FileNotFoundError as e:
        print("Error: {}".format(e))
        return None
    
    # Read ground truth file
    gt_data = load_ground_truth(gt_file)
    
    # Read frame time file
    frame_times = np.loadtxt(frame_time_file)
    print("Frame times range: {} to {}".format(frame_times[0], frame_times[-1]))
    
    # Read sensor data
    sensor_data = load_sensor_data(sensor_file)
    print("Original data shape: {}".format(sensor_data.shape))
    
    # Use only minimum dimension features
    sensor_data = sensor_data[:, :min_feature_dim+1]  # +1 because first column is timestamp
    print("Truncated data shape: {}".format(sensor_data.shape))
    
    all_features = []
    all_labels = []
    
    # Process each action
    for _, row in gt_data.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        label = row['action']
        
        # Find frames corresponding to the time period
        start_idx = np.searchsorted(frame_times, start_time)
        end_idx = np.searchsorted(frame_times, end_time)
        
        # Handle cases where time range exceeds data range
        if start_idx >= len(frame_times):
            print(f"Warning: Action '{label}' start time {start_time} is after the last frame time {frame_times[-1]}. Skipping action.")
            continue
            
        if end_idx > len(frame_times):
            print(f"Warning: Action '{label}' end time {end_time} exceeds the last frame time {frame_times[-1]}. Trimming to last available frame.")
            end_idx = len(frame_times)
            
        # Extract all frames in this time period
        frames = sensor_data[start_idx:end_idx, 1:]  # Exclude timestamp column
        
        if len(frames) < window_size:
            print("Warning: Action {} has fewer frames ({}) than window size ({})".format(label, len(frames), window_size))
            continue
        
        # Calculate stride to generate approximately 11 windows
        n_frames = len(frames)
        if n_frames > window_size:
            stride = max(1, (n_frames - window_size) // 10)  # 10 steps to generate 11 windows
            print("Adjusted stride: {}".format(stride))
        
        # Create sliding windows
        windows = create_sliding_windows(frames, window_size, stride)
        print("Action {}: From {} to {}, created {} windows".format(label, start_time, end_time, len(windows)))
        
        # Extract features from each window
        for window in windows:
            features = extract_features(window)
            all_features.append(features)
            all_labels.append(label)
    
    if not all_features:
        print("Warning: No features extracted for user {}".format(user_id))
        return None
    
    return all_features, all_labels

def process_data(pilot_dir='pilot', output_dir='processed_data', window_size=50, stride=5):
    """Process all user data and save
    
    Args:
        pilot_dir: Directory containing all user data
        output_dir: Output directory
        window_size: Window size
        stride: Sliding step size
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all user folders
    user_folders = [f for f in os.listdir(pilot_dir) if os.path.isdir(os.path.join(pilot_dir, f))]
    
    # Group users by letter part
    user_groups = {}
    for user in user_folders:
        # Separate letter and number parts
        match = re.match(r'([a-zA-Z]+)(\d+)', user)
        if match:
            letter_part, num_part = match.groups()
            if letter_part not in user_groups:
                user_groups[letter_part] = []
            user_groups[letter_part].append((user, int(num_part)))
    
    # Sort users within each group by number
    for group in user_groups:
        user_groups[group].sort(key=lambda x: x[1])
    
    # Assign training and testing sets
    training_users = []
    testing_users = []
    for group in user_groups:
        users = [u[0] for u in user_groups[group]]  # Get sorted usernames
        if len(users) > 1:
            training_users.extend(users[:-1])  # All but last user for training
            testing_users.append(users[-1])    # Last user for testing
        else:
            training_users.extend(users)  # If only one user, add to training set
    
    print("Training users:", training_users)
    print("Testing users:", testing_users)
    
    # Find minimum feature dimension across all user data
    min_feature_dim = float('inf')
    for user in user_folders:
        user_folder = os.path.join(pilot_dir, user)
        npy_files = glob.glob(os.path.join(user_folder, "*fmcw_16bit_diff_profiles.npy"))
        for npy_file in npy_files:
            data = np.load(npy_file)
            min_feature_dim = min(min_feature_dim, data.shape[1]-1)  # -1 because first column is timestamp
    
    print("Minimum feature dimension: {}".format(min_feature_dim))
    
    # Process training set user data
    X_train = []
    y_train = []
    for user in training_users:
        user_folder = os.path.join(pilot_dir, user)
        if not os.path.exists(user_folder):
            print("Warning: User folder {} not found".format(user_folder))
            continue
        result = process_user_data(user_folder, min_feature_dim, window_size, stride)
        if result is not None:
            features, labels = result
            X_train.extend(features)
            y_train.extend(labels)
    
    # Process testing set user data
    X_test = []
    y_test = []
    for user in testing_users:
        user_folder = os.path.join(pilot_dir, user)
        if not os.path.exists(user_folder):
            print("Warning: User folder {} not found".format(user_folder))
            continue
        result = process_user_data(user_folder, min_feature_dim, window_size, stride)
        if result is not None:
            features, labels = result
            X_test.extend(features)
            y_test.extend(labels)
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Create and save label encoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Print all possible action labels
    print("\nAll action labels:", label_encoder.classes_)
    print("Label distribution in training set:", np.unique(y_train, return_counts=True))
    print("Label distribution in testing set:", np.unique(y_test, return_counts=True))
    
    print("\nTraining set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)
    
    # Save processed data
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train_encoded)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test_encoded)
    np.save(os.path.join(output_dir, 'label_encoder.npy'), label_encoder)
    
    print("Data processing completed and saved to", output_dir)
    
    return X_train, y_train_encoded, X_test, y_test_encoded, label_encoder

if __name__ == "__main__":
    # Process data
    X_train, y_train, X_test, y_test, label_encoder = process_data() 