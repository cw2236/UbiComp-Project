import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.patches import Rectangle
import glob

def create_sliding_windows(data, window_size=10, stride=5):
    """Copy the sliding window logic from process_data.py"""
    n_frames, n_features = data.shape
    n_windows = ((n_frames - window_size) // stride) + 1
    windows = np.zeros((n_windows, window_size, n_features))
    
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        windows[i] = data[start_idx:end_idx]
    
    return windows

def get_user_files(user_dir):
    """Get ground truth and data files for a user"""
    # Find ground truth file
    gt_files = glob.glob(os.path.join(user_dir, "record_*_records.txt"))
    if not gt_files:
        raise FileNotFoundError(f"No ground truth file found in {user_dir}")
    gt_file = gt_files[0]
    
    # Find data file
    data_files = glob.glob(os.path.join(user_dir, "*_fmcw_16bit_diff_profiles.npy"))
    if not data_files:
        raise FileNotFoundError(f"No data file found in {user_dir}")
    data_file = data_files[0]
    
    # Find frame time file
    gt_prefix = gt_file[:-12]  # Remove '_records.txt'
    time_file = f"{gt_prefix}_frame_time.txt"
    if not os.path.exists(time_file):
        raise FileNotFoundError(f"No frame time file found: {time_file}")
    
    return data_file, time_file, gt_file

def visualize_windows():
    # Create output directory
    output_dir = "jasmine_comparison_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all jasmine user directories
    jasmine_dirs = [f"Pilot/jasmine{i}" for i in range(1, 8)]
    
    # Dictionary to store all actions and their data
    all_actions = {}
    
    # Process each user's data
    for user_dir in jasmine_dirs:
        try:
            data_file, time_file, gt_file = get_user_files(user_dir)
            user_id = os.path.basename(user_dir)
            
            # Load data
            sensor_data = np.load(data_file)
            frame_times = np.loadtxt(time_file)
            gt_data = pd.read_csv(gt_file, header=None, 
                                names=['id', 'start_time', 'end_time', 'action'])
            
            # Process each action
            for _, row in gt_data.iterrows():
                start_time = row['start_time']
                end_time = row['end_time']
                action = row['action']
                
                # Find corresponding frames
                mask = (frame_times >= start_time) & (frame_times <= end_time)
                action_indices = np.where(mask)[0]
                
                if len(action_indices) == 0:
                    print(f"No frames found for action {action} in time range {start_time}-{end_time} for {user_id}")
                    continue
                
                # Extract action data
                action_data = sensor_data[action_indices, 1:]  # Exclude timestamp column
                n_frames = len(action_data)
                
                # Use the same window parameters as process_data.py
                window_size = 20  # jasmine users are among users 8 and above
                
                # Calculate stride to generate approximately 11 windows
                if n_frames > window_size:
                    stride = max(1, (n_frames - window_size) // 10)
                else:
                    print(f"Action {action} has fewer frames ({n_frames}) than window size ({window_size}) for {user_id}")
                    continue
                
                # Create sliding windows
                windows = create_sliding_windows(action_data, window_size, stride)
                
                # Store the data for later comparison
                if action not in all_actions:
                    all_actions[action] = []
                all_actions[action].append({
                    'user_id': user_id,
                    'data': action_data,
                    'n_frames': n_frames,
                    'window_size': window_size,
                    'stride': stride,
                    'n_windows': len(windows),
                    'start_time': start_time,
                    'end_time': end_time
                })
                
        except Exception as e:
            print(f"Error processing {user_dir}: {e}")
            continue
    
    # Generate comparison visualizations for each action
    for action, user_data_list in all_actions.items():
        if len(user_data_list) < 7:  # Skip if we don't have data from all users
            print(f"Skipping {action} as we don't have data from all users")
            continue
            
        # Create figure with 7 subplots
        fig, axes = plt.subplots(7, 1, figsize=(15, 35))
        fig.suptitle(f'Comparison of "{action}" across all Jasmine users\n', fontsize=16)
        
        # Sort user data by user ID
        user_data_list.sort(key=lambda x: x['user_id'])
        
        for i, (ax, user_data) in enumerate(zip(axes, user_data_list)):
            user_id = user_data['user_id']
            action_data = user_data['data']
            n_frames = user_data['n_frames']
            window_size = user_data['window_size']
            stride = user_data['stride']
            n_windows = user_data['n_windows']
            start_time = user_data['start_time']
            end_time = user_data['end_time']
            
            # Plot original data
            data_line = ax.plot(range(n_frames), action_data[:, 0], 'b-', 
                              label='Original Data', alpha=0.7, linewidth=2)
            
            # Calculate y-axis limits with some padding
            y_min = np.min(action_data[:, 0])
            y_max = np.max(action_data[:, 0])
            y_range = y_max - y_min
            y_padding = y_range * 0.1
            
            # Plot window boxes
            colors = plt.cm.rainbow(np.linspace(0, 1, n_windows))
            for j in range(n_windows):
                start_idx = j * stride
                end_idx = start_idx + window_size
                
                # Create dashed rectangle for each window
                rect = Rectangle((start_idx, y_min - y_padding), 
                               window_size, y_range + 2*y_padding,
                               fill=False, 
                               linestyle='--',
                               edgecolor=colors[j],
                               label=f'Window {j+1}',
                               linewidth=1.5)
                ax.add_patch(rect)
            
            # Set axis limits
            ax.set_xlim(-2, n_frames + 2)
            ax.set_ylim(y_min - y_padding*1.2, y_max + y_padding*1.2)
            
            ax.set_title(f'{user_id}\n'
                        f'Time Range: {start_time:.3f} - {end_time:.3f}\n'
                        f'Total Frames: {n_frames}, Window Size: {window_size}, Stride: {stride}, Number of Windows: {n_windows}')
            ax.set_xlabel('Frame Index')
            ax.set_ylabel('Feature Value')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f'{action}_comparison.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved comparison visualization for {action} to {output_file}\n")

if __name__ == "__main__":
    visualize_windows() 