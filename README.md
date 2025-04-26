# Action Recognition using FMCW Radar Data

This project implements a machine learning model for action recognition using FMCW (Frequency Modulated Continuous Wave) radar data.

## Project Structure

```
.
├── process_data.py      # Data processing and augmentation
├── train_knn.py         # KNN model training and evaluation
├── check_data.py        # Data inspection tool
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Features

- Data preprocessing and augmentation
- KNN-based action recognition model
- Model evaluation and visualization
- Data inspection tools

## Requirements

- Python 3.x
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/action-recognition-fmcw.git
cd action-recognition-fmcw
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Process the data:
```bash
python process_data.py
```

2. Train and evaluate the model:
```bash
python train_knn.py
```

3. Check data statistics:
```bash
python check_data.py
```

## Data Format

- Input data: FMCW radar profiles in .npy format
- Ground truth: Action labels in .txt format
- Processed data: Features and labels in .npy format

## Model Performance

The current implementation achieves:
- Training accuracy: 100%
- Test accuracy: 100%

## License

MIT License 