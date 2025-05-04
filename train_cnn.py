import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc  # Add garbage collection

# Set random seed
torch.manual_seed(42)
np.random.seed(42)


class SimpleCNN(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.3):
        super(SimpleCNN, self).__init__()
        
        # Calculate dimensions for reshaping
        self.height = 16  # Fixed height
        self.width = (input_size + self.height - 1) // self.height * self.height  # Round up to nearest multiple of height
        self.pad_size = self.width * self.height - input_size
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(dropout),
            
            nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(dropout)
        )
        
        # Calculate CNN output size
        self.cnn_output_size = self._get_cnn_output_size()
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_output_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, num_classes)
        )
    
    def _get_cnn_output_size(self):
        # Create a dummy input to calculate CNN output size
        dummy_input = torch.zeros(1, 1, self.height, self.width)
        with torch.no_grad():
            output = self.cnn(dummy_input)
        return output.view(1, -1).size(1)
    
    def forward(self, x):
        # Pad input to make it divisible by height
        if self.pad_size > 0:
            x = torch.cat([x, torch.zeros(x.size(0), self.pad_size, device=x.device)], dim=1)
        
        # Reshape input for CNN: [batch, features] -> [batch, 1, height, width]
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.height, self.width)
        
        # CNN forward pass
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(batch_size, -1)
        
        # Fully connected layers
        out = self.fc(cnn_out)
        
        return out

def load_data():
    # Load training and testing data
    print('Loading training data...')
    X_train = np.load('processed_data/X_train.npy', mmap_mode='r')  # Use memory mapping
    y_train = np.load('processed_data/y_train.npy')
    
    print('Loading testing data...')
    X_test = np.load('processed_data/X_test.npy', mmap_mode='r')
    y_test = np.load('processed_data/y_test.npy')
    
    # Load label encoder
    label_encoder = np.load('processed_data/label_encoder.npy', allow_pickle=True).item()
    
    return X_train, y_train, X_test, y_test, label_encoder

def preprocess_data(X_train, y_train, X_test, y_test):
    # Standardize the data
    scaler = StandardScaler()
    # Calculate mean and variance first
    scaler.fit(X_train)
    
    # Create data loader
    def create_dataset(X, y, batch_size, shuffle=True):
        dataset = []
        for i in range(0, len(X), batch_size):
            batch_X = torch.FloatTensor(scaler.transform(X[i:i+batch_size]))
            batch_y = torch.LongTensor(y[i:i+batch_size])
            dataset.append((batch_X, batch_y))
        return dataset
    
    train_dataset = create_dataset(X_train, y_train, batch_size=16)
    test_dataset = create_dataset(X_test, y_test, batch_size=16, shuffle=False)
    
    # Clean up memory
    gc.collect()
    
    return train_dataset, test_dataset

def train_model(model, train_dataset, criterion, optimizer, scheduler, device, num_epochs=30):
    model.train()
    # Record training process loss and accuracy
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_dataset:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            # Clean up memory (ensure variables are deleted after use)
            del batch_X, batch_y, predicted, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        epoch_loss = total_loss / len(train_dataset)
        epoch_acc = 100 * correct / total
        scheduler.step(epoch_loss)
        
        # Record training loss and accuracy
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
            # Regular memory cleanup
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Plot learning curves
    plt.figure(figsize=(12, 4))
    
    # Plot loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('cnn_learning_curves.png')
    plt.close()
    
    return train_losses, train_accuracies

def evaluate_model(model, test_dataset, criterion, device, label_encoder):
    model.eval()
    all_predictions = []
    all_labels = []
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_dataset:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
            # Clean up memory
            del batch_X, batch_y, outputs, predicted
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    test_accuracy = 100 * correct / total
    print(f'\nTest Accuracy: {test_accuracy:.2f}%')
    
    # Get class names from label encoder
    class_names = list(label_encoder.classes_)
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Create confusion matrix with class names
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix_cnn.png')
    plt.close()

def main():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print('Loading data...')
    X_train, y_train, X_test, y_test, label_encoder = load_data()
    
    print('\nPreprocessing data...')
    train_dataset, test_dataset = preprocess_data(X_train, y_train, X_test, y_test)
    
    # Model parameters
    input_size = X_train.shape[1]  # Number of features
    num_classes = len(np.unique(y_train))
    
    # Initialize model, criterion, and optimizer
    model = SimpleCNN(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5)
    
    print('\nTraining model...')
    train_losses, train_accuracies = train_model(model, train_dataset, criterion, optimizer, scheduler, device, num_epochs=30)
    
    print('\nEvaluating model performance...')
    evaluate_model(model, test_dataset, criterion, device, label_encoder)
    
    # Save the model
    torch.save(model.state_dict(), 'cnn_model.pth')
    
    # Final memory cleanup
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == '__main__':
    main() 