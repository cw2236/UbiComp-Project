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

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        
        # Simplified LSTM architecture
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Reshape input for LSTM: [batch, features] -> [batch, 1, features]
        x = x.unsqueeze(1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout and final layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        return out

def load_data():
    # Load training and testing data
    X_train = np.load('processed_data/X_train.npy')
    y_train = np.load('processed_data/y_train.npy')
    X_test = np.load('processed_data/X_test.npy')
    y_test = np.load('processed_data/y_test.npy')
    
    # Load label encoder
    label_encoder = np.load('processed_data/label_encoder.npy', allow_pickle=True).item()
    
    return X_train, y_train, X_test, y_test, label_encoder

def preprocess_data(X_train, y_train, X_test, y_test):
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders with larger batch size
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    return train_loader, test_loader, X_test_scaled, y_test

def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        scheduler.step(epoch_loss)
        
        if (epoch + 1) % 5 == 0:  # Print less frequently
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

def evaluate_model(model, test_loader, criterion, device, label_encoder):
    model.eval()
    all_predictions = []
    all_labels = []
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
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
    plt.savefig('confusion_matrix_lstm.png')
    plt.close()

def main():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print('Loading data...')
    X_train, y_train, X_test, y_test, label_encoder = load_data()
    
    print('\nPreprocessing data...')
    train_loader, test_loader, X_test_scaled, y_test = preprocess_data(X_train, y_train, X_test, y_test)
    
    # Model parameters
    input_size = X_train.shape[1]  # Number of features
    hidden_size = 64  # Reduced hidden size
    num_classes = len(np.unique(y_train))
    
    # Initialize model, criterion, and optimizer
    model = LSTMClassifier(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)  # Increased learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5)
    
    print('\nTraining model...')
    train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=50)
    
    print('\nEvaluating model performance...')
    evaluate_model(model, test_loader, criterion, device, label_encoder)
    
    # Save the model
    torch.save(model.state_dict(), 'lstm_model.pth')

if __name__ == '__main__':
    main() 