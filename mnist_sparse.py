import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import pandas as pd

# The default model runs for 20 epochs in training and has app. 94% accuracy after
# that time. It has two hidden layers. The input layer is 28x28 picture expressed 
# in grey scale values from mnist_train.csv or mnist_test.csv. The hidden layers 
# have 512 and 10 nodes respectively. The output layer has 10 nodes corresponding 
# to 10 classification sections for number 0 through 9. The activation function 
# between layers is always ReLU. Cross entropy loss function is at use here. The 
# optimizer is SGD the whole thing, while learning rate is set as default to 0.1.
# All layers of this model are partially connected (aka. sparse) with 50% (0.5) of
# connections disconnected.

epochs = 20
lr = 0.01

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.5):
        super(SparseLinear, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        # Create a sparse mask
        self.mask = torch.rand(out_features, in_features) > sparsity

    def forward(self, x):
        # Apply the mask to the weights
        sparse_weights = self.fc.weight * self.mask
        return F.linear(x, sparse_weights, self.fc.bias)


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            SparseLinear(28*28, 512),
            nn.ReLU(),
            SparseLinear(512, 10),
            nn.ReLU(),
            SparseLinear(10, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def train(model, train_loader, criterion, optimizer, device):
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()  # Zero the gradients before the backward pass
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update weights

        total_loss += loss.item()  # Accumulate loss

        if batch_idx % 100 == 0:  # Print every 100 batches
            print(f'Batch {batch_idx}: Loss {loss.item()}')

    avg_loss = total_loss / len(train_loader)
    print(f'Average Loss: {avg_loss}')
        
# Prepare the data
def prepare_data():
    # Load the training and test data
    train_data = pd.read_csv('mnist_train.csv')
    test_data = pd.read_csv('mnist_test.csv')

    # Prepare the data (labels are in the first column, pixels in the remaining columns)
    X_train = torch.tensor(train_data.iloc[:, 1:].values, dtype=torch.float32) / 255.0  # Normalize pixel values
    y_train = torch.tensor(train_data.iloc[:, 0].values, dtype=torch.long)

    X_test = torch.tensor(test_data.iloc[:, 1:].values, dtype=torch.float32) / 255.0
    y_test = torch.tensor(test_data.iloc[:, 0].values, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader


def main():
    # Model, criterion, optimizer setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr)

    # Prepare data
    train_loader, _ = prepare_data()
    
    counter = 0 
    
    while counter < epochs:
        # Train the model
        train(model, train_loader, criterion, optimizer, device)
        counter += 1

    # Testing the model (optional)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Load test data
        _, test_loader = prepare_data()

        correct = 0
        total = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    main()