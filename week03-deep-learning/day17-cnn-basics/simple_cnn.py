import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# TODO 1: Define a simple CNN class
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)   # 28x28 image becomes 7x7 after pooling
        self.fc2 = nn.Linear(128, 10)           # 10 classes (digits 0-9)
        
    def forward(self, x):
        # TODO 2: Pass through first conv + relu + pool
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        
        # TODO 3: Pass through second conv + relu + pool
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, 32 * 7 * 7)
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Main
if __name__ == "__main__":
    # TODO 4: Load MNIST dataset with transforms (normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training simple CNN...")
    # We will train for only 1-2 epochs today for speed
    for epoch in range(2):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Average Loss: {running_loss/len(train_loader):.4f}")

    print("✅ Simple CNN training completed!")
