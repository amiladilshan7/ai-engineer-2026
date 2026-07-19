import torch
import torch.nn as nn

# Define a simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO 1: Create first layer (input 1 feature → 10 neurons)
        self.layer1 = nn.Linear(1, 10)
        
        # TODO 2: Create output layer (10 neurons → 1 output)
        self.layer2 = nn.Linear(10, 1)
        
    def forward(self, x):
        # TODO 3: Pass data through layer1 + ReLU activation
        x = torch.relu(self.layer1(x))
        
        # TODO 4: Pass through output layer
        x = self.layer2(x)
        return x

# Test the model
if __name__ == "__main__":
    model = SimpleNN()
    print("Model created successfully!")
    
    # Test with sample input
    test_input = torch.tensor([[2.0]])
    output = model(test_input)
    print("Input:", test_input)
    print("Output:", output)
