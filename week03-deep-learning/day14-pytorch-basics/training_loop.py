import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Create model
model = SimpleNN()

# TODO 1: Create loss function
criterion = nn.MSELoss()          # Hint: Mean Squared Error

# TODO 2: Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)   # Hint: Adam optimizer

# Training data (x → y relationship: roughly y = 3x)
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_train = torch.tensor([[3.0], [6.0], [9.0], [12.0], [15.0]])

print("Starting training...\n")

for epoch in range(200):
    optimizer.zero_grad()           # Clear old gradients
    
    output = model(x_train)         # Forward pass
    
    # TODO 3: Calculate loss
    loss = criterion(output, y_train)   # Hint: output vs y_train
    
    loss.backward()                 # Backward pass (calculate gradients)
    optimizer.step()                # Update weights
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test the trained model
test_input = torch.tensor([[6.0]])
prediction = model(test_input)
print(f"\nPrediction for 6.0 → {prediction.item():.2f} (should be close to 18)")
