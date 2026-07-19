import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 20)   # Increased neurons
        self.layer2 = nn.Linear(20, 1)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Better training data
x_train = torch.tensor([[i] for i in range(1, 11)], dtype=torch.float32)
y_train = torch.tensor([[3*i] for i in range(1, 11)], dtype=torch.float32)

losses = []

print("Training started...\n")

for epoch in range(500):
    optimizer.zero_grad()
    
    output = model(x_train)
    loss = criterion(output, y_train)
    
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Plot loss curve
plt.plot(losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_curve.png")
print("\n✅ Loss curve saved as 'loss_curve.png'")

# Final test
test_input = torch.tensor([[15.0]])
print(f"Prediction for 15.0 → {model(test_input).item():.2f} (should be close to 45)")
