import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 50)
        self.layer2 = nn.Linear(50, 1)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Create data with some noise
x = torch.linspace(0, 10, 100).unsqueeze(1)
y = 3 * x + torch.randn(100, 1) * 0.8   # noisy data

# TODO 1: Split data into train (70%) and test (30%)
train_size = int(0.7 * len(x))
x_train = x[:train_size]
y_train = y[:train_size]
x_test = x[train_size:]
y_test = y[train_size:]

train_losses = []
test_losses = []

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)
    
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    # TODO 2: Calculate test loss (without updating weights)
    with torch.no_grad():
        test_output = model(x_test)
        test_loss = criterion(test_output, y_test)
        test_losses.append(test_loss.item())
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

# Plot to see overfitting
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.title("Overfitting Example - Train vs Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("overfitting_plot.png")
print("\n✅ Plot saved as 'overfitting_plot.png'")
print("If test loss starts increasing while train loss decreases → Overfitting!")
