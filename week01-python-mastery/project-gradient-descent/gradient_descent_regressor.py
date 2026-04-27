import numpy as np
import matplotlib.pyplot as plt

class GradientDescentLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        m, n = X.shape
        # Initialize weights as a 1D array
        self.weights = np.zeros(n)
        self.bias = 0
        
        y = y.flatten() # Ensure y is 1D to match our predictions

        print(f"Starting training for {self.epochs} epochs...")
        for i in range(self.epochs):
            # 1. Predictions (Result is 1D)
            y_hat = np.dot(X, self.weights) + self.bias
            
            # 2. Loss
            loss = np.mean((y_hat - y)**2)
            self.loss_history.append(loss)

            # 3. Gradients
            # The fix: we ensure the dot product results in the correct shape (n,)
            error = y_hat - y
            dw = (2/m) * np.dot(X.T, error)
            db = (2/m) * np.sum(error)

            # 4. Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 100 == 0:
                print(f"Epoch {i}: Loss {loss:.4f}")

if __name__ == "__main__":
    print("--- STARTING GRADIENT DESCENT ---")
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.flatten() + np.random.randn(100) # Simplified data generation
    
    model = GradientDescentLinearRegression(learning_rate=0.1, epochs=500)
    model.fit(X, y)
    
    print(f"\nFinal Weights: {model.weights[0]:.4f} (Target: ~3.0)")
    print(f"Final Bias: {model.bias:.4f} (Target: ~4.0)")
    
    plt.plot(model.loss_history)
    plt.title("Loss Curve")
    plt.savefig("loss_curve.png")
    print("\n[SUCCESS] Script finished. Check loss_curve.png")
