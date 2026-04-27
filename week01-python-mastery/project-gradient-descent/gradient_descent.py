"""
Project: AI Engineer 2026 - Week 1
Author: Amila Dilshan
Date: April 25, 2026 (Day 04)

Project: Gradient Descent Linear Regression from Scratch
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GradientDescent")

class GradientDescentLinearRegression:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.loss_history = []
        logger.info(f"Model initialized (lr={learning_rate}, epochs={n_epochs})")

    def fit(self, X, y):
        """Train using Gradient Descent"""
        n_samples = X.shape[0]
        # Add bias column
        X_b = np.c_[np.ones((n_samples, 1)), X]
        
        # Initialize parameters
        self.weights = np.random.randn(X.shape[1]) * 0.01
        self.bias = 0.0

        for epoch in range(self.n_epochs):
            # Predictions
            y_pred = X_b @ np.r_[self.bias, self.weights]
            
            # Calculate gradients
            error = y_pred - y
            dw = (2 / n_samples) * (X_b.T @ error)[1:]
            db = (2 / n_samples) * np.sum(error)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Save loss
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)
            
            if epoch % 200 == 0:
                logger.info(f"Epoch {epoch} - Loss: {loss:.4f}")

        logger.info("Training finished!")

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ np.r_[self.bias, self.weights]

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2


# ==================== TEST ====================
if __name__ == "__main__":
    # Same synthetic data as yesterday
    np.random.seed(42)
    X = 2 * np.random.rand(200, 1)
    y = 3 * X.squeeze() + 7 + np.random.randn(200) * 0.5

    model = GradientDescentLinearRegression(learning_rate=0.01, n_epochs=1000)
    model.fit(X, y)

    print("\n=== Results ===")
    print(f"Learned bias: {model.bias:.4f}")
    print(f"Learned weight: {model.weights[0]:.4f}")
    print(f"R² Score: {model.score(X, y):.4f}")

    # Plot loss curve
    plt.plot(model.loss_history)
    plt.title("Loss Curve - Gradient Descent")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    print("✅ Loss curve saved as 'loss_curve.png'")
