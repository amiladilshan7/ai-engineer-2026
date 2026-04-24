"""
Project: AI Engineer 2026 - Week 1
Author: Amila Dilshan
Date: April 24, 2026 (Day 03)

Project: Linear Regression from Scratch
Architecture Principles:
- Single Responsibility Principle
- Separation of Concerns
- Clean, readable, typed code
- Proper logging and error handling
"""

import logging
import numpy as np

# ------------------- Logging Setup -------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LinearRegression")

class LinearRegressionFromScratch:
    """Linear Regression implemented from scratch using Normal Equation."""

    def __init__(self):
        self.weights: np.ndarray = None
        self.bias: float = None
        logger.info("LinearRegressionFromScratch model initialized")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info(f"Starting training on {X.shape[0]} samples, {X.shape[1]} features")
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        try:
            theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
            self.bias = theta[0]
            self.weights = theta[1:]
            logger.info("Training completed successfully")
        except np.linalg.LinAlgError:
            logger.error("Matrix is singular. Try adding regularization or more data.")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model must be trained first. Call fit() before predict().")
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ np.r_[self.bias, self.weights]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        logger.info(f"R² Score: {r2:.4f}")
        return r2

# ------------------- Improved Test with Train/Test Split -------------------
# ------------------- Improved Test with Train/Test Split + Plot -------------------
if __name__ == "__main__":
    np.random.seed(42)
    X = 2 * np.random.rand(200, 1)
    y = 3 * X.squeeze() + 7 + np.random.randn(200) * 0.5

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = LinearRegressionFromScratch()
    model.fit(X_train, y_train)
    
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)
    print(f"R² on Train: {r2_train:.4f}")
    print(f"R² on Test : {r2_test:.4f}")
    
    X_new = np.array([[0], [1], [2]])
    print("Predictions for new data:", model.predict(X_new))

    # === VISUALIZATION ===
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, color='blue', label='Training data')
    plt.scatter(X_test, y_test, color='green', label='Test data')
    
    # Plot the regression line
    X_plot = np.linspace(0, 2, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    plt.plot(X_plot, y_plot, color='red', linewidth=3, label='Regression Line')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression from Scratch')
    plt.legend()
    plt.grid(True)
    plt.show()

    # === SAVE PLOT AS PNG (works in WSL) ===
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, color='blue', label='Training data')
    plt.scatter(X_test, y_test, color='green', label='Test data')
    
    X_plot = np.linspace(0, 2, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    plt.plot(X_plot, y_plot, color='red', linewidth=3, label='Regression Line')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression from Scratch (Amila)')
    plt.legend()
    plt.grid(True)
    plt.savefig('linear_regression_plot.png')
    print("✅ Plot saved as 'linear_regression_plot.png'")
    # plt.show()  # commented out because of WSL
