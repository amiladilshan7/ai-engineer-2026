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
from typing import Tuple

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
        """
        Train the model using Normal Equation: w = (X^T X)^-1 X^T y
        X shape: (n_samples, n_features)
        y shape: (n_samples,)
        """
        logger.info(f"Starting training on {X.shape[0]} samples, {X.shape[1]} features")

        # Add bias term (column of 1s)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Normal Equation
        try:
            theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
            self.bias = theta[0]
            self.weights = theta[1:]
            logger.info("Training completed successfully")
        except np.linalg.LinAlgError:
            logger.error("Matrix is singular. Try adding regularization or more data.")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.weights is None:
            raise ValueError("Model must be trained first. Call fit() before predict().")
        
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ np.r_[self.bias, self.weights]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² score (coefficient of determination)."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        logger.info(f"R² Score: {r2:.4f}")
        return r2


# ------------------- Quick Test with Synthetic Data -------------------
if __name__ == "__main__":
    # Generate simple synthetic data: y = 3*X + 7 + noise
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)          # 100 samples, 1 feature
    y = 3 * X.squeeze() + 7 + np.random.randn(100) * 0.5

    model = LinearRegressionFromScratch()
    model.fit(X, y)
    
    # Test prediction
    X_test = np.array([[0], [1], [2]])
    predictions = model.predict(X_test)
    print("Predictions for X = [[0], [1], [2]]:", predictions)
    
    # Score
    r2 = model.score(X, y)
    print(f"Final R² on training data: {r2:.4f}")
