"""
Day 05 - Clean Architecture Main File
Purpose: This is the "control center" of the project.
It imports both models, runs them, and shows comparison.
"""

import numpy as np

# Import models from clean folder structure
from models.linear_regression import LinearRegressionFromScratch
from models.gradient_descent import GradientDescentLinearRegression

print("=== Day 05: Clean Project Comparison ===\n")

# Create same synthetic data (same as Day 03 and Day 04)
np.random.seed(42)
X = 2 * np.random.rand(200, 1)
y = 3 * X.squeeze() + 7 + np.random.randn(200) * 0.5

# ==================== MODEL 1: Normal Equation (Day 03) ====================
model_ne = LinearRegressionFromScratch()
model_ne.fit(X, y)
r2_ne = model_ne.score(X, y)

# ==================== MODEL 2: Gradient Descent (Day 04) ====================
model_gd = GradientDescentLinearRegression(learning_rate=0.01, n_epochs=1000)
model_gd.fit(X, y)
r2_gd = model_gd.score(X, y)

# ==================== COMPARISON ====================
print("📊 FINAL COMPARISON")
print("Model                    | R² Score")
print("-------------------------|----------")
print(f"Normal Equation          | {r2_ne:.4f}")
print(f"Gradient Descent         | {r2_gd:.4f}")
print("\n🎯 Both models give almost the same result.")
print("This shows our clean architecture is working perfectly!")
