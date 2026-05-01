"""
Day 06 - Data Handling + Evaluation Metrics
"""

import numpy as np
from sklearn.model_selection import train_test_split   # new library for split

from models.linear_regression import LinearRegressionFromScratch
from models.gradient_descent import GradientDescentLinearRegression
from utils import calculate_metrics

print("=== Day 06: Proper Data Handling & Metrics ===\n")

# Create data
np.random.seed(42)
X = 2 * np.random.rand(200, 1)
y = 3 * X.squeeze() + 7 + np.random.randn(200) * 0.5

# Proper train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}\n")

# Normal Equation Model
model_ne = LinearRegressionFromScratch()
model_ne.fit(X_train, y_train)
pred_ne = model_ne.predict(X_test)
metrics_ne = calculate_metrics(y_test, pred_ne)

# Gradient Descent Model
model_gd = GradientDescentLinearRegression(learning_rate=0.01, n_epochs=1000)
model_gd.fit(X_train, y_train)
pred_gd = model_gd.predict(X_test)
metrics_gd = calculate_metrics(y_test, pred_gd)

# Final Comparison
print("📊 FINAL COMPARISON (on Test Data)")
print("Model                    | MSE     | MAE     | R²")
print("-------------------------|---------|---------|------")
print(f"Normal Equation          | {metrics_ne['MSE']:.4f} | {metrics_ne['MAE']:.4f} | {metrics_ne['R2']:.4f}")
print(f"Gradient Descent         | {metrics_gd['MSE']:.4f} | {metrics_gd['MAE']:.4f} | {metrics_gd['R2']:.4f}")
print("\n🎯 Both models tested on unseen data!")
