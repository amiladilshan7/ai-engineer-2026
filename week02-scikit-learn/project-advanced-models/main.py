"""
Day 10 - Advanced Models: Random Forest & XGBoost
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("=== Day 10: Advanced Models ===\n")

# Create richer synthetic data
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(18, 65, 500),
    'experience': np.random.randint(0, 30, 500),
    'education_level': np.random.randint(1, 5, 500),   # 1=High School, 4=PhD
    'city_tier': np.random.randint(1, 4, 500),
    'performance_score': np.random.uniform(60, 95, 500)
})

# Target: Salary with complex relationship
data['salary'] = (
    data['experience'] * 2500 +
    data['education_level'] * 8000 +
    data['performance_score'] * 800 +
    np.random.normal(0, 8000, 500)
)

X = data.drop('salary', axis=1)
y = data['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}\n")

# ==================== Random Forest ====================
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("Random Forest Results:")
print(f"R² Score : {r2_score(y_test, rf_pred):.4f}")
print(f"MSE      : {mean_squared_error(y_test, rf_pred):.2f}")

# ==================== XGBoost ====================
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

print("\nXGBoost Results:")
print(f"R² Score : {r2_score(y_test, xgb_pred):.4f}")
print(f"MSE      : {mean_squared_error(y_test, xgb_pred):.2f}")

print("\n🎯 Tree-based models usually perform better than simple Linear Regression on complex data!")

