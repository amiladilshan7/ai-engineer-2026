"""
Day 11 - Cross Validation + Hyperparameter Tuning
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("=== Day 11: Cross Validation & Hyperparameter Tuning ===\n")

# Create data
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(18, 65, 500),
    'experience': np.random.randint(0, 30, 500),
    'education_level': np.random.randint(1, 5, 500),
    'performance_score': np.random.uniform(60, 95, 500)
})

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

# ==================== 1. Basic Random Forest ====================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

print("Basic Random Forest:")
print(f"R² Score : {r2_score(y_test, pred):.4f}")

# ==================== 2. Cross Validation ====================
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
print(f"\nCross Validation R² Scores: {cv_scores}")
print(f"Average CV R²: {cv_scores.mean():.4f}")

# ==================== 3. Hyperparameter Tuning with GridSearchCV ====================
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

print("\nTuning hyperparameters... (this may take a few seconds)")
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score : {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
pred_best = best_model.predict(X_test)
print(f"\nTuned Model Test R² Score: {r2_score(y_test, pred_best):.4f}")
