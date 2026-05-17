"""
Day 12 - Full End-to-End ML Project
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

print("=== Day 12: Full End-to-End ML Project ===\n")

# ==================== 1. Create Dataset ====================
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(22, 60, 1000),
    'experience': np.random.randint(0, 25, 1000),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
    'city': np.random.choice(['Colombo', 'Kandy', 'Galle'], 1000),
    'performance_score': np.random.uniform(50, 95, 1000)
})

data['salary'] = (
    data['experience'] * 3000 +
    data['performance_score'] * 1200 +
    data['age'] * 800 +
    np.random.normal(0, 10000, 1000)
)

print("Dataset Shape:", data.shape)
print(data.head())

X = data.drop('salary', axis=1)
y = data['salary']

# ==================== 2. Train/Test Split ====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==================== 3. Preprocessing Pipeline ====================
numeric_features = ['age', 'experience', 'performance_score']
categorical_features = ['education', 'city']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# ==================== 4. Models ====================
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    
    results[name] = {
        'R2': r2_score(y_test, pred),
        'MSE': mean_squared_error(y_test, pred)
    }
    
    print(f"\n{name} Results:")
    print(f"R² Score : {results[name]['R2']:.4f}")
    print(f"MSE      : {results[name]['MSE']:.2f}")

print("\n🎯 End-to-End Project Completed!")
print("We used: Data Creation → Preprocessing → Multiple Models → Evaluation")
