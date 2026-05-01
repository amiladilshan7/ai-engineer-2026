import numpy as np

def calculate_metrics(y_true, y_pred):
    """Calculate MSE, MAE and R²"""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {"MSE": mse, "MAE": mae, "R2": r2}
