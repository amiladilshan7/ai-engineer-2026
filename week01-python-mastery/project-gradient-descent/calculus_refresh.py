import numpy as np

def loss_function(w):
    """Simple loss: y = 3*X + 7, so loss = (3*w - 3)^2"""
    return (3 * w - 3) ** 2

def numerical_derivative(w, h=0.0001):
    """Calculate approximate derivative using central difference."""
    loss_plus = loss_function(w + h)
    loss_minus = loss_function(w - h)
    derivative = (loss_plus - loss_minus) / (2 * h)
    return derivative

# Test
if __name__ == "__main__":
    print("=== Calculus Refresh - Numerical Derivative ===\n")
    
    weights_to_test = [0.0, 1.0, 2.0, 3.0, 4.0]
    
    for w in weights_to_test:
        loss = loss_function(w)
        deriv = numerical_derivative(w)
        print(f"Weight w = {w:4.1f}  |  Loss = {loss:8.4f}  |  Derivative = {deriv:8.4f}")
    
    print("\nObservation:")
    print("- When w < 3, derivative is negative → increase w to reduce loss")
    print("- When w > 3, derivative is positive → decrease w to reduce loss")
    print("- At w = 3, derivative is almost 0 → this is the minimum loss")

