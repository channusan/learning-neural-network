"""
Loss Functions for Neural Networks

This module implements various loss functions and their derivatives
used in neural network training, with focus on binary classification.

Author: AI Assistant
"""

import numpy as np
from typing import Union, Optional
import warnings


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
    """
    Binary cross-entropy loss function.
    
    BCE = -1/n * Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
    
    Args:
        y_true: True binary labels (0 or 1) of shape (n_samples,)
        y_pred: Predicted probabilities of shape (n_samples,)
        epsilon: Small value to prevent log(0)
        
    Returns:
        Binary cross-entropy loss (scalar)
    """
    # Clip predictions to prevent log(0)
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate loss
    loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    
    return loss


def binary_cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
    """
    Derivative of binary cross-entropy loss with respect to predictions.
    
    dBCE/dy_pred = (y_pred - y_true) / (y_pred * (1 - y_pred))
    
    Args:
        y_true: True binary labels (0 or 1) of shape (n_samples,)
        y_pred: Predicted probabilities of shape (n_samples,)
        epsilon: Small value to prevent division by zero
        
    Returns:
        Gradient of loss with respect to predictions of shape (n_samples,)
    """
    # Clip predictions to prevent division by zero
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate derivative
    gradient = (y_pred_clipped - y_true) / (y_pred_clipped * (1 - y_pred_clipped))
    
    return gradient


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean squared error loss function.
    
    MSE = 1/n * Σ(y_true - y_pred)²
    
    Args:
        y_true: True values of shape (n_samples,)
        y_pred: Predicted values of shape (n_samples,)
        
    Returns:
        Mean squared error loss (scalar)
    """
    return np.mean((y_true - y_pred) ** 2)


def mean_squared_error_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Derivative of mean squared error loss with respect to predictions.
    
    dMSE/dy_pred = 2/n * (y_pred - y_true)
    
    Args:
        y_true: True values of shape (n_samples,)
        y_pred: Predicted values of shape (n_samples,)
        
    Returns:
        Gradient of loss with respect to predictions of shape (n_samples,)
    """
    return 2 * (y_pred - y_true) / len(y_true)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean absolute error loss function.
    
    MAE = 1/n * Σ|y_true - y_pred|
    
    Args:
        y_true: True values of shape (n_samples,)
        y_pred: Predicted values of shape (n_samples,)
        
    Returns:
        Mean absolute error loss (scalar)
    """
    return np.mean(np.abs(y_true - y_pred))


def mean_absolute_error_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Derivative of mean absolute error loss with respect to predictions.
    
    dMAE/dy_pred = sign(y_pred - y_true) / n
    
    Args:
        y_true: True values of shape (n_samples,)
        y_pred: Predicted values of shape (n_samples,)
        
    Returns:
        Gradient of loss with respect to predictions of shape (n_samples,)
    """
    return np.sign(y_pred - y_true) / len(y_true)


def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Hinge loss function for binary classification.
    
    Hinge = 1/n * Σ max(0, 1 - y_true * y_pred)
    where y_true is in {-1, 1}
    
    Args:
        y_true: True binary labels (-1 or 1) of shape (n_samples,)
        y_pred: Predicted values of shape (n_samples,)
        
    Returns:
        Hinge loss (scalar)
    """
    # Convert y_true from {0, 1} to {-1, 1} if needed
    if np.all(np.isin(y_true, [0, 1])):
        y_true = 2 * y_true - 1
    
    return np.mean(np.maximum(0, 1 - y_true * y_pred))


def hinge_loss_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Derivative of hinge loss with respect to predictions.
    
    dHinge/dy_pred = -y_true if y_true * y_pred < 1, else 0
    
    Args:
        y_true: True binary labels (-1 or 1) of shape (n_samples,)
        y_pred: Predicted values of shape (n_samples,)
        
    Returns:
        Gradient of loss with respect to predictions of shape (n_samples,)
    """
    # Convert y_true from {0, 1} to {-1, 1} if needed
    if np.all(np.isin(y_true, [0, 1])):
        y_true = 2 * y_true - 1
    
    return -y_true * (y_true * y_pred < 1).astype(float)


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
    """
    Categorical cross-entropy loss function for multi-class classification.
    
    CCE = -1/n * Σ Σ y_true * log(y_pred)
    
    Args:
        y_true: True one-hot encoded labels of shape (n_samples, n_classes)
        y_pred: Predicted probabilities of shape (n_samples, n_classes)
        epsilon: Small value to prevent log(0)
        
    Returns:
        Categorical cross-entropy loss (scalar)
    """
    # Clip predictions to prevent log(0)
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate loss
    loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
    
    return loss


def categorical_cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
    """
    Derivative of categorical cross-entropy loss with respect to predictions.
    
    dCCE/dy_pred = (y_pred - y_true) / n
    
    Args:
        y_true: True one-hot encoded labels of shape (n_samples, n_classes)
        y_pred: Predicted probabilities of shape (n_samples, n_classes)
        epsilon: Small value to prevent division by zero
        
    Returns:
        Gradient of loss with respect to predictions of shape (n_samples, n_classes)
    """
    return (y_pred - y_true) / len(y_true)


class LossFunction:
    """
    Wrapper class for loss functions and their derivatives.
    """
    
    def __init__(self, func, derivative_func, name: str):
        """
        Initialize loss function wrapper.
        
        Args:
            func: Loss function
            derivative_func: Derivative function
            name: Name of the loss function
        """
        self.func = func
        self.derivative = derivative_func
        self.name = name
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """Call the loss function."""
        return self.func(y_true, y_pred, **kwargs)
    
    def __str__(self) -> str:
        """String representation."""
        return self.name


# Pre-defined loss functions
BINARY_CROSS_ENTROPY = LossFunction(binary_cross_entropy, binary_cross_entropy_derivative, "Binary Cross-Entropy")
MEAN_SQUARED_ERROR = LossFunction(mean_squared_error, mean_squared_error_derivative, "Mean Squared Error")
MEAN_ABSOLUTE_ERROR = LossFunction(mean_absolute_error, mean_absolute_error_derivative, "Mean Absolute Error")
HINGE_LOSS = LossFunction(hinge_loss, hinge_loss_derivative, "Hinge Loss")
CATEGORICAL_CROSS_ENTROPY = LossFunction(categorical_cross_entropy, categorical_cross_entropy_derivative, "Categorical Cross-Entropy")


def get_loss_function(name: str) -> LossFunction:
    """
    Get loss function by name.
    
    Args:
        name: Name of loss function (case-insensitive)
        
    Returns:
        LossFunction object
        
    Raises:
        ValueError: If loss function name is not recognized
    """
    loss_functions = {
        'binary_cross_entropy': BINARY_CROSS_ENTROPY,
        'bce': BINARY_CROSS_ENTROPY,
        'mean_squared_error': MEAN_SQUARED_ERROR,
        'mse': MEAN_SQUARED_ERROR,
        'mean_absolute_error': MEAN_ABSOLUTE_ERROR,
        'mae': MEAN_ABSOLUTE_ERROR,
        'hinge_loss': HINGE_LOSS,
        'categorical_cross_entropy': CATEGORICAL_CROSS_ENTROPY,
        'cce': CATEGORICAL_CROSS_ENTROPY
    }
    
    name_lower = name.lower()
    if name_lower not in loss_functions:
        raise ValueError(f"Unknown loss function: {name}. "
                        f"Available: {list(loss_functions.keys())}")
    
    return loss_functions[name_lower]


def plot_loss_functions():
    """
    Plot various loss functions for binary classification.
    """
    import matplotlib.pyplot as plt
    
    # Generate sample data
    y_true = np.array([0, 0, 1, 1])
    y_pred_range = np.linspace(0.01, 0.99, 100)
    
    # Calculate losses for different y_pred values
    bce_losses = []
    mse_losses = []
    mae_losses = []
    
    for y_pred_val in y_pred_range:
        y_pred = np.full_like(y_true, y_pred_val, dtype=float)
        
        bce_losses.append(binary_cross_entropy(y_true, y_pred))
        mse_losses.append(mean_squared_error(y_true, y_pred))
        mae_losses.append(mean_absolute_error(y_true, y_pred))
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(y_pred_range, bce_losses, 'b-', linewidth=2, label='Binary Cross-Entropy')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Loss')
    plt.title('Binary Cross-Entropy Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(y_pred_range, mse_losses, 'r-', linewidth=2, label='Mean Squared Error')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Loss')
    plt.title('Mean Squared Error Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(y_pred_range, mae_losses, 'g-', linewidth=2, label='Mean Absolute Error')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Loss')
    plt.title('Mean Absolute Error Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def main():
    """Demonstrate loss functions."""
    print("Loss Functions Demo")
    print("=" * 25)
    
    # Generate sample data
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    
    print(f"True labels: {y_true}")
    print(f"Predicted probabilities: {y_pred}")
    print()
    
    # Test each loss function
    loss_functions = [BINARY_CROSS_ENTROPY, MEAN_SQUARED_ERROR, MEAN_ABSOLUTE_ERROR]
    
    for loss_func in loss_functions:
        loss = loss_func(y_true, y_pred)
        gradient = loss_func.derivative(y_true, y_pred)
        
        print(f"{loss_func.name}:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Gradient: {gradient}")
        print()
    
    # Test hinge loss (convert to {-1, 1} format)
    y_true_hinge = 2 * y_true - 1
    y_pred_hinge = 2 * y_pred - 1
    
    hinge_loss_val = hinge_loss(y_true_hinge, y_pred_hinge)
    hinge_gradient = hinge_loss_derivative(y_true_hinge, y_pred_hinge)
    
    print("Hinge Loss (converted to {-1, 1} format):")
    print(f"  Loss: {hinge_loss_val:.4f}")
    print(f"  Gradient: {hinge_gradient}")
    print()
    
    # Test gradient checking for binary cross-entropy
    def test_function(y_pred_test):
        return binary_cross_entropy(y_true, y_pred_test)
    
    analytical_grad = binary_cross_entropy_derivative(y_true, y_pred)
    numerical_grad = np.array([
        (test_function(y_pred + np.array([1e-6, 0, 0, 0])) - test_function(y_pred - np.array([1e-6, 0, 0, 0]))) / (2 * 1e-6),
        (test_function(y_pred + np.array([0, 1e-6, 0, 0])) - test_function(y_pred - np.array([0, 1e-6, 0, 0]))) / (2 * 1e-6),
        (test_function(y_pred + np.array([0, 0, 1e-6, 0])) - test_function(y_pred - np.array([0, 0, 1e-6, 0]))) / (2 * 1e-6),
        (test_function(y_pred + np.array([0, 0, 0, 1e-6])) - test_function(y_pred - np.array([0, 0, 0, 1e-6]))) / (2 * 1e-6)
    ])
    
    print("Gradient checking for Binary Cross-Entropy:")
    print(f"Analytical: {analytical_grad}")
    print(f"Numerical:  {numerical_grad}")
    print(f"Difference: {np.abs(analytical_grad - numerical_grad)}")


if __name__ == "__main__":
    main()



