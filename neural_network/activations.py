"""
Activation Functions for Neural Networks

This module implements various activation functions and their derivatives
used in neural network computations. All functions are designed to be
numerically stable and efficient.

Author: AI Assistant
"""

import numpy as np
from typing import Union
import warnings


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
    
    Args:
        x: Input value(s)
        
    Returns:
        Sigmoid of input value(s)
    """
    # Clip x to prevent overflow
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))


def sigmoid_derivative(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Derivative of sigmoid function: σ'(x) = σ(x) * (1 - σ(x))
    
    Args:
        x: Input value(s) (can be pre-computed sigmoid values)
        
    Returns:
        Derivative of sigmoid
    """
    # If x is already sigmoid values, use them directly
    if np.all((x >= 0) & (x <= 1)):
        return x * (1 - x)
    else:
        # Otherwise compute sigmoid first
        s = sigmoid(x)
        return s * (1 - s)


def tanh(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Hyperbolic tangent activation function: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Args:
        x: Input value(s)
        
    Returns:
        Tanh of input value(s)
    """
    # Clip x to prevent overflow
    x_clipped = np.clip(x, -500, 500)
    return np.tanh(x_clipped)


def tanh_derivative(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Derivative of tanh function: tanh'(x) = 1 - tanh²(x)
    
    Args:
        x: Input value(s) (can be pre-computed tanh values)
        
    Returns:
        Derivative of tanh
    """
    # If x is already tanh values, use them directly
    if np.all((x >= -1) & (x <= 1)):
        return 1 - x**2
    else:
        # Otherwise compute tanh first
        t = tanh(x)
        return 1 - t**2


def relu(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Rectified Linear Unit activation function: ReLU(x) = max(0, x)
    
    Args:
        x: Input value(s)
        
    Returns:
        ReLU of input value(s)
    """
    return np.maximum(0, x)


def relu_derivative(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Derivative of ReLU function: ReLU'(x) = 1 if x > 0, else 0
    
    Args:
        x: Input value(s)
        
    Returns:
        Derivative of ReLU
    """
    return (x > 0).astype(float)


def leaky_relu(x: Union[float, np.ndarray], alpha: float = 0.01) -> Union[float, np.ndarray]:
    """
    Leaky ReLU activation function: LeakyReLU(x) = x if x > 0, else αx
    
    Args:
        x: Input value(s)
        alpha: Negative slope coefficient (default: 0.01)
        
    Returns:
        Leaky ReLU of input value(s)
    """
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x: Union[float, np.ndarray], alpha: float = 0.01) -> Union[float, np.ndarray]:
    """
    Derivative of Leaky ReLU function: LeakyReLU'(x) = 1 if x > 0, else α
    
    Args:
        x: Input value(s)
        alpha: Negative slope coefficient (default: 0.01)
        
    Returns:
        Derivative of Leaky ReLU
    """
    return np.where(x > 0, 1, alpha)


def softmax(x: Union[float, np.ndarray], axis: int = -1) -> Union[float, np.ndarray]:
    """
    Softmax activation function: softmax(x_i) = e^(x_i) / Σ(e^(x_j))
    
    Args:
        x: Input value(s)
        axis: Axis along which to compute softmax (default: -1)
        
    Returns:
        Softmax of input value(s)
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_derivative(x: Union[float, np.ndarray], axis: int = -1) -> Union[float, np.ndarray]:
    """
    Derivative of softmax function (Jacobian matrix).
    For softmax, the derivative is: softmax'(x_i) = softmax(x_i) * (δ_ij - softmax(x_j))
    
    Args:
        x: Input value(s)
        axis: Axis along which softmax was computed
        
    Returns:
        Derivative of softmax (simplified for binary classification)
    """
    s = softmax(x, axis=axis)
    return s * (1 - s)


def linear(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Linear activation function: f(x) = x
    
    Args:
        x: Input value(s)
        
    Returns:
        Input value(s) unchanged
    """
    return x


def linear_derivative(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Derivative of linear function: f'(x) = 1
    
    Args:
        x: Input value(s)
        
    Returns:
        Array of ones with same shape as input
    """
    return np.ones_like(x)


class ActivationFunction:
    """
    Wrapper class for activation functions and their derivatives.
    """
    
    def __init__(self, func, derivative_func, name: str):
        """
        Initialize activation function wrapper.
        
        Args:
            func: Activation function
            derivative_func: Derivative function
            name: Name of the activation function
        """
        self.func = func
        self.derivative = derivative_func
        self.name = name
    
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Call the activation function."""
        return self.func(x)
    
    def __str__(self) -> str:
        """String representation."""
        return self.name


# Pre-defined activation functions
SIGMOID = ActivationFunction(sigmoid, sigmoid_derivative, "Sigmoid")
TANH = ActivationFunction(tanh, tanh_derivative, "Tanh")
RELU = ActivationFunction(relu, relu_derivative, "ReLU")
LEAKY_RELU = ActivationFunction(leaky_relu, leaky_relu_derivative, "Leaky ReLU")
SOFTMAX = ActivationFunction(softmax, softmax_derivative, "Softmax")
LINEAR = ActivationFunction(linear, linear_derivative, "Linear")


def get_activation(name: str) -> ActivationFunction:
    """
    Get activation function by name.
    
    Args:
        name: Name of activation function (case-insensitive)
        
    Returns:
        ActivationFunction object
        
    Raises:
        ValueError: If activation function name is not recognized
    """
    activations = {
        'sigmoid': SIGMOID,
        'tanh': TANH,
        'relu': RELU,
        'leaky_relu': LEAKY_RELU,
        'softmax': SOFTMAX,
        'linear': LINEAR
    }
    
    name_lower = name.lower()
    if name_lower not in activations:
        raise ValueError(f"Unknown activation function: {name}. "
                        f"Available: {list(activations.keys())}")
    
    return activations[name_lower]


def plot_activation_functions():
    """
    Plot all activation functions and their derivatives.
    """
    import matplotlib.pyplot as plt
    
    x = np.linspace(-5, 5, 1000)
    activations = [SIGMOID, TANH, RELU, LEAKY_RELU]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, activation in enumerate(activations):
        y = activation(x)
        y_deriv = activation.derivative(x)
        
        axes[i].plot(x, y, label=f'{activation.name}(x)', linewidth=2)
        axes[i].plot(x, y_deriv, label=f"{activation.name}'(x)", linewidth=2, linestyle='--')
        axes[i].set_title(f'{activation.name} Activation Function')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(-5, 5)
    
    plt.tight_layout()
    plt.show()


def main():
    """Demonstrate activation functions."""
    print("Activation Functions Demo")
    print("=" * 30)
    
    # Test with sample values
    x = np.array([-2, -1, 0, 1, 2])
    
    print(f"Input values: {x}")
    print()
    
    # Test each activation function
    for activation in [SIGMOID, TANH, RELU, LEAKY_RELU]:
        y = activation(x)
        y_deriv = activation.derivative(x)
        
        print(f"{activation.name}:")
        print(f"  Function: {y}")
        print(f"  Derivative: {y_deriv}")
        print()
    
    # Test softmax
    x_softmax = np.array([1, 2, 3])
    y_softmax = softmax(x_softmax)
    print(f"Softmax input: {x_softmax}")
    print(f"Softmax output: {y_softmax}")
    print(f"Sum: {np.sum(y_softmax)}")
    print()
    
    # Test gradient checking
    def test_function(x):
        return np.sum(sigmoid(x))
    
    x_test = np.array([1.0, 2.0, 3.0])
    analytical_grad = sigmoid_derivative(x_test)
    numerical_grad = np.array([
        (test_function(x_test + np.array([1e-6, 0, 0])) - test_function(x_test - np.array([1e-6, 0, 0]))) / (2 * 1e-6),
        (test_function(x_test + np.array([0, 1e-6, 0])) - test_function(x_test - np.array([0, 1e-6, 0]))) / (2 * 1e-6),
        (test_function(x_test + np.array([0, 0, 1e-6])) - test_function(x_test - np.array([0, 0, 1e-6]))) / (2 * 1e-6)
    ])
    
    print("Gradient checking for sigmoid:")
    print(f"Analytical: {analytical_grad}")
    print(f"Numerical:  {numerical_grad}")
    print(f"Difference: {np.abs(analytical_grad - numerical_grad)}")


if __name__ == "__main__":
    main()



