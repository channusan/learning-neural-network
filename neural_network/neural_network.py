"""
Neural Network Implementation

This module implements a 2-class classification neural network with a single
hidden layer, demonstrating forward and backward propagation, activation functions,
and gradient descent optimization.

Author: AI Assistant
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict, Any, List
import warnings
from .activations import tanh, tanh_derivative, sigmoid, sigmoid_derivative
from .loss_functions import binary_cross_entropy, binary_cross_entropy_derivative
from .utils import validate_input_data, calculate_all_metrics


class NeuralNetwork:
    """
    Neural Network with single hidden layer for binary classification.
    
    Architecture: 2 inputs -> 8 hidden neurons (tanh) -> 1 output (sigmoid)
    
    This implementation demonstrates:
    - Forward propagation
    - Backward propagation (backpropagation)
    - Gradient descent optimization
    - Non-linear classification capabilities
    """
    
    def __init__(self, input_size: int = 2, hidden_size: int = 8, output_size: int = 1,
                 learning_rate: float = 0.1, random_state: int = 42):
        """
        Initialize neural network.
        
        Args:
            input_size: Number of input features (default: 2)
            hidden_size: Number of hidden neurons (default: 8)
            output_size: Number of output neurons (default: 1)
            learning_rate: Learning rate for gradient descent
            random_state: Random seed for reproducible results
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Initialize weights and biases
        self._initialize_weights()
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
        self.is_fitted = False
        self.training_time_ms = 0.0
        
        # Cache for forward pass values (for backpropagation)
        self._cache = {}
    
    def _initialize_weights(self) -> None:
        """
        Initialize weights and biases using Xavier initialization.
        
        Xavier initialization helps prevent vanishing/exploding gradients by
        scaling weights based on the number of input neurons.
        """
        np.random.seed(self.random_state)
        
        # Xavier initialization for hidden layer weights
        # W1: (input_size, hidden_size)
        xavier_std = np.sqrt(2.0 / (self.input_size + self.hidden_size))
        self.W1 = np.random.normal(0, xavier_std, (self.input_size, self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))
        
        # Xavier initialization for output layer weights
        # W2: (hidden_size, output_size)
        xavier_std = np.sqrt(2.0 / (self.hidden_size + self.output_size))
        self.W2 = np.random.normal(0, xavier_std, (self.hidden_size, self.output_size))
        self.b2 = np.zeros((1, self.output_size))
    
    def _forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the network.
        
        Args:
            X: Input features of shape (n_samples, input_size)
            
        Returns:
            Output probabilities of shape (n_samples, output_size)
        """
        # Input to hidden layer
        # Z1 = X * W1 + b1
        self._cache['Z1'] = np.dot(X, self.W1) + self.b1
        
        # Apply tanh activation to hidden layer
        # A1 = tanh(Z1)
        self._cache['A1'] = tanh(self._cache['Z1'])
        
        # Hidden to output layer
        # Z2 = A1 * W2 + b2
        self._cache['Z2'] = np.dot(self._cache['A1'], self.W2) + self.b2
        
        # Apply sigmoid activation to output layer
        # A2 = sigmoid(Z2)
        self._cache['A2'] = sigmoid(self._cache['Z2'])
        
        return self._cache['A2']
    
    def _backward_propagation(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backward propagation to compute gradients.
        
        Args:
            X: Input features of shape (n_samples, input_size)
            y: True labels of shape (n_samples,)
            output: Network output of shape (n_samples, output_size)
            
        Returns:
            Dictionary containing gradients for all parameters
        """
        n_samples = X.shape[0]
        
        # Ensure y and output have the same shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if output.ndim == 1:
            output = output.reshape(-1, 1)
        
        # Output layer gradients
        # dZ2 = A2 - y (derivative of binary cross-entropy with sigmoid)
        dZ2 = output - y
        
        # Gradients for output layer weights and bias
        dW2 = (1 / n_samples) * np.dot(self._cache['A1'].T, dZ2)
        db2 = (1 / n_samples) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        # dA1 = dZ2 * W2^T
        dA1 = np.dot(dZ2, self.W2.T)
        
        # dZ1 = dA1 * tanh'(Z1)
        dZ1 = dA1 * tanh_derivative(self._cache['Z1'])
        
        # Gradients for hidden layer weights and bias
        dW1 = (1 / n_samples) * np.dot(X.T, dZ1)
        db1 = (1 / n_samples) * np.sum(dZ1, axis=0, keepdims=True)
        
        return {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
    
    def _update_parameters(self, gradients: Dict[str, np.ndarray]) -> None:
        """
        Update network parameters using gradient descent.
        
        Args:
            gradients: Dictionary containing gradients for all parameters
        """
        # Update hidden layer parameters
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        
        # Update output layer parameters
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Loss value
        """
        return binary_cross_entropy(y_true, y_pred)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
            verbose: bool = False, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'NeuralNetwork':
        """
        Train the neural network.
        
        Args:
            X: Training features of shape (n_samples, input_size)
            y: Training labels of shape (n_samples,)
            epochs: Number of training epochs
            verbose: Whether to print training progress
            validation_data: Optional validation data tuple (X_val, y_val)
            
        Returns:
            Self (for method chaining)
        """
        # Validate input data
        validate_input_data(X, y)
        
        # Clear training history
        self.loss_history = []
        self.accuracy_history = []
        
        if validation_data is not None:
            val_loss_history = []
            val_accuracy_history = []
        
        # Training loop
        print(f"   Starting training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Forward propagation
            output = self._forward_propagation(X)
            
            # Compute loss
            loss = self._compute_loss(y, output)
            self.loss_history.append(loss)
            
            # Compute accuracy
            y_pred = (output > 0.5).astype(int).flatten()
            accuracy = np.mean(y == y_pred)
            self.accuracy_history.append(accuracy)
            
            # Validation metrics
            if validation_data is not None:
                X_val, y_val = validation_data
                val_output = self._forward_propagation(X_val)
                val_loss = self._compute_loss(y_val, val_output)
                val_y_pred = (val_output > 0.5).astype(int).flatten()
                val_accuracy = np.mean(y_val == val_y_pred)
                
                val_loss_history.append(val_loss)
                val_accuracy_history.append(val_accuracy)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                if validation_data is not None:
                    print(f"     Epoch {epoch}: Loss = {loss:.6f}, Accuracy = {accuracy:.4f}, "
                          f"Val Loss = {val_loss:.6f}, Val Accuracy = {val_accuracy:.4f}")
                else:
                    print(f"     Epoch {epoch}: Loss = {loss:.6f}, Accuracy = {accuracy:.4f}")
            
            # Backward propagation
            gradients = self._backward_propagation(X, y, output)
            
            # Update parameters
            self._update_parameters(gradients)
        
        # Calculate training time
        end_time = time.time()
        self.training_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        
        print(f"   ✓ Training completed after {epochs} epochs")
        print(f"   ✓ Training time: {self.training_time_ms:.2f} ms")
        
        self.is_fitted = True
        
        # Store validation history if provided
        if validation_data is not None:
            self.val_loss_history = val_loss_history
            self.val_accuracy_history = val_accuracy_history
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features of shape (n_samples, input_size)
            
        Returns:
            Predicted probabilities of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        output = self._forward_propagation(X)
        return output.flatten()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels.
        
        Args:
            X: Input features of shape (n_samples, input_size)
            
        Returns:
            Predicted binary labels of shape (n_samples,)
        """
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.
        
        Args:
            X: Input features
            y: True labels
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
    
    def get_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive performance metrics.
        
        Args:
            X: Input features
            y: True labels
            
        Returns:
            Dictionary of performance metrics
        """
        y_pred = self.predict(X)
        return calculate_all_metrics(y, y_pred)
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training loss and accuracy history.
        
        Args:
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting history")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        axes[0].plot(self.loss_history, 'b-', linewidth=2, label='Training Loss')
        if hasattr(self, 'val_loss_history'):
            axes[0].plot(self.val_loss_history, 'r--', linewidth=2, label='Validation Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Binary Cross-Entropy Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(self.accuracy_history, 'b-', linewidth=2, label='Training Accuracy')
        if hasattr(self, 'val_accuracy_history'):
            axes[1].plot(self.val_accuracy_history, 'r--', linewidth=2, label='Validation Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_decision_boundary_weights(self) -> Dict[str, np.ndarray]:
        """
        Get weights for decision boundary visualization.
        
        Returns:
            Dictionary containing all network weights and biases
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting weights")
        
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary containing model information
        """
        if not self.is_fitted:
            return {"is_fitted": False}
        
        info = {
            "is_fitted": True,
            "architecture": f"{self.input_size}-{self.hidden_size}-{self.output_size}",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "n_epochs": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "final_accuracy": self.accuracy_history[-1] if self.accuracy_history else None,
            "total_parameters": (self.input_size * self.hidden_size + self.hidden_size + 
                               self.hidden_size * self.output_size + self.output_size),
            "training_time_ms": self.training_time_ms
        }
        
        if hasattr(self, 'val_loss_history'):
            info["final_val_loss"] = self.val_loss_history[-1]
            info["final_val_accuracy"] = self.val_accuracy_history[-1]
        
        return info
    
    def gradient_check(self, X: np.ndarray, y: np.ndarray, epsilon: float = 1e-7) -> Dict[str, float]:
        """
        Perform gradient checking to verify backpropagation implementation.
        
        Args:
            X: Input features
            y: True labels
            epsilon: Small value for numerical differentiation
            
        Returns:
            Dictionary containing relative errors for each parameter
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before gradient checking")
        
        # Forward pass
        output = self._forward_propagation(X)
        
        # Compute analytical gradients
        gradients = self._backward_propagation(X, y, output)
        
        # Compute numerical gradients
        def loss_function(params_flat):
            # Reshape parameters back to original shapes
            W1 = params_flat[:self.input_size * self.hidden_size].reshape(self.input_size, self.hidden_size)
            b1 = params_flat[self.input_size * self.hidden_size:self.input_size * self.hidden_size + self.hidden_size].reshape(1, self.hidden_size)
            W2 = params_flat[self.input_size * self.hidden_size + self.hidden_size:self.input_size * self.hidden_size + self.hidden_size + self.hidden_size * self.output_size].reshape(self.hidden_size, self.output_size)
            b2 = params_flat[self.input_size * self.hidden_size + self.hidden_size + self.hidden_size * self.output_size:].reshape(1, self.output_size)
            
            # Temporarily store original parameters
            orig_W1, orig_b1, orig_W2, orig_b2 = self.W1, self.b1, self.W2, self.b2
            
            # Set new parameters
            self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
            
            # Compute loss
            output = self._forward_propagation(X)
            loss = self._compute_loss(y, output)
            
            # Restore original parameters
            self.W1, self.b1, self.W2, self.b2 = orig_W1, orig_b1, orig_W2, orig_b2
            
            return loss
        
        # Flatten parameters
        params_flat = np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten()
        ])
        
        # Compute numerical gradients
        numerical_grad = np.zeros_like(params_flat)
        for i in range(len(params_flat)):
            params_plus = params_flat.copy()
            params_minus = params_flat.copy()
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            
            numerical_grad[i] = (loss_function(params_plus) - loss_function(params_minus)) / (2 * epsilon)
        
        # Flatten analytical gradients
        analytical_grad = np.concatenate([
            gradients['dW1'].flatten(),
            gradients['db1'].flatten(),
            gradients['dW2'].flatten(),
            gradients['db2'].flatten()
        ])
        
        # Compute relative errors
        relative_errors = {}
        param_names = ['W1', 'b1', 'W2', 'b2']
        param_sizes = [self.input_size * self.hidden_size, self.hidden_size, 
                      self.hidden_size * self.output_size, self.output_size]
        
        start_idx = 0
        for name, size in zip(param_names, param_sizes):
            end_idx = start_idx + size
            analytical = analytical_grad[start_idx:end_idx]
            numerical = numerical_grad[start_idx:end_idx]
            
            relative_error = np.linalg.norm(analytical - numerical) / (np.linalg.norm(analytical) + np.linalg.norm(numerical))
            relative_errors[f'{name}_relative_error'] = relative_error
            
            start_idx = end_idx
        
        return relative_errors


def main():
    """Demonstrate neural network functionality."""
    print("Neural Network Demo")
    print("=" * 25)
    
    # Generate sample data
    from .data_generator import DataGenerator
    from .utils import train_test_split
    
    generator = DataGenerator(random_state=42)
    X, y = generator.generate_xor_data(n_samples=400, noise=0.1)
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = NeuralNetwork(input_size=2, hidden_size=8, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, epochs=1000, verbose=True)
    
    # Evaluate model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get detailed metrics
    metrics = model.get_metrics(X_test, y_test)
    print("\nDetailed Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Plot training history
    model.plot_training_history()
    
    # Get model info
    info = model.get_model_info()
    print(f"\nModel Info:")
    print(f"  Architecture: {info['architecture']}")
    print(f"  Total Parameters: {info['total_parameters']}")
    print(f"  Epochs: {info['n_epochs']}")
    print(f"  Final Loss: {info['final_loss']:.6f}")
    print(f"  Final Accuracy: {info['final_accuracy']:.4f}")
    
    # Gradient checking
    print("\nGradient Checking:")
    gradient_errors = model.gradient_check(X_train[:10], y_train[:10])
    for param, error in gradient_errors.items():
        print(f"  {param}: {error:.2e}")


if __name__ == "__main__":
    main()
