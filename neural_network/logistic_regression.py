"""
Logistic Regression Implementation

This module implements logistic regression from scratch for binary classification,
serving as a baseline linear classifier to compare against neural networks.

Author: AI Assistant
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict, Any
import warnings
from .activations import sigmoid, sigmoid_derivative
from .loss_functions import binary_cross_entropy, binary_cross_entropy_derivative
from .utils import add_bias_term, validate_input_data, calculate_all_metrics


class LogisticRegression:
    """
    Logistic Regression classifier for binary classification.
    
    This implementation uses gradient descent optimization and serves as a
    baseline linear classifier to demonstrate the limitations of linear
    decision boundaries on non-linearly separable data.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6, random_state: int = 42):
        """
        Initialize logistic regression classifier.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            random_state: Random seed for reproducible results
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        
        # Model parameters
        self.weights = None
        self.bias = None
        self.is_fitted = False
        self.training_time_ms = 0.0
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
    
    def _initialize_weights(self, n_features: int) -> None:
        """
        Initialize model weights and bias.
        
        Args:
            n_features: Number of input features
        """
        np.random.seed(self.random_state)
        # Initialize weights with small random values
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid activation function.
        
        Args:
            x: Input values
            
        Returns:
            Sigmoid of input values
        """
        return sigmoid(x)
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Linear combination: z = X * weights + bias
        z = np.dot(X, self.weights) + self.bias
        
        # Apply sigmoid to get probabilities
        probabilities = self._sigmoid(z)
        
        return probabilities
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Loss value
        """
        y_pred = self._predict_proba(X)
        return binary_cross_entropy(y, y_pred)
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weights and bias.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Tuple of (weight_gradients, bias_gradient)
        """
        n_samples = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)
        
        # Compute error
        error = y_pred - y
        
        # Compute gradients
        weight_gradients = np.dot(X.T, error) / n_samples
        bias_gradient = np.mean(error)
        
        return weight_gradients, bias_gradient
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'LogisticRegression':
        """
        Fit the logistic regression model.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Binary labels of shape (n_samples,)
            verbose: Whether to print training progress
            
        Returns:
            Self (for method chaining)
        """
        # Validate input data
        validate_input_data(X, y)
        
        # Initialize weights
        n_features = X.shape[1]
        self._initialize_weights(n_features)
        
        # Training loop
        print(f"   Starting training for up to {self.max_iterations} iterations...")
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            # Compute predictions and loss
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)
            loss = binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)
            
            # Compute accuracy
            y_pred_binary = (y_pred > 0.5).astype(int)
            accuracy = np.mean(y == y_pred_binary)
            self.accuracy_history.append(accuracy)
            
            # Print progress
            if verbose and iteration % 100 == 0:
                print(f"     Iteration {iteration}: Loss = {loss:.6f}, Accuracy = {accuracy:.4f}")
            
            # Compute gradients
            weight_gradients, bias_gradient = self._compute_gradients(X, y)
            
            # Update parameters
            self.weights -= self.learning_rate * weight_gradients
            self.bias -= self.learning_rate * bias_gradient
            
            # Check convergence
            if iteration > 0:
                loss_change = abs(self.loss_history[-1] - self.loss_history[-2])
                if loss_change < self.tolerance:
                    if verbose:
                        print(f"     Converged at iteration {iteration}")
                    break
        
        # Calculate training time
        end_time = time.time()
        self.training_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        
        print(f"   ✓ Training completed after {len(self.loss_history)} iterations")
        print(f"   ✓ Training time: {self.training_time_ms:.2f} ms")
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities of shape (n_samples,)
        """
        return self._predict_proba(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted binary labels of shape (n_samples,)
        """
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.
        
        Args:
            X: Feature matrix
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
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of performance metrics
        """
        y_pred = self.predict(X)
        return calculate_all_metrics(y, y_pred)
    
    def get_decision_boundary_coefficients(self) -> Tuple[np.ndarray, float]:
        """
        Get coefficients for decision boundary equation.
        
        For logistic regression, the decision boundary is:
        w1*x1 + w2*x2 + b = 0
        
        Returns:
            Tuple of (weights, bias)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting coefficients")
        
        return self.weights.copy(), self.bias
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training loss and accuracy history.
        
        Args:
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting history")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.loss_history, 'b-', linewidth=2)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Binary Cross-Entropy Loss')
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(self.accuracy_history, 'r-', linewidth=2)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary containing model information
        """
        if not self.is_fitted:
            return {"is_fitted": False}
        
        return {
            "is_fitted": True,
            "n_features": len(self.weights),
            "weights": self.weights.copy(),
            "bias": self.bias,
            "n_iterations": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "final_accuracy": self.accuracy_history[-1] if self.accuracy_history else None,
            "learning_rate": self.learning_rate,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "training_time_ms": self.training_time_ms
        }


def main():
    """Demonstrate logistic regression functionality."""
    print("Logistic Regression Demo")
    print("=" * 30)
    
    # Generate sample data
    from .data_generator import DataGenerator
    
    generator = DataGenerator(random_state=42)
    X, y = generator.generate_xor_data(n_samples=400, noise=0.1)
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    from .utils import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = LogisticRegression(learning_rate=0.1, max_iterations=1000, verbose=True)
    model.fit(X_train, y_train)
    
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
    print(f"  Features: {info['n_features']}")
    print(f"  Iterations: {info['n_iterations']}")
    print(f"  Final Loss: {info['final_loss']:.6f}")
    print(f"  Final Accuracy: {info['final_accuracy']:.4f}")


if __name__ == "__main__":
    main()
