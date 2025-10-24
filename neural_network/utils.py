"""
Utility functions for the Neural Network Classification project.

This module contains helper functions for data preprocessing, metrics calculation,
and other common operations used throughout the project.

Author: AI Assistant
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training and testing sets.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
        test_size: Proportion of dataset to use for testing (0.0 to 1.0)
        random_state: Random seed for reproducible splits
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Create random indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def normalize_features(X: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize features using specified method.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        method: Normalization method ('standard', 'minmax', or 'none')
        
    Returns:
        Tuple of (normalized_X, normalization_params)
    """
    if method == 'none':
        return X, {}
    
    X_normalized = X.copy()
    params = {}
    
    if method == 'standard':
        # Z-score normalization: (x - mean) / std
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        X_normalized = (X - mean) / std
        params = {'mean': mean, 'std': std}
        
    elif method == 'minmax':
        # Min-max normalization: (x - min) / (max - min)
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        # Avoid division by zero
        range_vals = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
        X_normalized = (X - min_vals) / range_vals
        params = {'min': min_vals, 'max': max_vals}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return X_normalized, params


def denormalize_features(X: np.ndarray, params: Dict[str, Any], method: str = 'standard') -> np.ndarray:
    """
    Denormalize features using stored parameters.
    
    Args:
        X: Normalized feature matrix
        params: Normalization parameters from normalize_features
        method: Normalization method used
        
    Returns:
        Denormalized feature matrix
    """
    if method == 'none' or not params:
        return X
    
    X_denormalized = X.copy()
    
    if method == 'standard':
        mean = params['mean']
        std = params['std']
        X_denormalized = X * std + mean
        
    elif method == 'minmax':
        min_vals = params['min']
        max_vals = params['max']
        range_vals = max_vals - min_vals
        X_denormalized = X * range_vals + min_vals
    
    return X_denormalized


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    return np.mean(y_true == y_pred)


def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate precision score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Precision score (0.0 to 1.0)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # True positives: predicted positive and actually positive
    tp = np.sum((y_pred == 1) & (y_true == 1))
    # False positives: predicted positive but actually negative
    fp = np.sum((y_pred == 1) & (y_true == 0))
    
    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)


def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate recall score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Recall score (0.0 to 1.0)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # True positives: predicted positive and actually positive
    tp = np.sum((y_pred == 1) & (y_true == 1))
    # False negatives: predicted negative but actually positive
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)


def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        2x2 confusion matrix
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Create confusion matrix
    # [[TN, FP],
    #  [FN, TP]]
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    
    return np.array([[tn, fp], [fn, tp]])


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing all metrics
    """
    return {
        'accuracy': calculate_accuracy(y_true, y_pred),
        'precision': calculate_precision(y_true, y_pred),
        'recall': calculate_recall(y_true, y_pred),
        'f1_score': calculate_f1_score(y_true, y_pred)
    }


def add_bias_term(X: np.ndarray) -> np.ndarray:
    """
    Add bias term (column of ones) to feature matrix.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        
    Returns:
        Feature matrix with bias term of shape (n_samples, n_features + 1)
    """
    return np.column_stack([np.ones(X.shape[0]), X])


def remove_bias_term(X: np.ndarray) -> np.ndarray:
    """
    Remove bias term (first column) from feature matrix.
    
    Args:
        X: Feature matrix with bias term of shape (n_samples, n_features + 1)
        
    Returns:
        Feature matrix without bias term of shape (n_samples, n_features - 1)
    """
    return X[:, 1:]


def check_gradient_numerical(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Check gradient using numerical differentiation.
    
    Args:
        f: Function to compute gradient for
        x: Point at which to compute gradient
        h: Step size for numerical differentiation
        
    Returns:
        Numerical gradient
    """
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    
    return grad


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducible results.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)


def validate_input_data(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """
    Validate input data for common issues.
    
    Args:
        X: Feature matrix
        y: Optional label vector
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    
    if X.shape[0] == 0:
        raise ValueError("X cannot be empty")
    
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or infinite values")
    
    if y is not None:
        if len(y) != X.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or infinite values")
        
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("y must contain exactly 2 unique classes")
        
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError("y must contain only 0s and 1s")


def print_metrics(metrics: Dict[str, float], title: str = "Model Performance") -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics display
    """
    print(f"\n{title}")
    print("-" * len(title))
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")


def main():
    """Demonstrate utility functions."""
    print("Neural Network Utils Demo")
    print("=" * 30)
    
    # Generate sample data
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 2, 100)
    
    # Test train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Test normalization
    X_norm, params = normalize_features(X_train, method='standard')
    print(f"Normalized X shape: {X_norm.shape}")
    
    # Test metrics calculation
    y_pred = np.random.randint(0, 2, len(y_test))
    metrics = calculate_all_metrics(y_test, y_pred)
    print_metrics(metrics, "Sample Metrics")


if __name__ == "__main__":
    main()



