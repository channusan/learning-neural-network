"""
Synthetic Data Generator for Neural Network Classification

This module generates synthetic 2D datasets that are non-linearly separable,
making them ideal for demonstrating the advantages of neural networks over
linear classifiers like logistic regression.

Author: AI Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings


class DataGenerator:
    """
    Generates synthetic 2D classification datasets for educational purposes.
    
    This class creates datasets that are challenging for linear classifiers
    but can be solved effectively by neural networks with non-linear activation
    functions.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the data generator.
        
        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_xor_data(self, n_samples: int = 400, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate XOR-like dataset (non-linearly separable).
        
        Creates a dataset where the classification boundary forms an XOR pattern,
        which is impossible for linear classifiers to solve perfectly.
        
        Args:
            n_samples: Number of samples to generate (default: 400)
            noise: Amount of noise to add (0.0 = no noise, 1.0 = maximum noise)
            
        Returns:
            Tuple of (X, y) where:
            - X: 2D array of shape (n_samples, 2) containing coordinates
            - y: 1D array of shape (n_samples,) containing binary labels (0 or 1)
        """
        if n_samples < 4:
            raise ValueError("n_samples must be at least 4")
        if not 0 <= noise <= 1:
            raise ValueError("noise must be between 0 and 1")
        
        # Generate samples in four quadrants for XOR pattern
        n_per_quadrant = n_samples // 4
        remainder = n_samples % 4
        
        # Quadrant 1: positive class (top-right)
        x1_q1 = np.random.uniform(0.5, 1.0, n_per_quadrant + (1 if remainder > 0 else 0))
        x2_q1 = np.random.uniform(0.5, 1.0, n_per_quadrant + (1 if remainder > 0 else 0))
        y_q1 = np.ones(n_per_quadrant + (1 if remainder > 0 else 0))
        
        # Quadrant 2: negative class (top-left)
        x1_q2 = np.random.uniform(-1.0, -0.5, n_per_quadrant + (1 if remainder > 1 else 0))
        x2_q2 = np.random.uniform(0.5, 1.0, n_per_quadrant + (1 if remainder > 1 else 0))
        y_q2 = np.zeros(n_per_quadrant + (1 if remainder > 1 else 0))
        
        # Quadrant 3: positive class (bottom-left)
        x1_q3 = np.random.uniform(-1.0, -0.5, n_per_quadrant + (1 if remainder > 2 else 0))
        x2_q3 = np.random.uniform(-1.0, -0.5, n_per_quadrant + (1 if remainder > 2 else 0))
        y_q3 = np.ones(n_per_quadrant + (1 if remainder > 2 else 0))
        
        # Quadrant 4: negative class (bottom-right)
        x1_q4 = np.random.uniform(0.5, 1.0, n_per_quadrant)
        x2_q4 = np.random.uniform(-1.0, -0.5, n_per_quadrant)
        y_q4 = np.zeros(n_per_quadrant)
        
        # Combine all quadrants
        X = np.vstack([
            np.column_stack([x1_q1, x2_q1]),
            np.column_stack([x1_q2, x2_q2]),
            np.column_stack([x1_q3, x2_q3]),
            np.column_stack([x1_q4, x2_q4])
        ])
        y = np.hstack([y_q1, y_q2, y_q3, y_q4])
        
        # Add noise to make the problem more realistic
        if noise > 0:
            noise_scale = noise * 0.1  # Scale noise appropriately
            X += np.random.normal(0, noise_scale, X.shape)
        
        # Shuffle the data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Ensure y is integer type
        y = y.astype(int)
        
        return X, y
    
    def generate_circle_data(self, n_samples: int = 400, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate concentric circles dataset (non-linearly separable).
        
        Creates a dataset with two concentric circles where the inner circle
        is one class and the outer ring is another class.
        
        Args:
            n_samples: Number of samples to generate
            noise: Amount of noise to add
            
        Returns:
            Tuple of (X, y) containing features and labels
        """
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        
        # Generate inner circle (class 1)
        n_inner = n_samples // 2
        theta_inner = np.random.uniform(0, 2 * np.pi, n_inner)
        r_inner = np.random.uniform(0, 0.3, n_inner)
        x1_inner = r_inner * np.cos(theta_inner)
        x2_inner = r_inner * np.sin(theta_inner)
        y_inner = np.ones(n_inner)
        
        # Generate outer circle (class 0)
        n_outer = n_samples - n_inner
        theta_outer = np.random.uniform(0, 2 * np.pi, n_outer)
        r_outer = np.random.uniform(0.5, 1.0, n_outer)
        x1_outer = r_outer * np.cos(theta_outer)
        x2_outer = r_outer * np.sin(theta_outer)
        y_outer = np.zeros(n_outer)
        
        # Combine data
        X = np.vstack([
            np.column_stack([x1_inner, x2_inner]),
            np.column_stack([x1_outer, x2_outer])
        ])
        y = np.hstack([y_inner, y_outer])
        
        # Add noise
        if noise > 0:
            noise_scale = noise * 0.1
            X += np.random.normal(0, noise_scale, X.shape)
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Ensure y is integer type
        y = y.astype(int)
        
        return X, y
    
    def generate_moon_data(self, n_samples: int = 400, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate two interleaving half circles (moons) dataset.
        
        This is a classic non-linearly separable dataset where two classes
        form interleaving half circles.
        
        Args:
            n_samples: Number of samples to generate
            noise: Amount of noise to add
            
        Returns:
            Tuple of (X, y) containing features and labels
        """
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        
        # Generate first moon (class 0)
        n_moon1 = n_samples // 2
        theta1 = np.random.uniform(0, np.pi, n_moon1)
        x1_moon1 = np.cos(theta1) + np.random.normal(0, noise, n_moon1)
        x2_moon1 = np.sin(theta1) + np.random.normal(0, noise, n_moon1)
        y_moon1 = np.zeros(n_moon1)
        
        # Generate second moon (class 1)
        n_moon2 = n_samples - n_moon1
        theta2 = np.random.uniform(0, np.pi, n_moon2)
        x1_moon2 = 1 - np.cos(theta2) + np.random.normal(0, noise, n_moon2)
        x2_moon2 = 0.5 - np.sin(theta2) + np.random.normal(0, noise, n_moon2)
        y_moon2 = np.ones(n_moon2)
        
        # Combine data
        X = np.vstack([
            np.column_stack([x1_moon1, x2_moon1]),
            np.column_stack([x1_moon2, x2_moon2])
        ])
        y = np.hstack([y_moon1, y_moon2])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Ensure y is integer type
        y = y.astype(int)
        
        return X, y
    
    def plot_data(self, X: np.ndarray, y: np.ndarray, title: str = "Synthetic Dataset", 
                  save_path: Optional[str] = None) -> None:
        """
        Plot the generated dataset.
        
        Args:
            X: Feature matrix of shape (n_samples, 2)
            y: Label vector of shape (n_samples,)
            title: Title for the plot
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        # Plot class 0
        class_0_mask = y == 0
        plt.scatter(X[class_0_mask, 0], X[class_0_mask, 1], 
                   c='red', marker='o', label='Class 0', alpha=0.7, s=50)
        
        # Plot class 1
        class_1_mask = y == 1
        plt.scatter(X[class_1_mask, 0], X[class_1_mask, 1], 
                   c='blue', marker='s', label='Class 1', alpha=0.7, s=50)
        
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_data_info(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Get information about the generated dataset.
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            Dictionary containing dataset statistics
        """
        return {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'class_distribution': {
                'class_0': np.sum(y == 0),
                'class_1': np.sum(y == 1)
            },
            'feature_ranges': {
                'x1_min': np.min(X[:, 0]),
                'x1_max': np.max(X[:, 0]),
                'x2_min': np.min(X[:, 1]),
                'x2_max': np.max(X[:, 1])
            }
        }


def main():
    """Demonstrate the data generator functionality."""
    print("Neural Network Data Generator Demo")
    print("=" * 40)
    
    # Create data generator
    generator = DataGenerator(random_state=42)
    
    # Generate different types of datasets
    datasets = {
        'XOR': generator.generate_xor_data(n_samples=400, noise=0.1),
        'Circles': generator.generate_circle_data(n_samples=400, noise=0.1),
        'Moons': generator.generate_moon_data(n_samples=400, noise=0.1)
    }
    
    # Display information about each dataset
    for name, (X, y) in datasets.items():
        print(f"\n{name} Dataset:")
        info = generator.get_data_info(X, y)
        print(f"  Samples: {info['n_samples']}")
        print(f"  Features: {info['n_features']}")
        print(f"  Class 0: {info['class_distribution']['class_0']}")
        print(f"  Class 1: {info['class_distribution']['class_1']}")
        
        # Plot the dataset
        generator.plot_data(X, y, title=f"{name} Dataset")


if __name__ == "__main__":
    main()
