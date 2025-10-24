"""
Neural Network Classification Learning Project

A comprehensive implementation of a 2-class classification neural network
with a single hidden layer for educational purposes.

This package includes:
- Synthetic data generation
- Logistic regression baseline
- Neural network implementation
- Training and optimization
- Visualization tools
- Comprehensive testing

Author: AI Assistant
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

# Import main classes for easy access
from .neural_network import NeuralNetwork
from .logistic_regression import LogisticRegression
from .data_generator import DataGenerator
from .trainer import Trainer

__all__ = [
    'NeuralNetwork',
    'LogisticRegression', 
    'DataGenerator',
    'Trainer'
]



