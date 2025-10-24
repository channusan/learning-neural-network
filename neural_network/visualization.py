"""
Visualization utilities for Neural Network Classification project.

This module provides comprehensive plotting functions for data visualization,
decision boundaries, training progress, and model comparison.

Author: AI Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Tuple, Optional, Dict, Any, List
import warnings


def plot_data_distribution(X: np.ndarray, y: np.ndarray, title: str = "Data Distribution",
                          save_path: Optional[str] = None) -> None:
    """
    Plot the distribution of 2D data points with different colors for each class.
    
    Args:
        X: Feature matrix of shape (n_samples, 2)
        y: Binary labels of shape (n_samples,)
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


def plot_decision_boundary_2d(model, X: np.ndarray, y: np.ndarray, 
                             title: str = "Decision Boundary", resolution: int = 100,
                             save_path: Optional[str] = None) -> None:
    """
    Plot decision boundary for a 2D classification model.
    
    Args:
        model: Trained model with predict_proba method
        X: Feature matrix of shape (n_samples, 2)
        y: Binary labels of shape (n_samples,)
        title: Title for the plot
        resolution: Resolution of the decision boundary grid
        save_path: Optional path to save the plot
    """
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                        np.linspace(y_min, y_max, resolution))
    
    # Make predictions on the mesh grid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.colorbar(label='Predicted Probability')
    
    # Plot data points
    class_0_mask = y == 0
    class_1_mask = y == 1
    
    plt.scatter(X[class_0_mask, 0], X[class_0_mask, 1], 
               c='red', marker='o', label='Class 0', alpha=0.8, s=50, edgecolors='black')
    plt.scatter(X[class_1_mask, 0], X[class_1_mask, 1], 
               c='blue', marker='s', label='Class 1', alpha=0.8, s=50, edgecolors='black')
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(loss_history: List[float], accuracy_history: List[float],
                         val_loss_history: Optional[List[float]] = None,
                         val_accuracy_history: Optional[List[float]] = None,
                         title: str = "Training History", save_path: Optional[str] = None) -> None:
    """
    Plot training loss and accuracy over epochs.
    
    Args:
        loss_history: List of training losses
        accuracy_history: List of training accuracies
        val_loss_history: Optional list of validation losses
        val_accuracy_history: Optional list of validation accuracies
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(loss_history, 'b-', linewidth=2, label='Training Loss')
    if val_loss_history is not None:
        ax1.plot(val_loss_history, 'r--', linewidth=2, label='Validation Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(accuracy_history, 'b-', linewidth=2, label='Training Accuracy')
    if val_accuracy_history is not None:
        ax2.plot(val_accuracy_history, 'r--', linewidth=2, label='Validation Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(models: Dict[str, Any], X: np.ndarray, y: np.ndarray,
                         titles: Optional[Dict[str, str]] = None,
                         save_path: Optional[str] = None) -> None:
    """
    Plot decision boundaries for multiple models side by side.
    
    Args:
        models: Dictionary of model_name -> model_object
        X: Feature matrix
        y: Binary labels
        titles: Optional dictionary of model_name -> title
        save_path: Optional path to save the plot
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (model_name, model) in enumerate(models.items()):
        ax = axes[i]
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                            np.linspace(y_min, y_max, 50))
        
        # Make predictions
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict_proba(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        contour = ax.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap=plt.cm.RdYlBu)
        
        # Plot data points
        class_0_mask = y == 0
        class_1_mask = y == 1
        
        ax.scatter(X[class_0_mask, 0], X[class_0_mask, 1], 
                  c='red', marker='o', label='Class 0', alpha=0.8, s=30, edgecolors='black')
        ax.scatter(X[class_1_mask, 0], X[class_1_mask, 1], 
                  c='blue', marker='s', label='Class 1', alpha=0.8, s=30, edgecolors='black')
        
        # Set title
        title = titles.get(model_name, model_name) if titles else model_name
        ax.set_title(title)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         title: str = "Confusion Matrix", save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    from .utils import calculate_confusion_matrix
    
    cm = calculate_confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0, 1], ['Class 0', 'Class 1'])
    plt.yticks([0, 1], ['Class 0', 'Class 1'])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                           title: str = "Model Performance Comparison",
                           save_path: Optional[str] = None) -> None:
    """
    Plot bar chart comparing metrics across different models.
    
    Args:
        metrics_dict: Dictionary of model_name -> metrics_dict
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict[models[0]].keys())
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, model in enumerate(models):
        values = [metrics_dict[model][metric] for metric in metrics]
        ax.bar(x + i * width, values, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, model in enumerate(models):
        values = [metrics_dict[model][metric] for metric in metrics]
        for j, v in enumerate(values):
            ax.text(j + i * width, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(weights: np.ndarray, feature_names: List[str] = None,
                           title: str = "Feature Importance", save_path: Optional[str] = None) -> None:
    """
    Plot feature importance based on model weights.
    
    Args:
        weights: Model weights
        feature_names: Names of features
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(len(weights))]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(feature_names, np.abs(weights), alpha=0.7)
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Absolute Weight Value')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, weight in zip(bars, weights):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{abs(weight):.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_learning_curves(train_sizes: List[int], train_scores: List[float], 
                        val_scores: List[float], title: str = "Learning Curves",
                        save_path: Optional[str] = None) -> None:
    """
    Plot learning curves showing performance vs training set size.
    
    Args:
        train_sizes: List of training set sizes
        train_scores: List of training scores
        val_scores: List of validation scores
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(train_sizes, train_scores, 'o-', label='Training Score', linewidth=2)
    plt.plot(train_sizes, val_scores, 'o-', label='Validation Score', linewidth=2)
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_summary_plot(models: Dict[str, Any], X: np.ndarray, y: np.ndarray,
                       metrics_dict: Dict[str, Dict[str, float]],
                       save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive summary plot with multiple subplots.
    
    Args:
        models: Dictionary of model_name -> model_object
        X: Feature matrix
        y: Binary labels
        metrics_dict: Dictionary of model_name -> metrics_dict
        save_path: Optional path to save the plot
    """
    n_models = len(models)
    fig = plt.figure(figsize=(15, 10))
    
    # Create subplot layout
    gs = fig.add_gridspec(3, n_models, hspace=0.3, wspace=0.3)
    
    # Plot decision boundaries
    for i, (model_name, model) in enumerate(models.items()):
        ax = fig.add_subplot(gs[0, i])
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                            np.linspace(y_min, y_max, 50))
        
        # Make predictions
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict_proba(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap=plt.cm.RdYlBu)
        
        # Plot data points
        class_0_mask = y == 0
        class_1_mask = y == 1
        
        ax.scatter(X[class_0_mask, 0], X[class_0_mask, 1], 
                  c='red', marker='o', label='Class 0', alpha=0.8, s=20, edgecolors='black')
        ax.scatter(X[class_1_mask, 0], X[class_1_mask, 1], 
                  c='blue', marker='s', label='Class 1', alpha=0.8, s=20, edgecolors='black')
        
        ax.set_title(f'{model_name} Decision Boundary')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.grid(True, alpha=0.3)
    
    # Plot metrics comparison
    ax_metrics = fig.add_subplot(gs[1, :])
    models_list = list(metrics_dict.keys())
    metrics_list = list(metrics_dict[models_list[0]].keys())
    
    x = np.arange(len(metrics_list))
    width = 0.35
    
    for i, model in enumerate(models_list):
        values = [metrics_dict[model][metric] for metric in metrics_list]
        ax_metrics.bar(x + i * width, values, width, label=model, alpha=0.8)
    
    ax_metrics.set_xlabel('Metrics')
    ax_metrics.set_ylabel('Score')
    ax_metrics.set_title('Performance Metrics Comparison')
    ax_metrics.set_xticks(x + width / 2)
    ax_metrics.set_xticklabels(metrics_list)
    ax_metrics.legend()
    ax_metrics.grid(True, alpha=0.3)
    
    # Plot training history (if available)
    ax_history = fig.add_subplot(gs[2, :])
    
    for model_name, model in models.items():
        if hasattr(model, 'loss_history') and model.loss_history:
            ax_history.plot(model.loss_history, label=f'{model_name} Loss', linewidth=2)
    
    ax_history.set_xlabel('Epoch')
    ax_history.set_ylabel('Loss')
    ax_history.set_title('Training Loss History')
    ax_history.legend()
    ax_history.grid(True, alpha=0.3)
    
    plt.suptitle('Neural Network Classification - Complete Analysis', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    """Demonstrate visualization functions."""
    print("Visualization Demo")
    print("=" * 20)
    
    # Generate sample data
    from .data_generator import DataGenerator
    from .logistic_regression import LogisticRegression
    from .neural_network import NeuralNetwork
    from .utils import train_test_split
    
    generator = DataGenerator(random_state=42)
    X, y = generator.generate_xor_data(n_samples=400, noise=0.1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train models
    lr_model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    lr_model.fit(X_train, y_train)
    
    nn_model = NeuralNetwork(learning_rate=0.1, random_state=42)
    nn_model.fit(X_train, y_train, epochs=1000)
    
    # Plot data distribution
    plot_data_distribution(X, y, "XOR Dataset Distribution")
    
    # Plot decision boundaries
    models = {
        'Logistic Regression': lr_model,
        'Neural Network': nn_model
    }
    plot_model_comparison(models, X_test, y_test)
    
    # Plot training history
    plot_training_history(nn_model.loss_history, nn_model.accuracy_history)
    
    # Plot metrics comparison
    metrics_dict = {
        'Logistic Regression': lr_model.get_metrics(X_test, y_test),
        'Neural Network': nn_model.get_metrics(X_test, y_test)
    }
    plot_metrics_comparison(metrics_dict)
    
    # Create comprehensive summary
    create_summary_plot(models, X_test, y_test, metrics_dict)


if __name__ == "__main__":
    main()



