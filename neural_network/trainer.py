"""
Training utilities for Neural Network Classification project.

This module provides training configuration, early stopping, and other
training-related utilities for both logistic regression and neural networks.

Author: AI Assistant
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import warnings
from .utils import train_test_split, calculate_all_metrics


@dataclass
class TrainingConfig:
    """
    Configuration class for training parameters.
    """
    learning_rate: float = 0.1
    max_epochs: int = 1000
    batch_size: Optional[int] = None  # None means full batch
    early_stopping: bool = True
    patience: int = 50
    min_delta: float = 1e-6
    validation_split: float = 0.2
    random_state: int = 42
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if not 0 <= self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")
        if self.patience <= 0:
            raise ValueError("patience must be positive")


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    """
    
    def __init__(self, patience: int = 50, min_delta: float = 1e-6, 
                 monitor: str = 'val_loss', mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor ('val_loss', 'val_accuracy', 'loss', 'accuracy')
            mode: 'min' for loss metrics, 'max' for accuracy metrics
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")
    
    def __call__(self, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop early.
        
        Args:
            metrics: Dictionary of current metrics
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.monitor not in metrics:
            warnings.warn(f"Monitor metric '{self.monitor}' not found in metrics")
            return False
        
        current_score = metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = current_score
        elif self._is_improvement(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current score is an improvement over best score."""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


class LearningRateScheduler:
    """
    Learning rate scheduling utility.
    """
    
    def __init__(self, initial_lr: float, schedule_type: str = 'constant', 
                 **kwargs):
        """
        Initialize learning rate scheduler.
        
        Args:
            initial_lr: Initial learning rate
            schedule_type: Type of schedule ('constant', 'step', 'exponential', 'cosine')
            **kwargs: Additional parameters for specific schedules
        """
        self.initial_lr = initial_lr
        self.schedule_type = schedule_type
        self.kwargs = kwargs
        self.epoch = 0
        
        if schedule_type not in ['constant', 'step', 'exponential', 'cosine']:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    def __call__(self, epoch: int) -> float:
        """
        Get learning rate for given epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Learning rate for this epoch
        """
        self.epoch = epoch
        
        if self.schedule_type == 'constant':
            return self.initial_lr
        elif self.schedule_type == 'step':
            step_size = self.kwargs.get('step_size', 100)
            gamma = self.kwargs.get('gamma', 0.1)
            return self.initial_lr * (gamma ** (epoch // step_size))
        elif self.schedule_type == 'exponential':
            gamma = self.kwargs.get('gamma', 0.95)
            return self.initial_lr * (gamma ** epoch)
        elif self.schedule_type == 'cosine':
            max_epochs = self.kwargs.get('max_epochs', 1000)
            min_lr = self.kwargs.get('min_lr', 0.001)
            return min_lr + (self.initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / max_epochs)) / 2


class Trainer:
    """
    Generic trainer class for machine learning models.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            monitor='val_loss'
        ) if config.early_stopping else None
        
        self.lr_scheduler = LearningRateScheduler(
            initial_lr=config.learning_rate,
            schedule_type='constant'
        )
        
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def train_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train a model with the given configuration.
        
        Args:
            model: Model to train (must have fit method)
            X: Training features
            y: Training labels
            
        Returns:
            Dictionary containing training results
        """
        # Split data if validation split is specified
        if self.config.validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.validation_split, 
                random_state=self.config.random_state
            )
        else:
            X_train, X_val, y_train, y_val = X, None, y, None
        
        # Train model
        if hasattr(model, 'fit'):
            # Check if model supports epochs parameter (neural network)
            if hasattr(model, '_forward_propagation'):  # Neural network
                if X_val is not None:
                    model.fit(X_train, y_train, epochs=self.config.max_epochs, 
                             verbose=self.config.verbose, validation_data=(X_val, y_val))
                else:
                    model.fit(X_train, y_train, epochs=self.config.max_epochs, 
                             verbose=self.config.verbose)
            else:  # Logistic regression
                model.fit(X_train, y_train, verbose=self.config.verbose)
        else:
            raise ValueError("Model must have a 'fit' method")
        
        # Store training history
        self.training_history = {
            'train_loss': getattr(model, 'loss_history', []),
            'train_accuracy': getattr(model, 'accuracy_history', []),
            'val_loss': getattr(model, 'val_loss_history', []),
            'val_accuracy': getattr(model, 'val_accuracy_history', [])
        }
        
        return {
            'model': model,
            'training_history': self.training_history,
            'config': self.config
        }
    
    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if hasattr(model, 'get_metrics'):
            return model.get_metrics(X, y)
        elif hasattr(model, 'score'):
            return {'accuracy': model.score(X, y)}
        else:
            raise ValueError("Model must have 'get_metrics' or 'score' method")
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path: Optional path to save the plot
        """
        from .visualization import plot_training_history
        
        plot_training_history(
            self.training_history['train_loss'],
            self.training_history['train_accuracy'],
            self.training_history['val_loss'] if self.training_history['val_loss'] else None,
            self.training_history['val_accuracy'] if self.training_history['val_accuracy'] else None,
            save_path=save_path
        )


def train_and_compare_models(X: np.ndarray, y: np.ndarray, 
                           models: Dict[str, Any], 
                           config: TrainingConfig) -> Dict[str, Any]:
    """
    Train multiple models and compare their performance.
    
    Args:
        X: Training features
        y: Training labels
        models: Dictionary of model_name -> model_instance
        config: Training configuration
        
    Returns:
        Dictionary containing training results for all models
    """
    results = {}
    trainer = Trainer(config)
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        print(f"   Initializing training for {model_name}...")
        
        # Train model
        print(f"   Starting training process...")
        training_result = trainer.train_model(model, X, y)
        results[model_name] = training_result
        print(f"   ✓ Training completed for {model_name}")
        
        # Evaluate model
        print(f"   Preparing test data for evaluation...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.random_state
        )
        
        print(f"   Computing test metrics...")
        metrics = trainer.evaluate_model(model, X_test, y_test)
        results[model_name]['test_metrics'] = metrics
        
        print(f"   ✓ {model_name} - Test Accuracy: {metrics.get('accuracy', 0):.4f}")
    
    return results


def create_training_report(results: Dict[str, Any]) -> str:
    """
    Create a comprehensive training report.
    
    Args:
        results: Results from train_and_compare_models
        
    Returns:
        Formatted training report string
    """
    report = "=" * 60 + "\n"
    report += "NEURAL NETWORK CLASSIFICATION - TRAINING REPORT\n"
    report += "=" * 60 + "\n\n"
    
    for model_name, result in results.items():
        report += f"Model: {model_name}\n"
        report += "-" * 30 + "\n"
        
        # Model info
        if hasattr(result['model'], 'get_model_info'):
            info = result['model'].get_model_info()
            report += f"Architecture: {info.get('architecture', 'N/A')}\n"
            report += f"Total Parameters: {info.get('total_parameters', 'N/A')}\n"
            report += f"Epochs Trained: {info.get('n_epochs', 'N/A')}\n"
            report += f"Training Time: {info.get('training_time_ms', 'N/A'):.2f} ms\n"
        
        # Training metrics
        history = result['training_history']
        if history['train_loss']:
            report += f"Final Training Loss: {history['train_loss'][-1]:.6f}\n"
            report += f"Final Training Accuracy: {history['train_accuracy'][-1]:.4f}\n"
        
        if history['val_loss']:
            report += f"Final Validation Loss: {history['val_loss'][-1]:.6f}\n"
            report += f"Final Validation Accuracy: {history['val_accuracy'][-1]:.4f}\n"
        
        # Test metrics
        test_metrics = result['test_metrics']
        report += "Test Performance:\n"
        for metric, value in test_metrics.items():
            report += f"  {metric.capitalize()}: {value:.4f}\n"
        
        report += "\n"
    
    return report


def main():
    """Demonstrate trainer functionality."""
    print("Trainer Demo")
    print("=" * 15)
    
    # Generate sample data
    from .data_generator import DataGenerator
    from .logistic_regression import LogisticRegression
    from .neural_network import NeuralNetwork
    
    generator = DataGenerator(random_state=42)
    X, y = generator.generate_xor_data(n_samples=400, noise=0.1)
    
    # Create training configuration
    config = TrainingConfig(
        learning_rate=0.1,
        max_epochs=1000,
        early_stopping=True,
        patience=50,
        validation_split=0.2,
        verbose=True
    )
    
    # Create models
    models = {
        'Logistic Regression': LogisticRegression(learning_rate=0.1, max_iterations=1000),
        'Neural Network': NeuralNetwork(learning_rate=0.1, random_state=42)
    }
    
    # Train and compare models
    results = train_and_compare_models(X, y, models, config)
    
    # Create and print report
    report = create_training_report(results)
    print(report)
    
    # Plot training history for neural network
    if 'Neural Network' in results:
        trainer = Trainer(config)
        trainer.training_history = results['Neural Network']['training_history']
        trainer.plot_training_history()


if __name__ == "__main__":
    main()
