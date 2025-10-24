#!/usr/bin/env python3
"""
Neural Network Classification Learning Project - Main Demonstration

This script demonstrates the complete neural network classification project,
including data generation, model training, comparison, and visualization.

Author: AI Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

# Add the neural_network package to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neural_network.data_generator import DataGenerator
from neural_network.logistic_regression import LogisticRegression
from neural_network.neural_network import NeuralNetwork
from neural_network.trainer import TrainingConfig, train_and_compare_models, create_training_report
from neural_network.visualization import (
    plot_data_distribution, plot_model_comparison, create_summary_plot
)
from neural_network.utils import train_test_split, set_random_seed


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="""
Neural Network Classification Learning Project

This program demonstrates the power of neural networks for non-linear classification
by comparing them against logistic regression on synthetic datasets. It generates
non-linearly separable data and trains both models to show how neural networks
can learn complex decision boundaries that linear models cannot.

Key Features:
- Generates synthetic 2D datasets (XOR, circles, moons)
- Implements logistic regression from scratch
- Implements neural network with single hidden layer
- Compares performance and visualizes decision boundaries
- Provides comprehensive training metrics and timing
- Creates educational visualizations

The program is designed for learning neural network fundamentals including:
- Forward and backward propagation
- Activation functions (tanh, sigmoid)
- Cross-entropy loss computation
- Gradient descent optimization
- Non-linear classification capabilities
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic demo with default settings
  python3 main.py

  # Quick demo with fewer epochs
  python3 main.py --epochs 100 --verbose

  # Try different data types
  python3 main.py --data-type circles --epochs 500

  # Save plots to a directory
  python3 main.py --save-plots ./results --epochs 200

  # Disable plots for headless environments
  python3 main.py --no-plots --epochs 1000

  # Customize neural network architecture
  python3 main.py --hidden-size 16 --learning-rate 0.05 --epochs 500

Expected Results:
- Logistic Regression: ~50-60% accuracy (limited by linear decision boundary)
- Neural Network: ~95-99% accuracy (can learn non-linear patterns)
- Neural network should show significant improvement over logistic regression
        """
    )
    
    parser.add_argument(
        '--data-type', 
        choices=['xor', 'circles', 'moons'], 
        default='xor',
        help='Type of synthetic data to generate. XOR creates non-linearly separable data that benefits from neural networks. Circles and moons are other classic non-linear patterns.'
    )
    
    parser.add_argument(
        '--n-samples', 
        type=int, 
        default=400,
        help='Number of training samples to generate. More samples provide better learning but take longer to train.'
    )
    
    parser.add_argument(
        '--noise', 
        type=float, 
        default=0.1,
        help='Amount of noise to add to the data (0.0 to 1.0). Higher noise makes classification more challenging but more realistic.'
    )
    
    parser.add_argument(
        '--learning-rate', 
        type=float, 
        default=0.1,
        help='Learning rate for gradient descent optimization. Higher values train faster but may overshoot optimal weights.'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=1000,
        help='Number of training epochs. More epochs allow better learning but increase training time.'
    )
    
    parser.add_argument(
        '--hidden-size', 
        type=int, 
        default=8,
        help='Number of hidden neurons in the neural network. More neurons increase model capacity but also training time.'
    )
    
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.2,
        help='Fraction of data to use for testing (0.0 to 1.0). Remaining data is used for training.'
    )
    
    parser.add_argument(
        '--random-state', 
        type=int, 
        default=42,
        help='Random seed for reproducible results. Use the same seed to get identical results across runs.'
    )
    
    parser.add_argument(
        '--no-plots', 
        action='store_true',
        help='Disable plotting and visualization. Useful for headless environments or when you only want the performance metrics.'
    )
    
    parser.add_argument(
        '--save-plots', 
        type=str,
        help='Directory path to save all generated plots as PNG files. If not specified, plots are displayed on screen.'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable detailed progress output during training, showing loss and accuracy at regular intervals.'
    )
    
    return parser.parse_args()


def generate_data(data_type: str, n_samples: int, noise: float, random_state: int):
    """Generate synthetic data based on specified type."""
    generator = DataGenerator(random_state=random_state)
    
    if data_type == 'xor':
        X, y = generator.generate_xor_data(n_samples=n_samples, noise=noise)
    elif data_type == 'circles':
        X, y = generator.generate_circle_data(n_samples=n_samples, noise=noise)
    elif data_type == 'moons':
        X, y = generator.generate_moon_data(n_samples=n_samples, noise=noise)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return X, y


def create_models(learning_rate: float, hidden_size: int, random_state: int):
    """Create logistic regression and neural network models."""
    models = {
        'Logistic Regression': LogisticRegression(
            learning_rate=learning_rate,
            max_iterations=1000,
            random_state=random_state
        ),
        'Neural Network': NeuralNetwork(
            input_size=2,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            random_state=random_state
        )
    }
    
    return models


def run_experiment(args):
    """Run the complete neural network classification experiment."""
    print("=" * 60)
    print("NEURAL NETWORK CLASSIFICATION LEARNING PROJECT")
    print("=" * 60)
    print(f"Data Type: {args.data_type}")
    print(f"Number of Samples: {args.n_samples}")
    print(f"Noise Level: {args.noise}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Hidden Neurons: {args.hidden_size}")
    print(f"Test Size: {args.test_size}")
    print(f"Random State: {args.random_state}")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_random_seed(args.random_state)
    
    # Generate data
    print("\n1. Generating synthetic data...")
    print("   Creating data generator...")
    X, y = generate_data(args.data_type, args.n_samples, args.noise, args.random_state)
    print(f"   ✓ Generated {X.shape[0]} samples with {X.shape[1]} features")
    print(f"   ✓ Class distribution: {np.bincount(y)}")
    
    # Split data
    print("\n2. Splitting data into train/test sets...")
    print("   Performing train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"   ✓ Training set: {X_train.shape[0]} samples")
    print(f"   ✓ Test set: {X_test.shape[0]} samples")
    
    # Create models
    print("\n3. Creating models...")
    print("   Initializing Logistic Regression...")
    print("   Initializing Neural Network...")
    models = create_models(args.learning_rate, args.hidden_size, args.random_state)
    print(f"   ✓ Created {len(models)} models: {list(models.keys())}")
    
    # Create training configuration
    print("\n4. Setting up training configuration...")
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        early_stopping=True,
        patience=50,
        validation_split=0.0,  # No validation split since we already split the data
        verbose=args.verbose
    )
    print(f"   ✓ Learning rate: {config.learning_rate}")
    print(f"   ✓ Max epochs: {config.max_epochs}")
    print(f"   ✓ Early stopping: {config.early_stopping}")
    
    # Train models
    print("\n5. Training models...")
    print("   Starting model training process...")
    results = train_and_compare_models(X_train, y_train, models, config)
    
    # Evaluate models on test set
    print("\n6. Evaluating models on test set...")
    print("   Computing test metrics for each model...")
    for model_name, result in results.items():
        print(f"   Evaluating {model_name}...")
        model = result['model']
        test_metrics = model.get_metrics(X_test, y_test)
        result['test_metrics'] = test_metrics
        print(f"   ✓ {model_name} - Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Create training report
    print("\n7. Generating training report...")
    print("   Compiling performance statistics...")
    report = create_training_report(results)
    print(report)
    
    # Create visualizations
    if not args.no_plots:
        print("\n8. Creating visualizations...")
        
        # Prepare save paths if requested
        save_paths = {}
        if args.save_plots:
            print(f"   Preparing to save plots to: {args.save_plots}")
            os.makedirs(args.save_plots, exist_ok=True)
            save_paths = {
                'data_dist': os.path.join(args.save_plots, 'data_distribution.png'),
                'model_comparison': os.path.join(args.save_plots, 'model_comparison.png'),
                'summary': os.path.join(args.save_plots, 'summary_plot.png')
            }
        
        print("   Plotting data distribution...")
        plot_data_distribution(X, y, f"{args.data_type.upper()} Dataset Distribution", 
                              save_path=save_paths.get('data_dist'))
        
        print("   Plotting decision boundaries...")
        plot_model_comparison(
            {name: result['model'] for name, result in results.items()},
            X_test, y_test,
            titles={name: f"{name} Decision Boundary" for name in results.keys()},
            save_path=save_paths.get('model_comparison')
        )
        
        print("   Creating comprehensive summary plot...")
        metrics_dict = {name: result['test_metrics'] for name, result in results.items()}
        create_summary_plot(
            {name: result['model'] for name, result in results.items()},
            X_test, y_test,
            metrics_dict,
            save_path=save_paths.get('summary')
        )
        
        # Confirm plots saved
        if args.save_plots:
            print(f"   ✓ Plots saved to: {args.save_plots}")
            for plot_name, plot_path in save_paths.items():
                if os.path.exists(plot_path):
                    print(f"     - {plot_name}: {plot_path}")
    else:
        print("\n8. Skipping visualizations (--no-plots flag set)")
    
    # Performance comparison
    print("\n9. Performance Comparison Summary:")
    print("-" * 40)
    
    lr_accuracy = results['Logistic Regression']['test_metrics']['accuracy']
    nn_accuracy = results['Neural Network']['test_metrics']['accuracy']
    improvement = nn_accuracy - lr_accuracy
    improvement_pct = (improvement / lr_accuracy) * 100
    
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    print(f"Neural Network Accuracy:     {nn_accuracy:.4f}")
    print(f"Improvement:                 {improvement:.4f} ({improvement_pct:.1f}%)")
    
    if improvement > 0.1:  # 10% improvement
        print("✅ Neural network shows significant improvement over logistic regression!")
    elif improvement > 0.05:  # 5% improvement
        print("✅ Neural network shows moderate improvement over logistic regression.")
    else:
        print("⚠️  Neural network shows limited improvement. Consider:")
        print("   - Increasing hidden layer size")
        print("   - Adjusting learning rate")
        print("   - Training for more epochs")
        print("   - Using a more complex dataset")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return results


def main():
    """Main function."""
    try:
        args = parse_arguments()
        results = run_experiment(args)
        
        # Return success code
        return 0
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
