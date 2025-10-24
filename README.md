# Neural Network Classification Learning Project

A comprehensive educational implementation of a 2-class classification neural network with a single hidden layer, designed to demonstrate neural network fundamentals and compare performance with logistic regression.

## 🎯 Project Overview

This project implements a complete neural network from scratch to teach fundamental concepts including:
- Forward and backward propagation
- Activation functions (tanh, sigmoid)
- Cross-entropy loss computation
- Gradient descent optimization
- Non-linear classification capabilities

## 🚀 Features

- **Synthetic Data Generation**: Creates non-linearly separable 2D datasets (XOR pattern)
- **Logistic Regression Baseline**: Linear classifier for performance comparison
- **Neural Network Implementation**: 2-8-1 architecture with tanh activation
- **Comprehensive Visualization**: Decision boundaries, training curves, and performance metrics
- **Extensive Testing**: Unit tests for all components with gradient checking
- **Educational Focus**: Well-documented code with clear explanations

## 📋 Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Scikit-learn
- Pytest (for testing)

## 🛠️ Installation

1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Quick Start

Run the main demonstration:
```bash
python main.py
```

This will:
1. Generate synthetic training data (400 samples)
2. Train both logistic regression and neural network models
3. Display performance comparison
4. Show decision boundary visualizations
5. Plot training progress

## 📁 Project Structure

```
neural_network/
├── __init__.py              # Package initialization
├── data_generator.py        # Synthetic data generation
├── logistic_regression.py   # Logistic regression implementation
├── neural_network.py        # Neural network implementation
├── activations.py           # Activation functions
├── loss_functions.py        # Loss functions
├── trainer.py              # Training logic
├── visualization.py        # Plotting utilities
├── utils.py                # Utility functions
└── *_test.py               # Unit tests for each module

main.py                     # Main demonstration script
requirements.txt            # Project dependencies
README.md                   # This file
```

## 🧪 Testing

Run all tests:
```bash
python -m pytest
```

Run specific test file:
```bash
python -m pytest neural_network/neural_network_test.py
```

Run with coverage:
```bash
python -m pytest --cov=neural_network
```

## 📊 Expected Results

The neural network should significantly outperform logistic regression on the non-linearly separable data:
- **Logistic Regression**: ~50-60% accuracy (limited by linear decision boundary)
- **Neural Network**: ~95-99% accuracy (can learn non-linear patterns)

## 🎓 Learning Objectives

After working with this project, you should understand:
1. How neural networks process information through forward propagation
2. How gradients are computed through backpropagation
3. The role of activation functions in non-linear learning
4. How loss functions guide the learning process
5. Why neural networks excel at non-linear classification tasks

## 🔧 Customization

The project is designed to be easily customizable:
- Modify data patterns in `data_generator.py`
- Adjust network architecture in `neural_network.py`
- Change training parameters in `trainer.py`
- Add new visualizations in `visualization.py`

## 📝 License

This project is for educational purposes. Feel free to use and modify as needed.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to improve the educational value of this project.



