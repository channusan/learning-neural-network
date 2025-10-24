
# Learning Neural Networks (AI-Assisted Prototype)

## 🧠 Overview
This project is a **concept prototype** demonstrating AI-assisted workflows for neural network learning and development.  
It includes AI-generated PRDs, task breakdowns, and prototype neural network code.  
**Educational/portfolio use only.**

# Neural Network Classification Learning Project

A comprehensive educational implementation of a 2-class classification neural network with a single hidden layer, designed to demonstrate neural network fundamentals and compare performance with logistic regression.

---

## ⚙️ Tech Stack
- **Languages:** Python, NumPy, PyTorch/TensorFlow
- **AI Tools:** Cursor AI / VibeCode
- **Integrations:** `ai-dev-tasks` for automated PRD and task generation

---

## 🧩 Features
- Generate neural network structures programmatically
- AI-assisted PRD and task list generation
- Prototype neural network training examples
- Explore AI-assisted development workflows


---

## 🙏 Acknowledgments
- Portions of this project are derived from [snarktank/ai-dev-tasks](https://github.com/snarktank/ai-dev-tasks) (Apache 2.0)
- AI-assisted code generated with Cursor AI

---

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


## 📝 License

This project is for educational purposes. Feel free to use and modify as needed.