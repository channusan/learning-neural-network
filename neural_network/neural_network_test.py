"""
Unit tests for neural_network module.

Author: AI Assistant
"""

import pytest
import numpy as np
from neural_network.neural_network import NeuralNetwork
from neural_network.data_generator import DataGenerator


class TestNeuralNetwork:
    """Test cases for NeuralNetwork class."""
    
    def test_init(self):
        """Test NeuralNetwork initialization."""
        nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
        
        assert nn.input_size == 2
        assert nn.hidden_size == 4
        assert nn.output_size == 1
        assert nn.learning_rate == 0.1
        assert not nn.is_fitted
        
        # Check weight shapes
        assert nn.W1.shape == (2, 4)
        assert nn.b1.shape == (1, 4)
        assert nn.W2.shape == (4, 1)
        assert nn.b2.shape == (1, 1)
    
    def test_forward_propagation(self):
        """Test forward propagation."""
        nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, random_state=42)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        output = nn._forward_propagation(X)
        
        # Check output shape
        assert output.shape == (2, 1)
        
        # Check output is between 0 and 1 (sigmoid output)
        assert np.all(output >= 0)
        assert np.all(output <= 1)
        
        # Check cache is populated
        assert 'Z1' in nn._cache
        assert 'A1' in nn._cache
        assert 'Z2' in nn._cache
        assert 'A2' in nn._cache
    
    def test_backward_propagation(self):
        """Test backward propagation."""
        nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, random_state=42)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1, 0])
        
        # Forward pass first
        output = nn._forward_propagation(X)
        
        # Backward pass
        gradients = nn._backward_propagation(X, y, output)
        
        # Check gradient shapes
        assert gradients['dW1'].shape == (2, 4)
        assert gradients['db1'].shape == (1, 4)
        assert gradients['dW2'].shape == (4, 1)
        assert gradients['db2'].shape == (1, 1)
    
    def test_fit(self):
        """Test model training."""
        # Generate simple data
        generator = DataGenerator(random_state=42)
        X, y = generator.generate_xor_data(n_samples=50, noise=0.1)
        
        nn = NeuralNetwork(input_size=2, hidden_size=4, learning_rate=0.1, random_state=42)
        
        # Train model
        nn.fit(X, y, epochs=100, verbose=False)
        
        # Check model is fitted
        assert nn.is_fitted
        assert len(nn.loss_history) == 100
        assert len(nn.accuracy_history) == 100
        
        # Check loss decreases over time
        assert nn.loss_history[-1] < nn.loss_history[0]
    
    def test_predict_proba(self):
        """Test probability prediction."""
        # Generate data and train model
        generator = DataGenerator(random_state=42)
        X, y = generator.generate_xor_data(n_samples=50, noise=0.1)
        
        nn = NeuralNetwork(input_size=2, hidden_size=4, learning_rate=0.1, random_state=42)
        nn.fit(X, y, epochs=100, verbose=False)
        
        # Test prediction
        X_test = np.array([[1.0, 1.0], [0.0, 0.0]])
        probabilities = nn.predict_proba(X_test)
        
        # Check output shape and range
        assert probabilities.shape == (2,)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
    
    def test_predict(self):
        """Test binary prediction."""
        # Generate data and train model
        generator = DataGenerator(random_state=42)
        X, y = generator.generate_xor_data(n_samples=50, noise=0.1)
        
        nn = NeuralNetwork(input_size=2, hidden_size=4, learning_rate=0.1, random_state=42)
        nn.fit(X, y, epochs=100, verbose=False)
        
        # Test prediction
        X_test = np.array([[1.0, 1.0], [0.0, 0.0]])
        predictions = nn.predict(X_test)
        
        # Check output shape and values
        assert predictions.shape == (2,)
        assert set(predictions) <= {0, 1}
    
    def test_score(self):
        """Test accuracy scoring."""
        # Generate data and train model
        generator = DataGenerator(random_state=42)
        X, y = generator.generate_xor_data(n_samples=50, noise=0.1)
        
        nn = NeuralNetwork(input_size=2, hidden_size=4, learning_rate=0.1, random_state=42)
        nn.fit(X, y, epochs=100, verbose=False)
        
        # Test scoring
        score = nn.score(X, y)
        
        # Check score is between 0 and 1
        assert 0 <= score <= 1
    
    def test_get_metrics(self):
        """Test metrics calculation."""
        # Generate data and train model
        generator = DataGenerator(random_state=42)
        X, y = generator.generate_xor_data(n_samples=50, noise=0.1)
        
        nn = NeuralNetwork(input_size=2, hidden_size=4, learning_rate=0.1, random_state=42)
        nn.fit(X, y, epochs=100, verbose=False)
        
        # Test metrics
        metrics = nn.get_metrics(X, y)
        
        # Check all expected metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
    
    def test_gradient_check(self):
        """Test gradient checking."""
        # Generate small dataset for gradient checking
        generator = DataGenerator(random_state=42)
        X, y = generator.generate_xor_data(n_samples=10, noise=0.1)
        
        nn = NeuralNetwork(input_size=2, hidden_size=4, learning_rate=0.1, random_state=42)
        nn.fit(X, y, epochs=10, verbose=False)
        
        # Test gradient checking
        gradient_errors = nn.gradient_check(X, y)
        
        # Check all parameters have gradient errors
        expected_params = ['W1_relative_error', 'b1_relative_error', 'W2_relative_error', 'b2_relative_error']
        for param in expected_params:
            assert param in gradient_errors
            assert gradient_errors[param] < 1e-3  # Should be very small
    
    def test_model_info(self):
        """Test model information extraction."""
        nn = NeuralNetwork(input_size=2, hidden_size=4, learning_rate=0.1, random_state=42)
        
        # Test before fitting
        info = nn.get_model_info()
        assert not info['is_fitted']
        
        # Generate data and train
        generator = DataGenerator(random_state=42)
        X, y = generator.generate_xor_data(n_samples=50, noise=0.1)
        nn.fit(X, y, epochs=100, verbose=False)
        
        # Test after fitting
        info = nn.get_model_info()
        assert info['is_fitted']
        assert info['architecture'] == '2-4-1'
        assert info['total_parameters'] == 17  # 2*4 + 4 + 4*1 + 1 = 17
        assert info['n_epochs'] == 100
    
    def test_unfitted_model_errors(self):
        """Test errors when using unfitted model."""
        nn = NeuralNetwork(input_size=2, hidden_size=4, learning_rate=0.1, random_state=42)
        X = np.array([[1.0, 2.0]])
        
        with pytest.raises(ValueError):
            nn.predict_proba(X)
        
        with pytest.raises(ValueError):
            nn.predict(X)
        
        with pytest.raises(ValueError):
            nn.score(X, np.array([1]))
        
        with pytest.raises(ValueError):
            nn.get_metrics(X, np.array([1]))


if __name__ == "__main__":
    pytest.main([__file__])



