"""
Unit tests for data_generator module.

Author: AI Assistant
"""

import pytest
import numpy as np
from neural_network.data_generator import DataGenerator


class TestDataGenerator:
    """Test cases for DataGenerator class."""
    
    def test_init(self):
        """Test DataGenerator initialization."""
        generator = DataGenerator(random_state=42)
        assert generator.random_state == 42
    
    def test_generate_xor_data(self):
        """Test XOR data generation."""
        generator = DataGenerator(random_state=42)
        X, y = generator.generate_xor_data(n_samples=100, noise=0.1)
        
        # Check shapes
        assert X.shape == (100, 2)
        assert y.shape == (100,)
        
        # Check data types
        assert X.dtype == np.float64
        assert y.dtype == np.int64
        
        # Check labels are binary
        assert set(y) == {0, 1}
        
        # Check roughly equal class distribution
        class_counts = np.bincount(y)
        assert len(class_counts) == 2
        assert abs(class_counts[0] - class_counts[1]) <= 2  # Allow small difference
    
    def test_generate_circle_data(self):
        """Test circle data generation."""
        generator = DataGenerator(random_state=42)
        X, y = generator.generate_circle_data(n_samples=100, noise=0.1)
        
        # Check shapes
        assert X.shape == (100, 2)
        assert y.shape == (100,)
        
        # Check labels are binary
        assert set(y) == {0, 1}
    
    def test_generate_moon_data(self):
        """Test moon data generation."""
        generator = DataGenerator(random_state=42)
        X, y = generator.generate_moon_data(n_samples=100, noise=0.1)
        
        # Check shapes
        assert X.shape == (100, 2)
        assert y.shape == (100,)
        
        # Check labels are binary
        assert set(y) == {0, 1}
    
    def test_invalid_n_samples(self):
        """Test error handling for invalid n_samples."""
        generator = DataGenerator(random_state=42)
        
        with pytest.raises(ValueError):
            generator.generate_xor_data(n_samples=0)
        
        with pytest.raises(ValueError):
            generator.generate_xor_data(n_samples=1)
    
    def test_invalid_noise(self):
        """Test error handling for invalid noise values."""
        generator = DataGenerator(random_state=42)
        
        with pytest.raises(ValueError):
            generator.generate_xor_data(n_samples=100, noise=-0.1)
        
        with pytest.raises(ValueError):
            generator.generate_xor_data(n_samples=100, noise=1.1)
    
    def test_get_data_info(self):
        """Test data info extraction."""
        generator = DataGenerator(random_state=42)
        X, y = generator.generate_xor_data(n_samples=100, noise=0.1)
        
        info = generator.get_data_info(X, y)
        
        assert info['n_samples'] == 100
        assert info['n_features'] == 2
        assert info['n_classes'] == 2
        assert 'class_distribution' in info
        assert 'feature_ranges' in info


if __name__ == "__main__":
    pytest.main([__file__])



