"""Tests for password strength evaluation system."""

import pytest
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.utils import (
    calculate_entropy,
    detect_keyboard_patterns,
    detect_sequential_patterns,
    detect_repeated_patterns,
    estimate_time_to_crack,
    validate_password_input,
    hash_password,
    anonymize_output,
)
from src.features import PasswordFeatureExtractor
from src.models import EntropyBasedModel


class TestUtils:
    """Test utility functions."""
    
    def test_calculate_entropy(self):
        """Test entropy calculation."""
        # Test empty password
        assert calculate_entropy("") == 0.0
        
        # Test single character
        assert calculate_entropy("a") == 0.0
        
        # Test repeated characters
        assert calculate_entropy("aaaa") == 0.0
        
        # Test diverse characters
        entropy = calculate_entropy("abcd")
        assert entropy > 0
        assert entropy <= 2.0  # Max entropy for 4 unique chars
    
    def test_detect_keyboard_patterns(self):
        """Test keyboard pattern detection."""
        # Test qwerty pattern
        patterns = detect_keyboard_patterns("qwerty123")
        assert "keyboard_sequence_qwerty" in patterns
        
        # Test asdf pattern
        patterns = detect_keyboard_patterns("asdf")
        assert "keyboard_sequence_asdf" in patterns
        
        # Test no patterns
        patterns = detect_keyboard_patterns("random")
        assert len(patterns) == 0
    
    def test_detect_sequential_patterns(self):
        """Test sequential pattern detection."""
        # Test sequential numbers
        patterns = detect_sequential_patterns("123456")
        assert "sequential_numbers" in patterns
        
        # Test sequential letters
        patterns = detect_sequential_patterns("abcdef")
        assert "sequential_letters" in patterns
        
        # Test no patterns
        patterns = detect_sequential_patterns("random")
        assert len(patterns) == 0
    
    def test_detect_repeated_patterns(self):
        """Test repeated pattern detection."""
        # Test repeated characters
        patterns = detect_repeated_patterns("aaa")
        assert "repeated_characters" in patterns
        
        # Test repeated substrings
        patterns = detect_repeated_patterns("abab")
        assert "repeated_substring_2" in patterns
        
        # Test no patterns
        patterns = detect_repeated_patterns("random")
        assert len(patterns) == 0
    
    def test_estimate_time_to_crack(self):
        """Test time to crack estimation."""
        # Test empty password
        assert estimate_time_to_crack("") == 0.0
        
        # Test short password
        time = estimate_time_to_crack("a")
        assert time > 0
        
        # Test longer password
        time_long = estimate_time_to_crack("abcdefgh")
        assert time_long > estimate_time_to_crack("a")
    
    def test_validate_password_input(self):
        """Test password input validation."""
        # Test valid password
        is_valid, error = validate_password_input("validpassword")
        assert is_valid
        assert error is None
        
        # Test empty password
        is_valid, error = validate_password_input("")
        assert not is_valid
        assert "empty" in error.lower()
        
        # Test too long password
        long_password = "a" * 200
        is_valid, error = validate_password_input(long_password)
        assert not is_valid
        assert "too long" in error.lower()
        
        # Test suspicious pattern
        is_valid, error = validate_password_input("<script>alert('xss')</script>")
        assert not is_valid
        assert "suspicious" in error.lower()
    
    def test_hash_password(self):
        """Test password hashing."""
        password = "testpassword"
        hashed = hash_password(password)
        
        # Should be different from original
        assert hashed != password
        
        # Should contain salt and hash
        assert ":" in hashed
        
        # Should be deterministic with same salt
        salt = "testsalt"
        hashed1 = hash_password(password, salt)
        hashed2 = hash_password(password, salt)
        assert hashed1 == hashed2
    
    def test_anonymize_output(self):
        """Test output anonymization."""
        # Test short text
        text = "short"
        anonymized = anonymize_output(text, max_length=10)
        assert anonymized == text
        
        # Test long text
        text = "a" * 100
        anonymized = anonymize_output(text, max_length=10)
        assert len(anonymized) <= 10
        assert anonymized.endswith("...")


class TestFeatureExtractor:
    """Test feature extraction."""
    
    def test_extract_basic_features(self):
        """Test basic feature extraction."""
        extractor = PasswordFeatureExtractor()
        
        features = extractor.extract_basic_features("Test123!")
        
        # Check required features
        assert "length" in features
        assert "lowercase_count" in features
        assert "uppercase_count" in features
        assert "digit_count" in features
        assert "symbol_count" in features
        
        # Check values
        assert features["length"] == 8
        assert features["lowercase_count"] == 3
        assert features["uppercase_count"] == 1
        assert features["digit_count"] == 3
        assert features["symbol_count"] == 1
    
    def test_extract_entropy_features(self):
        """Test entropy feature extraction."""
        extractor = PasswordFeatureExtractor()
        
        features = extractor.extract_entropy_features("Test123!")
        
        # Check required features
        assert "shannon_entropy" in features
        assert "char_set_size" in features
        assert "max_entropy" in features
        assert "entropy_ratio" in features
        assert "time_to_crack" in features
        
        # Check values
        assert features["shannon_entropy"] > 0
        assert features["char_set_size"] > 0
        assert features["max_entropy"] > 0
        assert 0 <= features["entropy_ratio"] <= 1
        assert features["time_to_crack"] > 0
    
    def test_extract_pattern_features(self):
        """Test pattern feature extraction."""
        extractor = PasswordFeatureExtractor()
        
        features = extractor.extract_pattern_features("qwerty123")
        
        # Check required features
        assert "keyboard_pattern_count" in features
        assert "sequential_pattern_count" in features
        assert "repeated_pattern_count" in features
        assert "total_pattern_count" in features
        assert "pattern_density" in features
        
        # Check values
        assert features["keyboard_pattern_count"] >= 0
        assert features["sequential_pattern_count"] >= 0
        assert features["repeated_pattern_count"] >= 0
        assert features["total_pattern_count"] >= 0
        assert 0 <= features["pattern_density"] <= 1
    
    def test_extract_all_features(self):
        """Test extraction of all features."""
        extractor = PasswordFeatureExtractor()
        
        features = extractor.extract_all_features("Test123!")
        
        # Should have features from all categories
        assert len(features) > 20  # Should have many features
        
        # Check some key features exist
        assert "length" in features
        assert "shannon_entropy" in features
        assert "keyboard_pattern_count" in features
    
    def test_extract_features_batch(self):
        """Test batch feature extraction."""
        extractor = PasswordFeatureExtractor()
        
        passwords = ["password", "Test123!", "qwerty"]
        feature_df = extractor.extract_features_batch(passwords)
        
        # Check DataFrame structure
        assert len(feature_df) == 3
        assert len(feature_df.columns) > 20
        
        # Check feature values
        assert feature_df.iloc[0]["length"] == 8  # "password"
        assert feature_df.iloc[1]["length"] == 8  # "Test123!"
        assert feature_df.iloc[2]["length"] == 6  # "qwerty"


class TestEntropyBasedModel:
    """Test entropy-based model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        config = OmegaConf.create({
            "entropy_thresholds": {"weak": 20, "moderate": 40, "strong": 60},
            "complexity_weights": {
                "length": 0.3,
                "character_diversity": 0.2,
                "entropy": 0.3,
                "pattern_detection": 0.2
            },
            "classification": {
                "weak_threshold": 0.33,
                "moderate_threshold": 0.66
            }
        })
        
        model = EntropyBasedModel(config)
        assert model.config == config
        assert not model.is_fitted
    
    def test_password_score_calculation(self):
        """Test password score calculation."""
        config = OmegaConf.create({
            "entropy_thresholds": {"weak": 20, "moderate": 40, "strong": 60},
            "complexity_weights": {
                "length": 0.3,
                "character_diversity": 0.2,
                "entropy": 0.3,
                "pattern_detection": 0.2
            },
            "classification": {
                "weak_threshold": 0.33,
                "moderate_threshold": 0.66
            }
        })
        
        model = EntropyBasedModel(config)
        
        # Test score calculation
        score = model._calculate_password_score("Test123!")
        assert 0 <= score <= 1
        
        # Test empty password
        score_empty = model._calculate_password_score("")
        assert score_empty == 0.0
    
    def test_password_classification(self):
        """Test password classification."""
        config = OmegaConf.create({
            "entropy_thresholds": {"weak": 20, "moderate": 40, "strong": 60},
            "complexity_weights": {
                "length": 0.3,
                "character_diversity": 0.2,
                "entropy": 0.3,
                "pattern_detection": 0.2
            },
            "classification": {
                "weak_threshold": 0.33,
                "moderate_threshold": 0.66
            }
        })
        
        model = EntropyBasedModel(config)
        
        # Test classification
        classification = model._classify_password("Test123!")
        assert classification in ["weak", "moderate", "strong"]
        
        # Test empty password
        classification_empty = model._classify_password("")
        assert classification_empty == "weak"


class TestDataGeneration:
    """Test synthetic data generation."""
    
    def test_password_generation(self):
        """Test password generation."""
        from src.data import SyntheticPasswordGenerator
        
        config = OmegaConf.create({
            "password_lengths": {"min": 4, "max": 16},
            "character_sets": {
                "lowercase": "abcdefghijklmnopqrstuvwxyz",
                "uppercase": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                "digits": "0123456789",
                "symbols": "!@#$%^&*()_+-=[]{}|;:,.<>?"
            },
            "pattern_probabilities": {
                "dictionary_word": 0.3,
                "keyboard_pattern": 0.2,
                "sequential": 0.15,
                "repeated_chars": 0.1,
                "personal_info": 0.1,
                "random": 0.15
            },
            "weak_patterns": ["password", "123456", "qwerty"],
            "breach_simulation": {
                "enabled": True,
                "breach_probability": 0.1,
                "common_breached_passwords": 1000
            }
        })
        
        generator = SyntheticPasswordGenerator(config)
        
        # Test password generation
        password = generator.generate_password("random")
        assert isinstance(password, str)
        assert len(password) >= 4
        
        # Test dataset generation
        dataset = generator.generate_dataset(100)
        assert len(dataset) == 100
        assert "password" in dataset.columns
        assert "strength_label" in dataset.columns
        assert "pattern_type" in dataset.columns
        
        # Test strength labels
        strength_labels = dataset["strength_label"].unique()
        assert all(label in ["weak", "moderate", "strong"] for label in strength_labels)


if __name__ == "__main__":
    pytest.main([__file__])
