"""Feature engineering for password strength evaluation."""

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from ..utils import (
    calculate_entropy,
    detect_keyboard_patterns,
    detect_repeated_patterns,
    detect_sequential_patterns,
    estimate_time_to_crack,
)


class PasswordFeatureExtractor:
    """Extract comprehensive features from passwords for strength evaluation."""
    
    def __init__(self, config: Dict = None):
        """Initialize the feature extractor.
        
        Args:
            config: Configuration dictionary for feature extraction
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = None
        self.is_fitted = False
        
    def extract_basic_features(self, password: str) -> Dict[str, float]:
        """Extract basic statistical features from password.
        
        Args:
            password: Password to analyze
            
        Returns:
            Dictionary of basic features
        """
        features = {}
        
        # Length features
        features["length"] = len(password)
        features["length_log"] = np.log(len(password) + 1)
        
        # Character type counts
        features["lowercase_count"] = sum(1 for c in password if c.islower())
        features["uppercase_count"] = sum(1 for c in password if c.isupper())
        features["digit_count"] = sum(1 for c in password if c.isdigit())
        features["symbol_count"] = sum(1 for c in password if not c.isalnum())
        
        # Character type ratios
        if len(password) > 0:
            features["lowercase_ratio"] = features["lowercase_count"] / len(password)
            features["uppercase_ratio"] = features["uppercase_count"] / len(password)
            features["digit_ratio"] = features["digit_count"] / len(password)
            features["symbol_ratio"] = features["symbol_count"] / len(password)
        else:
            features["lowercase_ratio"] = 0
            features["uppercase_ratio"] = 0
            features["digit_ratio"] = 0
            features["symbol_ratio"] = 0
        
        # Character diversity
        unique_chars = len(set(password))
        features["unique_char_count"] = unique_chars
        features["char_diversity"] = unique_chars / len(password) if len(password) > 0 else 0
        
        return features
    
    def extract_entropy_features(self, password: str) -> Dict[str, float]:
        """Extract entropy-based features from password.
        
        Args:
            password: Password to analyze
            
        Returns:
            Dictionary of entropy features
        """
        features = {}
        
        # Shannon entropy
        features["shannon_entropy"] = calculate_entropy(password)
        
        # Character set entropy
        char_set_size = 0
        if any(c.islower() for c in password):
            char_set_size += 26
        if any(c.isupper() for c in password):
            char_set_size += 26
        if any(c.isdigit() for c in password):
            char_set_size += 10
        if any(not c.isalnum() for c in password):
            char_set_size += 32
        
        features["char_set_size"] = char_set_size
        features["max_entropy"] = len(password) * np.log2(char_set_size) if char_set_size > 0 else 0
        features["entropy_ratio"] = features["shannon_entropy"] / features["max_entropy"] if features["max_entropy"] > 0 else 0
        
        # Time to crack estimation
        features["time_to_crack"] = estimate_time_to_crack(password)
        features["time_to_crack_log"] = np.log(features["time_to_crack"] + 1)
        
        return features
    
    def extract_pattern_features(self, password: str) -> Dict[str, float]:
        """Extract pattern-based features from password.
        
        Args:
            password: Password to analyze
            
        Returns:
            Dictionary of pattern features
        """
        features = {}
        
        # Detect various patterns
        keyboard_patterns = detect_keyboard_patterns(password)
        sequential_patterns = detect_sequential_patterns(password)
        repeated_patterns = detect_repeated_patterns(password)
        
        # Pattern counts
        features["keyboard_pattern_count"] = len(keyboard_patterns)
        features["sequential_pattern_count"] = len(sequential_patterns)
        features["repeated_pattern_count"] = len(repeated_patterns)
        features["total_pattern_count"] = len(keyboard_patterns) + len(sequential_patterns) + len(repeated_patterns)
        
        # Pattern ratios
        if len(password) > 0:
            features["pattern_density"] = features["total_pattern_count"] / len(password)
        else:
            features["pattern_density"] = 0
        
        # Specific pattern indicators
        features["has_keyboard_pattern"] = 1 if keyboard_patterns else 0
        features["has_sequential_pattern"] = 1 if sequential_patterns else 0
        features["has_repeated_pattern"] = 1 if repeated_patterns else 0
        
        return features
    
    def extract_linguistic_features(self, password: str) -> Dict[str, float]:
        """Extract linguistic features from password.
        
        Args:
            password: Password to analyze
            
        Returns:
            Dictionary of linguistic features
        """
        features = {}
        
        # Common dictionary words
        common_words = [
            "password", "admin", "welcome", "hello", "world",
            "computer", "internet", "security", "system", "user",
            "login", "access", "account", "profile", "settings",
            "qwerty", "asdf", "zxcv", "123456", "abcdef"
        ]
        
        password_lower = password.lower()
        word_matches = sum(1 for word in common_words if word in password_lower)
        
        features["dictionary_word_count"] = word_matches
        features["has_dictionary_word"] = 1 if word_matches > 0 else 0
        
        # Leet speak detection
        leet_mappings = {
            "a": "@", "e": "3", "i": "!", "o": "0", "s": "$", "t": "7"
        }
        
        leet_count = sum(1 for char, replacement in leet_mappings.items() 
                        if replacement in password_lower)
        features["leet_speak_count"] = leet_count
        features["has_leet_speak"] = 1 if leet_count > 0 else 0
        
        # Palindrome detection
        features["is_palindrome"] = 1 if password == password[::-1] else 0
        
        return features
    
    def extract_all_features(self, password: str) -> Dict[str, float]:
        """Extract all available features from password.
        
        Args:
            password: Password to analyze
            
        Returns:
            Dictionary of all features
        """
        features = {}
        
        # Combine all feature types
        features.update(self.extract_basic_features(password))
        features.update(self.extract_entropy_features(password))
        features.update(self.extract_pattern_features(password))
        features.update(self.extract_linguistic_features(password))
        
        return features
    
    def extract_features_batch(self, passwords: List[str]) -> pd.DataFrame:
        """Extract features for a batch of passwords.
        
        Args:
            passwords: List of passwords to analyze
            
        Returns:
            DataFrame with extracted features
        """
        feature_dicts = []
        
        for password in passwords:
            features = self.extract_all_features(password)
            feature_dicts.append(features)
        
        return pd.DataFrame(feature_dicts)
    
    def fit_transform(self, passwords: List[str]) -> np.ndarray:
        """Fit the feature extractor and transform passwords to feature matrix.
        
        Args:
            passwords: List of passwords for training
            
        Returns:
            Feature matrix
        """
        # Extract features
        feature_df = self.extract_features_batch(passwords)
        
        # Fit scaler
        feature_matrix = self.scaler.fit_transform(feature_df.values)
        
        # Store feature names
        self.feature_names = feature_df.columns.tolist()
        self.is_fitted = True
        
        return feature_matrix
    
    def transform(self, passwords: List[str]) -> np.ndarray:
        """Transform passwords to feature matrix using fitted extractor.
        
        Args:
            passwords: List of passwords to transform
            
        Returns:
            Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        # Extract features
        feature_df = self.extract_features_batch(passwords)
        
        # Transform using fitted scaler
        feature_matrix = self.scaler.transform(feature_df.values)
        
        return feature_matrix
    
    def get_feature_names(self) -> List[str]:
        """Get the names of extracted features.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted first")
        return self.feature_names
    
    def get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance from trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted first")
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
        
        importance_dict = dict(zip(self.feature_names, model.feature_importances_))
        return importance_dict
    
    def explain_prediction(self, password: str, model, top_k: int = 10) -> Dict[str, any]:
        """Explain prediction for a single password.
        
        Args:
            password: Password to explain
            model: Trained model
            top_k: Number of top features to return
            
        Returns:
            Dictionary with explanation details
        """
        # Extract features
        features = self.extract_all_features(password)
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # Get prediction
        prediction = model.predict(feature_vector)[0]
        prediction_proba = model.predict_proba(feature_vector)[0]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, model.feature_importances_))
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:top_k]
        else:
            top_features = []
        
        return {
            "password": password,
            "prediction": prediction,
            "prediction_proba": prediction_proba,
            "top_features": top_features,
            "all_features": features
        }
