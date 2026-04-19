"""Synthetic password data generation for training and evaluation."""

import random
import string
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig


class SyntheticPasswordGenerator:
    """Generate synthetic password datasets for training and evaluation."""
    
    def __init__(self, config: DictConfig):
        """Initialize the password generator.
        
        Args:
            config: Configuration object with generation parameters
        """
        self.config = config
        self.character_sets = config.character_sets
        self.pattern_probabilities = config.pattern_probabilities
        self.weak_patterns = config.weak_patterns
        
    def generate_password(self, pattern_type: str = "random") -> str:
        """Generate a single password based on specified pattern.
        
        Args:
            pattern_type: Type of pattern to generate (random, dictionary_word, etc.)
            
        Returns:
            Generated password string
        """
        if pattern_type == "random":
            return self._generate_random_password()
        elif pattern_type == "dictionary_word":
            return self._generate_dictionary_password()
        elif pattern_type == "keyboard_pattern":
            return self._generate_keyboard_pattern_password()
        elif pattern_type == "sequential":
            return self._generate_sequential_password()
        elif pattern_type == "repeated_chars":
            return self._generate_repeated_password()
        elif pattern_type == "personal_info":
            return self._generate_personal_info_password()
        else:
            return self._generate_random_password()
    
    def _generate_random_password(self) -> str:
        """Generate a random password with mixed character types."""
        length = random.randint(
            self.config.password_lengths.min,
            self.config.password_lengths.max
        )
        
        password = ""
        char_sets = []
        
        # Ensure at least one character from each set
        if random.random() < 0.8:  # 80% chance to include lowercase
            char_sets.append(self.character_sets.lowercase)
        if random.random() < 0.6:  # 60% chance to include uppercase
            char_sets.append(self.character_sets.uppercase)
        if random.random() < 0.7:  # 70% chance to include digits
            char_sets.append(self.character_sets.digits)
        if random.random() < 0.4:  # 40% chance to include symbols
            char_sets.append(self.character_sets.symbols)
        
        # If no character sets selected, use lowercase
        if not char_sets:
            char_sets = [self.character_sets.lowercase]
        
        # Generate password
        for _ in range(length):
            char_set = random.choice(char_sets)
            password += random.choice(char_set)
        
        return password
    
    def _generate_dictionary_password(self) -> str:
        """Generate password based on dictionary words."""
        # Common dictionary words
        dictionary_words = [
            "password", "admin", "welcome", "hello", "world",
            "computer", "internet", "security", "system", "user",
            "login", "access", "account", "profile", "settings"
        ]
        
        base_word = random.choice(dictionary_words)
        
        # Modify the word
        modifications = [
            lambda w: w + str(random.randint(1000, 9999)),
            lambda w: w.capitalize() + "!",
            lambda w: w + random.choice(self.character_sets.symbols),
            lambda w: w.replace("a", "@").replace("e", "3").replace("o", "0"),
            lambda w: w + w[::-1][:3],  # Add reversed suffix
        ]
        
        modification = random.choice(modifications)
        return modification(base_word)
    
    def _generate_keyboard_pattern_password(self) -> str:
        """Generate password based on keyboard patterns."""
        patterns = [
            "qwerty", "asdf", "zxcv", "123456", "abcdef",
            "qwertyuiop", "asdfghjkl", "zxcvbnm"
        ]
        
        pattern = random.choice(patterns)
        
        # Sometimes add numbers or symbols
        if random.random() < 0.5:
            pattern += str(random.randint(10, 99))
        if random.random() < 0.3:
            pattern += random.choice(self.character_sets.symbols)
        
        return pattern
    
    def _generate_sequential_password(self) -> str:
        """Generate password with sequential patterns."""
        patterns = [
            "123456789", "abcdefgh", "ABCDEFGH",
            "987654321", "zyxwvuts", "ZYXWVUTS"
        ]
        
        pattern = random.choice(patterns)
        length = random.randint(4, min(8, len(pattern)))
        
        start_idx = random.randint(0, len(pattern) - length)
        return pattern[start_idx:start_idx + length]
    
    def _generate_repeated_password(self) -> str:
        """Generate password with repeated characters."""
        char = random.choice(string.ascii_letters + string.digits)
        length = random.randint(4, 8)
        
        password = char * length
        
        # Sometimes add a suffix
        if random.random() < 0.5:
            password += str(random.randint(10, 99))
        
        return password
    
    def _generate_personal_info_password(self) -> str:
        """Generate password based on personal information patterns."""
        # Common personal info patterns
        personal_patterns = [
            "john123", "jane2024", "mike01", "sarah99",
            "admin2024", "user123", "test123", "demo456"
        ]
        
        return random.choice(personal_patterns)
    
    def generate_dataset(self, size: int) -> pd.DataFrame:
        """Generate a dataset of passwords with labels.
        
        Args:
            size: Number of passwords to generate
            
        Returns:
            DataFrame with passwords and strength labels
        """
        passwords = []
        labels = []
        patterns = []
        
        for _ in range(size):
            # Choose pattern type based on probabilities
            pattern_type = np.random.choice(
                list(self.pattern_probabilities.keys()),
                p=list(self.pattern_probabilities.values())
            )
            
            password = self.generate_password(pattern_type)
            strength_label = self._evaluate_strength(password)
            
            passwords.append(password)
            labels.append(strength_label)
            patterns.append(pattern_type)
        
        return pd.DataFrame({
            "password": passwords,
            "strength_label": labels,
            "pattern_type": patterns
        })
    
    def _evaluate_strength(self, password: str) -> str:
        """Evaluate password strength based on basic rules.
        
        Args:
            password: Password to evaluate
            
        Returns:
            Strength label (weak, moderate, strong)
        """
        length = len(password)
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_symbol = any(not c.isalnum() for c in password)
        
        # Check if password is in weak patterns
        if password.lower() in [p.lower() for p in self.weak_patterns]:
            return "weak"
        
        # Calculate complexity score
        complexity_score = sum([has_upper, has_lower, has_digit, has_symbol])
        
        # Rule-based classification
        if length >= 12 and complexity_score >= 3:
            return "strong"
        elif length >= 8 and complexity_score >= 2:
            return "moderate"
        else:
            return "weak"
    
    def generate_breach_dataset(self, size: int) -> pd.DataFrame:
        """Generate dataset simulating breached passwords.
        
        Args:
            size: Number of passwords to generate
            
        Returns:
            DataFrame with breached passwords
        """
        passwords = []
        breach_sources = []
        
        # Generate from common breached passwords
        for _ in range(size):
            if random.random() < self.config.breach_simulation.breach_probability:
                # Use common breached password
                password = random.choice(self.weak_patterns)
                source = "common_breach"
            else:
                # Generate weak password
                password = self.generate_password("dictionary_word")
                source = "synthetic_weak"
            
            passwords.append(password)
            breach_sources.append(source)
        
        return pd.DataFrame({
            "password": passwords,
            "breach_source": breach_sources,
            "is_breached": True
        })
    
    def split_dataset(self, df: pd.DataFrame, train_ratio: float = 0.7, 
                     val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/validation/test sets.
        
        Args:
            df: Input DataFrame
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Shuffle the dataset
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n_samples = len(df_shuffled)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        train_df = df_shuffled[:train_size]
        val_df = df_shuffled[train_size:train_size + val_size]
        test_df = df_shuffled[train_size + val_size:]
        
        return train_df, val_df, test_df
