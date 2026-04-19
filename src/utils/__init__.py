"""Utility functions for password strength evaluation system."""

import hashlib
import logging
import os
import random
import re
import string
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for all random number generators.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(device_preference: str = "auto") -> torch.device:
    """Get the best available device for computation.
    
    Args:
        device_preference: Preferred device (auto, cpu, cuda, mps)
        
    Returns:
        PyTorch device object
    """
    if device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_preference)


def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Hash a password for privacy protection.
    
    Args:
        password: Password to hash
        salt: Optional salt for hashing
        
    Returns:
        Hashed password string
    """
    if salt is None:
        salt = os.urandom(32).hex()
    
    # Use PBKDF2 for secure hashing
    hashed = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
    )
    return f"{salt}:{hashed.hex()}"


def anonymize_output(text: str, max_length: int = 50) -> str:
    """Anonymize text output by truncating and masking.
    
    Args:
        text: Text to anonymize
        max_length: Maximum length before truncation
        
    Returns:
        Anonymized text
    """
    if len(text) <= max_length:
        return text
    
    # Truncate and add ellipsis
    truncated = text[:max_length - 3]
    return f"{truncated}..."


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, output_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        output_path: Output file path
    """
    OmegaConf.save(config, output_path)


def validate_password_input(password: str) -> Tuple[bool, Optional[str]]:
    """Validate password input for security and safety.
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(password, str):
        return False, "Password must be a string"
    
    if len(password) == 0:
        return False, "Password cannot be empty"
    
    if len(password) > 128:
        return False, "Password too long (max 128 characters)"
    
    # Check for potentially malicious patterns
    suspicious_patterns = [
        r"<script",
        r"javascript:",
        r"data:",
        r"vbscript:",
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, password, re.IGNORECASE):
            return False, f"Suspicious pattern detected: {pattern}"
    
    return True, None


def calculate_entropy(password: str) -> float:
    """Calculate Shannon entropy of a password.
    
    Args:
        password: Password to analyze
        
    Returns:
        Shannon entropy in bits
    """
    if not password:
        return 0.0
    
    # Count character frequencies
    char_counts = {}
    for char in password:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Calculate entropy
    entropy = 0.0
    password_length = len(password)
    
    for count in char_counts.values():
        probability = count / password_length
        if probability > 0:
            entropy -= probability * np.log2(probability)
    
    return entropy


def detect_keyboard_patterns(password: str) -> List[str]:
    """Detect common keyboard patterns in password.
    
    Args:
        password: Password to analyze
        
    Returns:
        List of detected patterns
    """
    patterns = []
    
    # Common keyboard sequences
    keyboard_sequences = [
        "qwerty", "asdf", "zxcv", "123456", "abcdef",
        "qwertyuiop", "asdfghjkl", "zxcvbnm",
        "qaz", "wsx", "edc", "rfv", "tgb", "yhn", "ujm", "ik", "ol", "p"
    ]
    
    password_lower = password.lower()
    
    for sequence in keyboard_sequences:
        if sequence in password_lower:
            patterns.append(f"keyboard_sequence_{sequence}")
    
    return patterns


def detect_sequential_patterns(password: str) -> List[str]:
    """Detect sequential patterns in password.
    
    Args:
        password: Password to analyze
        
    Returns:
        List of detected patterns
    """
    patterns = []
    
    # Check for sequential numbers
    for i in range(len(password) - 2):
        if password[i:i+3].isdigit():
            if (int(password[i+1]) - int(password[i]) == 1 and 
                int(password[i+2]) - int(password[i+1]) == 1):
                patterns.append("sequential_numbers")
                break
    
    # Check for sequential letters
    for i in range(len(password) - 2):
        if password[i:i+3].isalpha():
            if (ord(password[i+1].lower()) - ord(password[i].lower()) == 1 and 
                ord(password[i+2].lower()) - ord(password[i+1].lower()) == 1):
                patterns.append("sequential_letters")
                break
    
    return patterns


def detect_repeated_patterns(password: str) -> List[str]:
    """Detect repeated character patterns in password.
    
    Args:
        password: Password to analyze
        
    Returns:
        List of detected patterns
    """
    patterns = []
    
    # Check for repeated characters
    for i in range(len(password) - 2):
        if password[i] == password[i+1] == password[i+2]:
            patterns.append("repeated_characters")
            break
    
    # Check for repeated substrings
    for length in range(2, len(password) // 2 + 1):
        for i in range(len(password) - length * 2 + 1):
            substring = password[i:i+length]
            if password[i+length:i+length*2] == substring:
                patterns.append(f"repeated_substring_{length}")
                break
    
    return patterns


def estimate_time_to_crack(password: str, attempts_per_second: int = 1e9) -> float:
    """Estimate time to crack password using brute force.
    
    Args:
        password: Password to analyze
        attempts_per_second: Brute force attempts per second
        
    Returns:
        Estimated time in seconds
    """
    if not password:
        return 0.0
    
    # Determine character set size
    char_set_size = 0
    if any(c.islower() for c in password):
        char_set_size += 26
    if any(c.isupper() for c in password):
        char_set_size += 10
    if any(c.isdigit() for c in password):
        char_set_size += 10
    if any(not c.isalnum() for c in password):
        char_set_size += 32
    
    # Calculate combinations
    combinations = char_set_size ** len(password)
    
    # Average time to crack (half of total combinations)
    time_to_crack = combinations / (2 * attempts_per_second)
    
    return time_to_crack


def format_time_duration(seconds: float) -> str:
    """Format time duration in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds:.3f} seconds"
    elif seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    elif seconds < 31536000:
        return f"{seconds/86400:.1f} days"
    else:
        return f"{seconds/31536000:.1f} years"
