#!/usr/bin/env python3
"""
Project 896: Password Strength Evaluation - Modernized Demo

This is a simple demo script showcasing the modernized password strength evaluation system.
For the full system with machine learning, interactive demo, and comprehensive analysis,
see the main project structure in the src/ directory.

Run the full system:
1. python scripts/train.py --data_size 10000
2. streamlit run demo/app.py
"""

import re
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import (
    calculate_entropy,
    detect_keyboard_patterns,
    detect_sequential_patterns,
    detect_repeated_patterns,
    estimate_time_to_crack,
    format_time_duration,
)


def evaluate_password_basic(password: str) -> str:
    """Basic password strength evaluation (original implementation)."""
    length = len(password)
    has_upper = bool(re.search(r'[A-Z]', password))
    has_lower = bool(re.search(r'[a-z]', password))
    has_digit = bool(re.search(r'\d', password))
    has_symbol = bool(re.search(r'\W', password))  # Non-alphanumeric
    score = sum([has_upper, has_lower, has_digit, has_symbol])
 
    # Rule-based classification
    if length >= 12 and score == 4:
        return 'Strong'
    elif length >= 8 and score >= 3:
        return 'Moderate'
    else:
        return 'Weak'


def evaluate_password_advanced(password: str) -> dict:
    """Advanced password strength evaluation with detailed analysis."""
    # Basic evaluation
    basic_strength = evaluate_password_basic(password)
    
    # Advanced analysis
    entropy = calculate_entropy(password)
    time_to_crack = estimate_time_to_crack(password)
    
    # Pattern detection
    keyboard_patterns = detect_keyboard_patterns(password)
    sequential_patterns = detect_sequential_patterns(password)
    repeated_patterns = detect_repeated_patterns(password)
    
    # Character analysis
    char_analysis = {
        "length": len(password),
        "lowercase": sum(1 for c in password if c.islower()),
        "uppercase": sum(1 for c in password if c.isupper()),
        "digits": sum(1 for c in password if c.isdigit()),
        "symbols": sum(1 for c in password if not c.isalnum()),
        "unique_chars": len(set(password))
    }
    
    # Calculate diversity
    diversity = char_analysis["unique_chars"] / char_analysis["length"] if char_analysis["length"] > 0 else 0
    
    return {
        "password": password,
        "basic_strength": basic_strength,
        "entropy": entropy,
        "time_to_crack": time_to_crack,
        "time_to_crack_formatted": format_time_duration(time_to_crack),
        "diversity": diversity,
        "patterns": {
            "keyboard": keyboard_patterns,
            "sequential": sequential_patterns,
            "repeated": repeated_patterns
        },
        "char_analysis": char_analysis
    }


def main():
    """Main demo function."""
    print("=" * 60)
    print("PASSWORD STRENGTH EVALUATION DEMO")
    print("=" * 60)
    print()
    
    # Test passwords
    test_passwords = [
        "password",              # weak
        "Pass1234",              # moderate
        "Str0ng!Pass2024",       # strong
        "letmein",               # weak
        "Admin@987",             # moderate
        "aB1@",                  # weak (short)
        "qwerty123",             # weak (keyboard pattern)
        "123456789",             # weak (sequential)
        "MyS3cur3P@ssw0rd!",     # strong
        "abcdefgh",              # weak (sequential letters)
    ]
    
    print("BASIC EVALUATION:")
    print("-" * 30)
    for pwd in test_passwords:
        strength = evaluate_password_basic(pwd)
        print(f"{pwd:20} -> {strength}")
    
    print("\nADVANCED EVALUATION:")
    print("-" * 30)
    
    for pwd in test_passwords:
        analysis = evaluate_password_advanced(pwd)
        
        print(f"\nPassword: {analysis['password']}")
        print(f"Basic Strength: {analysis['basic_strength']}")
        print(f"Entropy: {analysis['entropy']:.2f} bits")
        print(f"Time to Crack: {analysis['time_to_crack_formatted']}")
        print(f"Character Diversity: {analysis['diversity']:.2f}")
        
        # Pattern analysis
        total_patterns = (len(analysis['patterns']['keyboard']) + 
                         len(analysis['patterns']['sequential']) + 
                         len(analysis['patterns']['repeated']))
        
        if total_patterns > 0:
            print(f"Weak Patterns Detected ({total_patterns}):")
            for pattern_type, patterns in analysis['patterns'].items():
                if patterns:
                    print(f"  {pattern_type}: {', '.join(patterns)}")
        else:
            print("No weak patterns detected")
        
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print()
    print("For the full system with machine learning and interactive demo:")
    print("1. python scripts/train.py --data_size 10000")
    print("2. streamlit run demo/app.py")
    print()
    print("See README.md for complete documentation.")


if __name__ == "__main__":
    main()

