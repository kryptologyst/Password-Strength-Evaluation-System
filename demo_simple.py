#!/usr/bin/env python3
"""Simple demo script to test the password strength evaluation system."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import (
    calculate_entropy,
    detect_keyboard_patterns,
    detect_sequential_patterns,
    detect_repeated_patterns,
    estimate_time_to_crack,
    format_time_duration,
)


def demo_password_analysis():
    """Demo the password analysis functionality."""
    print("Password Strength Analysis Demo")
    print("=" * 40)
    
    test_passwords = [
        "password",
        "Pass1234", 
        "Str0ng!Pass2024",
        "qwerty123",
        "123456789",
        "MyS3cur3P@ssw0rd!"
    ]
    
    for password in test_passwords:
        print(f"\nPassword: {password}")
        print(f"Length: {len(password)}")
        print(f"Entropy: {calculate_entropy(password):.2f} bits")
        print(f"Time to crack: {format_time_duration(estimate_time_to_crack(password))}")
        
        # Pattern detection
        keyboard = detect_keyboard_patterns(password)
        sequential = detect_sequential_patterns(password)
        repeated = detect_repeated_patterns(password)
        
        patterns = keyboard + sequential + repeated
        if patterns:
            print(f"Weak patterns: {', '.join(patterns)}")
        else:
            print("No weak patterns detected")
        
        print("-" * 30)


if __name__ == "__main__":
    demo_password_analysis()
