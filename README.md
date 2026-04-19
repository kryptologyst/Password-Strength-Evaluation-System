# Password Strength Evaluation System

A comprehensive password strength evaluation system designed for security research and education. This system provides advanced password analysis using entropy calculations, pattern detection, and machine learning-based classification.

## Features

### Core Analysis
- **Entropy Calculation**: Shannon entropy analysis for password randomness
- **Pattern Detection**: Identification of keyboard patterns, sequential sequences, and repeated characters
- **Character Analysis**: Comprehensive character type and diversity analysis
- **Time to Crack**: Brute force attack time estimation
- **Breach Simulation**: Mock breach database checking

### Machine Learning
- **Entropy-Based Model**: Advanced feature extraction and classification
- **Random Forest Classifier**: Robust ensemble learning approach
- **Cross-Validation**: Comprehensive model evaluation
- **Feature Importance**: Explainable AI insights
- **Hyperparameter Optimization**: Automated model tuning

### Security Metrics
- Length and character diversity scoring
- Entropy-based strength assessment
- Pattern recognition and penalty scoring
- Time-to-crack estimation
- Breach probability assessment

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/kryptologyst/Password-Strength-Evaluation-System.git
cd Password-Strength-Evaluation-System

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Quick Start
```bash
# Train the model
python scripts/train.py --data_size 10000

# Launch the demo
streamlit run demo/app.py
```

## Usage

### Training a Model

```bash
# Basic training with default configuration
python scripts/train.py

# Custom configuration and dataset size
python scripts/train.py --config configs/config.yaml --data_size 20000 --output_dir results

# Verbose output
python scripts/train.py --verbose
```

### Using the Demo

Launch the Streamlit demo application:

```bash
streamlit run demo/app.py
```

The demo provides:
- **Single Password Analysis**: Detailed analysis of individual passwords
- **Batch Analysis**: Process multiple passwords from CSV or text input
- **Model Insights**: View feature importance and model configuration
- **Interactive Visualizations**: Real-time charts and graphs

### Programmatic Usage

```python
from src.models import EntropyBasedModel
from src.utils import load_config

# Load configuration
config = load_config("configs/config.yaml")

# Initialize and train model
model = EntropyBasedModel(config.model)
model.fit(train_passwords, train_labels)

# Make predictions
predictions = model.predict(test_passwords)
probabilities = model.predict_proba(test_passwords)

# Get explanations
explanation = model.explain_prediction("mypassword123")
```

## Configuration

The system uses YAML configuration files for flexible setup:

### Main Configuration (`configs/config.yaml`)
```yaml
# Global settings
seed: 42
device: auto
log_level: INFO

# Data settings
data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  synthetic_size: 10000

# Model settings
model:
  name: entropy_based
  entropy_thresholds:
    weak: 20
    moderate: 40
    strong: 60
  complexity_weights:
    length: 0.3
    character_diversity: 0.2
    entropy: 0.3
    pattern_detection: 0.2
```

### Data Configuration (`configs/data/synthetic.yaml`)
```yaml
# Password generation parameters
password_lengths:
  min: 4
  max: 32

# Pattern probabilities
pattern_probabilities:
  dictionary_word: 0.3
  keyboard_pattern: 0.2
  sequential: 0.15
  repeated_chars: 0.1
  personal_info: 0.1
  random: 0.15
```

## Dataset Schema

### Training Data Format
```csv
password,strength_label,pattern_type
"password123","weak","dictionary_word"
"Str0ng!Pass2024","strong","random"
"qwerty123","weak","keyboard_pattern"
```

### Generated Features
- **Basic Features**: Length, character counts, ratios, diversity
- **Entropy Features**: Shannon entropy, character set entropy, time to crack
- **Pattern Features**: Keyboard patterns, sequential patterns, repeated patterns
- **Linguistic Features**: Dictionary words, leet speak, palindromes

## Evaluation Metrics

### Classification Metrics
- Accuracy, Precision, Recall, F1-Score (macro and weighted)
- Per-class precision, recall, and F1-score
- Confusion matrix analysis

### Security-Specific Metrics
- Entropy correlation with strength labels
- Time-to-crack correlation
- Breach detection precision
- Pattern detection recall

### Robustness Metrics
- Threshold sensitivity analysis
- Adversarial robustness testing
- Cross-validation stability

## Project Structure

```
password-strength-evaluation/
├── src/                          # Source code
│   ├── data/                     # Data generation and processing
│   ├── features/                 # Feature extraction
│   ├── models/                   # Model implementations
│   ├── eval/                     # Evaluation metrics
│   ├── viz/                      # Visualization tools
│   └── utils/                    # Utility functions
├── configs/                      # Configuration files
│   ├── data/                     # Data-specific configs
│   ├── model/                    # Model-specific configs
│   └── evaluation/               # Evaluation configs
├── scripts/                      # Training and utility scripts
├── demo/                         # Streamlit demo application
├── tests/                        # Unit tests
├── assets/                       # Generated plots and reports
├── outputs/                      # Model outputs and results
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                     # This file
```

## API Reference

### Core Classes

#### `EntropyBasedModel`
Main password strength evaluation model.

```python
class EntropyBasedModel:
    def __init__(self, config: DictConfig)
    def fit(self, X: List[str], y: List[str]) -> 'EntropyBasedModel'
    def predict(self, X: List[str]) -> np.ndarray
    def predict_proba(self, X: List[str]) -> np.ndarray
    def explain_prediction(self, password: str) -> Dict
    def get_feature_importance(self) -> Dict[str, float]
```

#### `PasswordFeatureExtractor`
Feature extraction for password analysis.

```python
class PasswordFeatureExtractor:
    def extract_all_features(self, password: str) -> Dict[str, float]
    def extract_features_batch(self, passwords: List[str]) -> pd.DataFrame
    def fit_transform(self, passwords: List[str]) -> np.ndarray
    def transform(self, passwords: List[str]) -> np.ndarray
```

#### `PasswordStrengthEvaluator`
Comprehensive model evaluation.

```python
class PasswordStrengthEvaluator:
    def evaluate_model(self, model, X_test, y_test) -> Dict
    def generate_report(self, results, output_path) -> str
    def create_leaderboard(self, results) -> pd.DataFrame
```

### Utility Functions

```python
# Password analysis
calculate_entropy(password: str) -> float
detect_keyboard_patterns(password: str) -> List[str]
detect_sequential_patterns(password: str) -> List[str]
detect_repeated_patterns(password: str) -> List[str]
estimate_time_to_crack(password: str) -> float

# Privacy and security
hash_password(password: str) -> str
anonymize_output(text: str) -> str
validate_password_input(password: str) -> Tuple[bool, str]
```

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_models.py
```

### Code Quality
```bash
# Format code
black src/ scripts/ demo/

# Lint code
ruff src/ scripts/ demo/

# Type checking
mypy src/
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Privacy and Security

### Data Protection
- Passwords are hashed using PBKDF2 for privacy
- Output is anonymized to prevent information leakage
- No passwords are stored permanently
- Input validation prevents malicious patterns

### Ethical Guidelines
- Research and educational use only
- No production security operations
- No exploitation or offensive capabilities
- Transparent about limitations and accuracy

### Compliance
- GDPR-compliant data handling
- No PII collection or storage
- Secure data processing practices
- Audit logging for transparency

## Limitations and Disclaimers

### Accuracy Limitations
- Results are estimates and may be inaccurate
- Model performance depends on training data
- Real-world password cracking may differ from estimates
- Pattern detection may have false positives/negatives

### Scope Limitations
- Designed for research and education only
- Not suitable for production security operations
- Does not replace professional security assessments
- May not detect all password weaknesses

### Technical Limitations
- Synthetic data may not reflect real-world patterns
- Model may not generalize to all password types
- Performance depends on hardware and configuration
- Some features may not work on all systems

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Make changes with tests
5. Run quality checks
6. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include comprehensive docstrings
- Write unit tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{password_strength_evaluation,
  title={Password Strength Evaluation System},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Password-Strength-Evaluation-System}
}
```

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation
- Review the code examples

## Changelog

### Version 1.0.0
- Initial release
- Entropy-based password strength evaluation
- Machine learning classification
- Interactive Streamlit demo
- Comprehensive evaluation metrics
- Privacy and security features
# Password-Strength-Evaluation-System
