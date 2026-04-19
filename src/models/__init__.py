"""Entropy-based password strength evaluation model."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

from ..features import PasswordFeatureExtractor
from ..utils import (
    calculate_entropy,
    detect_keyboard_patterns,
    detect_repeated_patterns,
    detect_sequential_patterns,
    estimate_time_to_crack,
)


class EntropyBasedModel(BaseEstimator, ClassifierMixin):
    """Entropy-based password strength evaluation model."""
    
    def __init__(self, config: DictConfig):
        """Initialize the entropy-based model.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.feature_extractor = PasswordFeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_fitted = False
        
        # Configuration parameters
        self.entropy_thresholds = config.entropy_thresholds
        self.complexity_weights = config.complexity_weights
        
    def _calculate_password_score(self, password: str) -> float:
        """Calculate comprehensive password strength score.
        
        Args:
            password: Password to evaluate
            
        Returns:
            Strength score between 0 and 1
        """
        if not password:
            return 0.0
        
        # Length component
        length_score = min(len(password) / 20.0, 1.0)  # Normalize to max 20 chars
        
        # Character diversity component
        unique_chars = len(set(password))
        diversity_score = unique_chars / len(password) if len(password) > 0 else 0
        
        # Entropy component
        entropy = calculate_entropy(password)
        entropy_score = min(entropy / 6.0, 1.0)  # Normalize to max 6 bits
        
        # Pattern penalty component
        keyboard_patterns = detect_keyboard_patterns(password)
        sequential_patterns = detect_sequential_patterns(password)
        repeated_patterns = detect_repeated_patterns(password)
        
        pattern_penalty = len(keyboard_patterns) + len(sequential_patterns) + len(repeated_patterns)
        pattern_score = max(0.0, 1.0 - (pattern_penalty * 0.2))
        
        # Combine components with weights
        total_score = (
            self.complexity_weights.length * length_score +
            self.complexity_weights.character_diversity * diversity_score +
            self.complexity_weights.entropy * entropy_score +
            self.complexity_weights.pattern_detection * pattern_score
        )
        
        return min(max(total_score, 0.0), 1.0)
    
    def _classify_password(self, password: str) -> str:
        """Classify password strength based on entropy and patterns.
        
        Args:
            password: Password to classify
            
        Returns:
            Strength classification (weak, moderate, strong)
        """
        score = self._calculate_password_score(password)
        
        if score < self.config.classification.weak_threshold / 100.0:
            return "weak"
        elif score < self.config.classification.moderate_threshold / 100.0:
            return "moderate"
        else:
            return "strong"
    
    def fit(self, X: Union[List[str], pd.DataFrame], y: Optional[List[str]] = None) -> 'EntropyBasedModel':
        """Fit the entropy-based model.
        
        Args:
            X: Training passwords
            y: Training labels (optional, can be generated from passwords)
            
        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            passwords = X['password'].tolist()
        else:
            passwords = X
        
        # Generate labels if not provided
        if y is None:
            labels = [self._classify_password(pwd) for pwd in passwords]
        else:
            labels = y
        
        # Encode labels
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        
        # Extract features
        feature_matrix = self.feature_extractor.fit_transform(passwords)
        
        # Train model (using Random Forest as default)
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        self.model.fit(feature_matrix, encoded_labels)
        self.is_fitted = True
        
        self.logger.info(f"Model fitted on {len(passwords)} passwords")
        return self
    
    def predict(self, X: Union[List[str], pd.DataFrame]) -> np.ndarray:
        """Predict password strength classes.
        
        Args:
            X: Passwords to predict
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, pd.DataFrame):
            passwords = X['password'].tolist()
        else:
            passwords = X
        
        # Extract features
        feature_matrix = self.feature_extractor.transform(passwords)
        
        # Predict
        encoded_predictions = self.model.predict(feature_matrix)
        
        # Decode predictions
        predictions = self.label_encoder.inverse_transform(encoded_predictions)
        
        return predictions
    
    def predict_proba(self, X: Union[List[str], pd.DataFrame]) -> np.ndarray:
        """Predict password strength class probabilities.
        
        Args:
            X: Passwords to predict
            
        Returns:
            Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, pd.DataFrame):
            passwords = X['password'].tolist()
        else:
            passwords = X
        
        # Extract features
        feature_matrix = self.feature_extractor.transform(passwords)
        
        # Predict probabilities
        probabilities = self.model.predict_proba(feature_matrix)
        
        return probabilities
    
    def score(self, X: Union[List[str], pd.DataFrame], y: List[str]) -> float:
        """Calculate model accuracy.
        
        Args:
            X: Test passwords
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return self.feature_extractor.get_feature_importance(self.model)
    
    def explain_prediction(self, password: str, top_k: int = 10) -> Dict[str, any]:
        """Explain prediction for a single password.
        
        Args:
            password: Password to explain
            top_k: Number of top features to return
            
        Returns:
            Dictionary with explanation details
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before explanation")
        
        return self.feature_extractor.explain_prediction(password, self.model, top_k)
    
    def evaluate(self, X_test: Union[List[str], pd.DataFrame], 
                y_test: List[str]) -> Dict[str, any]:
        """Evaluate model performance on test data.
        
        Args:
            X_test: Test passwords
            y_test: True test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Get predictions
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = self.score(X_test, y_test)
        
        # Classification report
        report = classification_report(y_test, predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Cross-validation score
        if isinstance(X_test, pd.DataFrame):
            passwords = X_test['password'].tolist()
        else:
            passwords = X_test
        
        feature_matrix = self.feature_extractor.transform(passwords)
        cv_scores = cross_val_score(self.model, feature_matrix, y_test, cv=5)
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm,
            "cross_val_scores": cv_scores,
            "cross_val_mean": cv_scores.mean(),
            "cross_val_std": cv_scores.std(),
            "predictions": predictions,
            "probabilities": probabilities
        }
    
    def optimize_hyperparameters(self, X: Union[List[str], pd.DataFrame], 
                               y: List[str]) -> Dict[str, any]:
        """Optimize model hyperparameters using grid search.
        
        Args:
            X: Training passwords
            y: Training labels
            
        Returns:
            Dictionary with optimization results
        """
        if isinstance(X, pd.DataFrame):
            passwords = X['password'].tolist()
        else:
            passwords = X
        
        # Extract features
        feature_matrix = self.feature_extractor.fit_transform(passwords)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(feature_matrix, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        import joblib
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_extractor': self.feature_extractor,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from file.
        
        Args:
            filepath: Path to load the model from
        """
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_extractor = model_data['feature_extractor']
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        
        self.logger.info(f"Model loaded from {filepath}")
