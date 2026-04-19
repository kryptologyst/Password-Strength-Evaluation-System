"""Comprehensive evaluation module for password strength models."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from ..models import EntropyBasedModel
from ..utils import anonymize_output


class PasswordStrengthEvaluator:
    """Comprehensive evaluator for password strength models."""
    
    def __init__(self, config: DictConfig):
        """Initialize the evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Evaluation settings
        self.metrics = config.metrics
        self.thresholds = config.thresholds
        self.cross_validation = config.cross_validation
        
    def evaluate_model(self, model: EntropyBasedModel, 
                      X_test: Union[List[str], pd.DataFrame],
                      y_test: List[str]) -> Dict[str, any]:
        """Evaluate model performance comprehensively.
        
        Args:
            model: Trained model to evaluate
            X_test: Test passwords
            y_test: True test labels
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info("Starting comprehensive model evaluation")
        
        # Get predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # Calculate classification metrics
        classification_metrics = self._calculate_classification_metrics(y_test, predictions)
        
        # Calculate security-specific metrics
        security_metrics = self._calculate_security_metrics(model, X_test, y_test)
        
        # Calculate robustness metrics
        robustness_metrics = self._calculate_robustness_metrics(model, X_test, y_test)
        
        # Cross-validation if enabled
        cv_results = None
        if self.cross_validation.enabled:
            cv_results = self._cross_validate_model(model, X_test, y_test)
        
        # Generate visualizations
        plots = self._generate_evaluation_plots(y_test, predictions, probabilities)
        
        return {
            "classification_metrics": classification_metrics,
            "security_metrics": security_metrics,
            "robustness_metrics": robustness_metrics,
            "cross_validation": cv_results,
            "plots": plots,
            "predictions": predictions,
            "probabilities": probabilities
        }
    
    def _calculate_classification_metrics(self, y_true: List[str], 
                                       y_pred: List[str]) -> Dict[str, float]:
        """Calculate standard classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with classification metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro")
        metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro")
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
        
        metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted")
        metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted")
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")
        
        # Per-class metrics
        unique_labels = sorted(set(y_true + y_pred))
        for label in unique_labels:
            metrics[f"precision_{label}"] = precision_score(
                y_true, y_pred, labels=[label], average="micro"
            )
            metrics[f"recall_{label}"] = recall_score(
                y_true, y_pred, labels=[label], average="micro"
            )
            metrics[f"f1_{label}"] = f1_score(
                y_true, y_pred, labels=[label], average="micro"
            )
        
        return metrics
    
    def _calculate_security_metrics(self, model: EntropyBasedModel,
                                  X_test: Union[List[str], pd.DataFrame],
                                  y_test: List[str]) -> Dict[str, float]:
        """Calculate security-specific metrics.
        
        Args:
            model: Trained model
            X_test: Test passwords
            y_test: True test labels
            
        Returns:
            Dictionary with security metrics
        """
        metrics = {}
        
        if isinstance(X_test, pd.DataFrame):
            passwords = X_test['password'].tolist()
        else:
            passwords = X_test
        
        # Entropy correlation
        entropies = []
        for password in passwords:
            from ..utils import calculate_entropy
            entropies.append(calculate_entropy(password))
        
        # Convert labels to numeric for correlation
        label_mapping = {"weak": 0, "moderate": 1, "strong": 2}
        y_numeric = [label_mapping[label] for label in y_test]
        
        metrics["entropy_correlation"] = np.corrcoef(entropies, y_numeric)[0, 1]
        
        # Time to crack correlation
        times_to_crack = []
        for password in passwords:
            from ..utils import estimate_time_to_crack
            times_to_crack.append(estimate_time_to_crack(password))
        
        metrics["time_to_crack_correlation"] = np.corrcoef(
            np.log(times_to_crack), y_numeric
        )[0, 1]
        
        # Breach detection precision
        breach_passwords = [pwd for pwd, label in zip(passwords, y_test) 
                          if label == "weak"]
        if breach_passwords:
            breach_predictions = model.predict(breach_passwords)
            metrics["breach_detection_precision"] = np.mean(
                [pred == "weak" for pred in breach_predictions]
            )
        else:
            metrics["breach_detection_precision"] = 0.0
        
        # Pattern detection recall
        pattern_passwords = []
        for password in passwords:
            from ..utils import (
                detect_keyboard_patterns,
                detect_sequential_patterns,
                detect_repeated_patterns
            )
            patterns = (detect_keyboard_patterns(password) + 
                        detect_sequential_patterns(password) + 
                        detect_repeated_patterns(password))
            if patterns:
                pattern_passwords.append(password)
        
        if pattern_passwords:
            pattern_predictions = model.predict(pattern_passwords)
            metrics["pattern_detection_recall"] = np.mean(
                [pred in ["weak", "moderate"] for pred in pattern_predictions]
            )
        else:
            metrics["pattern_detection_recall"] = 0.0
        
        return metrics
    
    def _calculate_robustness_metrics(self, model: EntropyBasedModel,
                                    X_test: Union[List[str], pd.DataFrame],
                                    y_test: List[str]) -> Dict[str, float]:
        """Calculate robustness metrics.
        
        Args:
            model: Trained model
            X_test: Test passwords
            y_test: True test labels
            
        Returns:
            Dictionary with robustness metrics
        """
        metrics = {}
        
        # Threshold sensitivity
        probabilities = model.predict_proba(X_test)
        threshold_scores = []
        
        for threshold in np.arange(0.1, 0.9, 0.1):
            # Adjust predictions based on threshold
            adjusted_predictions = []
            for prob in probabilities:
                if prob[0] > threshold:  # weak
                    adjusted_predictions.append("weak")
                elif prob[1] > threshold:  # moderate
                    adjusted_predictions.append("moderate")
                else:  # strong
                    adjusted_predictions.append("strong")
            
            accuracy = accuracy_score(y_test, adjusted_predictions)
            threshold_scores.append(accuracy)
        
        metrics["threshold_sensitivity"] = np.std(threshold_scores)
        metrics["threshold_stability"] = np.mean(threshold_scores)
        
        # Adversarial robustness (simplified)
        adversarial_scores = []
        if isinstance(X_test, pd.DataFrame):
            passwords = X_test['password'].tolist()
        else:
            passwords = X_test
        
        for password in passwords[:100]:  # Sample for efficiency
            # Simple adversarial examples (add/remove characters)
            adversarial_passwords = [
                password + "1",
                password + "!",
                password[:-1] if len(password) > 1 else password,
                password.replace("a", "@") if "a" in password else password
            ]
            
            original_pred = model.predict([password])[0]
            adversarial_preds = model.predict(adversarial_passwords)
            
            robustness = np.mean([pred == original_pred for pred in adversarial_preds])
            adversarial_scores.append(robustness)
        
        metrics["adversarial_robustness"] = np.mean(adversarial_scores)
        
        return metrics
    
    def _cross_validate_model(self, model: EntropyBasedModel,
                            X: Union[List[str], pd.DataFrame],
                            y: List[str]) -> Dict[str, any]:
        """Perform cross-validation on the model.
        
        Args:
            model: Model to validate
            X: Training data
            y: Training labels
            
        Returns:
            Dictionary with cross-validation results
        """
        if isinstance(X, pd.DataFrame):
            passwords = X['password'].tolist()
        else:
            passwords = X
        
        # Extract features
        feature_matrix = model.feature_extractor.fit_transform(passwords)
        
        # Cross-validation
        cv = StratifiedKFold(
            n_splits=self.cross_validation.folds,
            shuffle=self.cross_validation.shuffle,
            random_state=42
        )
        
        cv_scores = cross_val_score(
            model.model,
            feature_matrix,
            y,
            cv=cv,
            scoring='accuracy'
        )
        
        return {
            "scores": cv_scores,
            "mean": cv_scores.mean(),
            "std": cv_scores.std(),
            "folds": self.cross_validation.folds
        }
    
    def _generate_evaluation_plots(self, y_true: List[str], 
                                 y_pred: List[str],
                                 probabilities: np.ndarray) -> Dict[str, plt.Figure]:
        """Generate evaluation plots.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            probabilities: Prediction probabilities
            
        Returns:
            Dictionary with matplotlib figures
        """
        plots = {}
        
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plots['confusion_matrix'] = fig
        
        # ROC Curves (for each class)
        fig, ax = plt.subplots(figsize=(8, 6))
        unique_labels = sorted(set(y_true))
        
        for i, label in enumerate(unique_labels):
            y_binary = [1 if l == label else 0 for l in y_true]
            prob_binary = probabilities[:, i]
            
            fpr, tpr, _ = roc_curve(y_binary, prob_binary)
            auc = roc_auc_score(y_binary, prob_binary)
            
            ax.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        plots['roc_curves'] = fig
        
        # Precision-Recall Curves
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for i, label in enumerate(unique_labels):
            y_binary = [1 if l == label else 0 for l in y_true]
            prob_binary = probabilities[:, i]
            
            precision, recall, _ = precision_recall_curve(y_binary, prob_binary)
            
            ax.plot(recall, precision, label=f'{label}')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        plots['precision_recall'] = fig
        
        return plots
    
    def generate_report(self, evaluation_results: Dict[str, any],
                       output_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_model
            output_path: Optional path to save report
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 60)
        report.append("PASSWORD STRENGTH EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Classification Metrics
        report.append("CLASSIFICATION METRICS")
        report.append("-" * 30)
        cm = evaluation_results["classification_metrics"]
        for metric, value in cm.items():
            report.append(f"{metric}: {value:.4f}")
        report.append("")
        
        # Security Metrics
        report.append("SECURITY-SPECIFIC METRICS")
        report.append("-" * 30)
        sm = evaluation_results["security_metrics"]
        for metric, value in sm.items():
            report.append(f"{metric}: {value:.4f}")
        report.append("")
        
        # Robustness Metrics
        report.append("ROBUSTNESS METRICS")
        report.append("-" * 30)
        rm = evaluation_results["robustness_metrics"]
        for metric, value in rm.items():
            report.append(f"{metric}: {value:.4f}")
        report.append("")
        
        # Cross-Validation Results
        if evaluation_results["cross_validation"]:
            report.append("CROSS-VALIDATION RESULTS")
            report.append("-" * 30)
            cv = evaluation_results["cross_validation"]
            report.append(f"Mean CV Score: {cv['mean']:.4f}")
            report.append(f"CV Std: {cv['std']:.4f}")
            report.append(f"Folds: {cv['folds']}")
            report.append("")
        
        # Feature Importance
        report.append("TOP FEATURES")
        report.append("-" * 30)
        # This would need to be implemented based on the model type
        report.append("Feature importance analysis not implemented in this version")
        report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to {output_path}")
        
        return report_text
    
    def create_leaderboard(self, results: List[Dict[str, any]]) -> pd.DataFrame:
        """Create a leaderboard comparing multiple models.
        
        Args:
            results: List of evaluation results for different models
            
        Returns:
            DataFrame with leaderboard
        """
        leaderboard_data = []
        
        for i, result in enumerate(results):
            row = {
                "Model": f"Model_{i+1}",
                "Accuracy": result["classification_metrics"]["accuracy"],
                "F1_Macro": result["classification_metrics"]["f1_macro"],
                "Entropy_Correlation": result["security_metrics"]["entropy_correlation"],
                "Adversarial_Robustness": result["robustness_metrics"]["adversarial_robustness"],
                "CV_Mean": result["cross_validation"]["mean"] if result["cross_validation"] else 0,
                "CV_Std": result["cross_validation"]["std"] if result["cross_validation"] else 0
            }
            leaderboard_data.append(row)
        
        leaderboard = pd.DataFrame(leaderboard_data)
        leaderboard = leaderboard.sort_values("Accuracy", ascending=False)
        
        return leaderboard
