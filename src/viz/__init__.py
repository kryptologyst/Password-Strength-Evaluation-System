"""Visualization module for password strength evaluation."""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

from ..utils import anonymize_output, format_time_duration


class PasswordStrengthVisualizer:
    """Visualization tools for password strength analysis."""
    
    def __init__(self, config: Dict = None):
        """Initialize the visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_password_distribution(self, passwords: List[str], 
                                 labels: List[str],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Plot distribution of password characteristics.
        
        Args:
            passwords: List of passwords
            labels: Strength labels
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Length distribution
        lengths = [len(pwd) for pwd in passwords]
        df = pd.DataFrame({'length': lengths, 'strength': labels})
        
        sns.histplot(data=df, x='length', hue='strength', kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Password Length Distribution')
        axes[0, 0].set_xlabel('Length')
        axes[0, 0].set_ylabel('Count')
        
        # Character diversity
        diversities = [len(set(pwd)) / len(pwd) if len(pwd) > 0 else 0 for pwd in passwords]
        df['diversity'] = diversities
        
        sns.boxplot(data=df, x='strength', y='diversity', ax=axes[0, 1])
        axes[0, 1].set_title('Character Diversity by Strength')
        axes[0, 1].set_xlabel('Strength')
        axes[0, 1].set_ylabel('Diversity Ratio')
        
        # Entropy distribution
        from ..utils import calculate_entropy
        entropies = [calculate_entropy(pwd) for pwd in passwords]
        df['entropy'] = entropies
        
        sns.boxplot(data=df, x='strength', y='entropy', ax=axes[1, 0])
        axes[1, 0].set_title('Entropy Distribution by Strength')
        axes[1, 0].set_xlabel('Strength')
        axes[1, 0].set_ylabel('Entropy (bits)')
        
        # Time to crack
        from ..utils import estimate_time_to_crack
        times_to_crack = [estimate_time_to_crack(pwd) for pwd in passwords]
        df['time_to_crack'] = times_to_crack
        df['time_to_crack_log'] = np.log(df['time_to_crack'] + 1)
        
        sns.boxplot(data=df, x='strength', y='time_to_crack_log', ax=axes[1, 1])
        axes[1, 1].set_title('Time to Crack (log scale) by Strength')
        axes[1, 1].set_xlabel('Strength')
        axes[1, 1].set_ylabel('Log(Time to Crack)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict[str, float],
                              top_k: int = 15,
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot feature importance.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance
            top_k: Number of top features to show
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:top_k]
        
        features, importances = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importances)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_k} Feature Importance')
        ax.invert_yaxis()  # Top feature at top
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig
    
    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str],
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(set(y_true + y_pred))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curves(self, y_true: List[str], y_prob: np.ndarray,
                       save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curves for each class.
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import roc_curve, roc_auc_score
        
        unique_labels = sorted(set(y_true))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i, label in enumerate(unique_labels):
            y_binary = [1 if l == label else 0 for l in y_true]
            prob_binary = y_prob[:, i]
            
            fpr, tpr, _ = roc_curve(y_binary, prob_binary)
            auc = roc_auc_score(y_binary, prob_binary)
            
            ax.plot(fpr, tpr, label=f'{label} (AUC = {auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"ROC curves saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curves(self, y_true: List[str], y_prob: np.ndarray,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot precision-recall curves for each class.
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        unique_labels = sorted(set(y_true))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i, label in enumerate(unique_labels):
            y_binary = [1 if l == label else 0 for l in y_true]
            prob_binary = y_prob[:, i]
            
            precision, recall, _ = precision_recall_curve(y_binary, prob_binary)
            ap = average_precision_score(y_binary, prob_binary)
            
            ax.plot(recall, precision, label=f'{label} (AP = {ap:.3f})', linewidth=2)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Precision-recall curves saved to {save_path}")
        
        return fig
    
    def create_interactive_dashboard(self, passwords: List[str], 
                                   labels: List[str],
                                   predictions: List[str],
                                   probabilities: np.ndarray) -> go.Figure:
        """Create interactive dashboard using Plotly.
        
        Args:
            passwords: List of passwords
            labels: True strength labels
            predictions: Predicted strength labels
            probabilities: Prediction probabilities
            
        Returns:
            Plotly figure
        """
        # Anonymize passwords for display
        anonymized_passwords = [anonymize_output(pwd, 10) for pwd in passwords]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Length vs Strength', 'Entropy Distribution',
                          'Prediction Confidence', 'Confusion Matrix'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Length vs Strength
        lengths = [len(pwd) for pwd in passwords]
        df = pd.DataFrame({
            'length': lengths,
            'strength': labels,
            'password': anonymized_passwords
        })
        
        for strength in sorted(set(labels)):
            strength_data = df[df['strength'] == strength]
            fig.add_trace(
                go.Scatter(
                    x=strength_data['length'],
                    y=[strength] * len(strength_data),
                    mode='markers',
                    name=f'True {strength}',
                    text=strength_data['password'],
                    hovertemplate='<b>%{text}</b><br>Length: %{x}<br>Strength: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Entropy distribution
        from ..utils import calculate_entropy
        entropies = [calculate_entropy(pwd) for pwd in passwords]
        
        for strength in sorted(set(labels)):
            strength_indices = [i for i, label in enumerate(labels) if label == strength]
            strength_entropies = [entropies[i] for i in strength_indices]
            
            fig.add_trace(
                go.Box(
                    y=strength_entropies,
                    name=f'{strength}',
                    boxpoints='outliers'
                ),
                row=1, col=2
            )
        
        # Prediction confidence
        max_probs = np.max(probabilities, axis=1)
        fig.add_trace(
            go.Histogram(
                x=max_probs,
                name='Prediction Confidence',
                nbinsx=20
            ),
            row=2, col=1
        )
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, predictions)
        labels_unique = sorted(set(labels + predictions))
        
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=labels_unique,
                y=labels_unique,
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                showscale=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Password Strength Analysis Dashboard",
            showlegend=True,
            height=800
        )
        
        return fig
    
    def plot_pattern_analysis(self, passwords: List[str], 
                            labels: List[str],
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot pattern analysis for passwords.
        
        Args:
            passwords: List of passwords
            labels: Strength labels
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        from ..utils import (
            detect_keyboard_patterns,
            detect_sequential_patterns,
            detect_repeated_patterns
        )
        
        # Analyze patterns
        pattern_data = []
        for password, label in zip(passwords, labels):
            keyboard = detect_keyboard_patterns(password)
            sequential = detect_sequential_patterns(password)
            repeated = detect_repeated_patterns(password)
            
            pattern_data.append({
                'password': anonymize_output(password, 10),
                'strength': label,
                'keyboard_patterns': len(keyboard),
                'sequential_patterns': len(sequential),
                'repeated_patterns': len(repeated),
                'total_patterns': len(keyboard) + len(sequential) + len(repeated)
            })
        
        df = pd.DataFrame(pattern_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Pattern counts by strength
        pattern_counts = df.groupby('strength')[['keyboard_patterns', 
                                               'sequential_patterns', 
                                               'repeated_patterns']].mean()
        
        pattern_counts.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Average Pattern Counts by Strength')
        axes[0, 0].set_xlabel('Strength')
        axes[0, 0].set_ylabel('Average Pattern Count')
        axes[0, 0].legend(['Keyboard', 'Sequential', 'Repeated'])
        
        # Total patterns distribution
        sns.boxplot(data=df, x='strength', y='total_patterns', ax=axes[0, 1])
        axes[0, 1].set_title('Total Pattern Count Distribution')
        axes[0, 1].set_xlabel('Strength')
        axes[0, 1].set_ylabel('Total Pattern Count')
        
        # Pattern density
        df['pattern_density'] = df['total_patterns'] / df['password'].str.len()
        sns.boxplot(data=df, x='strength', y='pattern_density', ax=axes[1, 0])
        axes[1, 0].set_title('Pattern Density by Strength')
        axes[1, 0].set_xlabel('Strength')
        axes[1, 0].set_ylabel('Pattern Density')
        
        # Pattern correlation with strength
        strength_mapping = {'weak': 0, 'moderate': 1, 'strong': 2}
        df['strength_numeric'] = df['strength'].map(strength_mapping)
        
        correlation = df['total_patterns'].corr(df['strength_numeric'])
        axes[1, 1].scatter(df['total_patterns'], df['strength_numeric'], alpha=0.6)
        axes[1, 1].set_title(f'Pattern Count vs Strength\n(Correlation: {correlation:.3f})')
        axes[1, 1].set_xlabel('Total Pattern Count')
        axes[1, 1].set_ylabel('Strength (0=weak, 1=moderate, 2=strong)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Pattern analysis plot saved to {save_path}")
        
        return fig
    
    def save_all_plots(self, evaluation_results: Dict[str, any],
                      output_dir: str) -> None:
        """Save all evaluation plots to directory.
        
        Args:
            evaluation_results: Results from model evaluation
            output_dir: Directory to save plots
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save confusion matrix
        if 'confusion_matrix' in evaluation_results['plots']:
            cm_path = os.path.join(output_dir, 'confusion_matrix.png')
            evaluation_results['plots']['confusion_matrix'].savefig(cm_path)
        
        # Save ROC curves
        if 'roc_curves' in evaluation_results['plots']:
            roc_path = os.path.join(output_dir, 'roc_curves.png')
            evaluation_results['plots']['roc_curves'].savefig(roc_path)
        
        # Save precision-recall curves
        if 'precision_recall' in evaluation_results['plots']:
            pr_path = os.path.join(output_dir, 'precision_recall.png')
            evaluation_results['plots']['precision_recall'].savefig(pr_path)
        
        self.logger.info(f"All plots saved to {output_dir}")
