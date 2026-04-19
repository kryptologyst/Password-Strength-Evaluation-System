#!/usr/bin/env python3
"""Main training script for password strength evaluation system."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data import SyntheticPasswordGenerator
from src.eval import PasswordStrengthEvaluator
from src.models import EntropyBasedModel
from src.utils import (
    load_config,
    save_config,
    set_deterministic_seed,
    setup_logging,
)
from src.viz import PasswordStrengthVisualizer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train password strength evaluation model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                      help="Output directory for results")
    parser.add_argument("--data_size", type=int, default=10000,
                      help="Size of synthetic dataset to generate")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set random seed
    set_deterministic_seed(config.seed)
    logger.info(f"Set random seed to {config.seed}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    logger.info("Generating synthetic password dataset")
    data_generator = SyntheticPasswordGenerator(config.data)
    
    # Generate main dataset
    main_dataset = data_generator.generate_dataset(args.data_size)
    logger.info(f"Generated {len(main_dataset)} passwords")
    
    # Generate breach dataset
    breach_dataset = data_generator.generate_breach_dataset(args.data_size // 10)
    logger.info(f"Generated {len(breach_dataset)} breached passwords")
    
    # Split datasets
    train_df, val_df, test_df = data_generator.split_dataset(
        main_dataset, 
        config.data.train_split, 
        config.data.val_split
    )
    
    logger.info(f"Dataset splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Save datasets
    train_df.to_csv(output_dir / "train_data.csv", index=False)
    val_df.to_csv(output_dir / "val_data.csv", index=False)
    test_df.to_csv(output_dir / "test_data.csv", index=False)
    breach_dataset.to_csv(output_dir / "breach_data.csv", index=False)
    
    # Initialize model
    logger.info("Initializing entropy-based model")
    model = EntropyBasedModel(config.model)
    
    # Train model
    logger.info("Training model")
    model.fit(train_df, train_df['strength_label'])
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set")
    val_predictions = model.predict(val_df)
    val_accuracy = model.score(val_df, val_df['strength_label'])
    logger.info(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Initialize evaluator
    evaluator = PasswordStrengthEvaluator(config.evaluation)
    
    # Comprehensive evaluation on test set
    logger.info("Running comprehensive evaluation")
    evaluation_results = evaluator.evaluate_model(model, test_df, test_df['strength_label'])
    
    # Print key metrics
    logger.info("Evaluation Results:")
    logger.info(f"Test Accuracy: {evaluation_results['classification_metrics']['accuracy']:.4f}")
    logger.info(f"F1 Macro: {evaluation_results['classification_metrics']['f1_macro']:.4f}")
    logger.info(f"Entropy Correlation: {evaluation_results['security_metrics']['entropy_correlation']:.4f}")
    
    # Generate report
    report = evaluator.generate_report(evaluation_results, output_dir / "evaluation_report.txt")
    logger.info("Generated evaluation report")
    
    # Save model
    model.save_model(output_dir / "trained_model.pkl")
    logger.info("Saved trained model")
    
    # Generate visualizations
    logger.info("Generating visualizations")
    visualizer = PasswordStrengthVisualizer()
    
    # Plot password distribution
    dist_plot = visualizer.plot_password_distribution(
        test_df['password'].tolist(),
        test_df['strength_label'].tolist(),
        output_dir / "password_distribution.png"
    )
    
    # Plot confusion matrix
    cm_plot = visualizer.plot_confusion_matrix(
        test_df['strength_label'].tolist(),
        evaluation_results['predictions'],
        output_dir / "confusion_matrix.png"
    )
    
    # Plot ROC curves
    roc_plot = visualizer.plot_roc_curves(
        test_df['strength_label'].tolist(),
        evaluation_results['probabilities'],
        output_dir / "roc_curves.png"
    )
    
    # Plot feature importance
    feature_importance = model.get_feature_importance()
    fi_plot = visualizer.plot_feature_importance(
        feature_importance,
        save_path=output_dir / "feature_importance.png"
    )
    
    # Create interactive dashboard
    dashboard = visualizer.create_interactive_dashboard(
        test_df['password'].tolist(),
        test_df['strength_label'].tolist(),
        evaluation_results['predictions'],
        evaluation_results['probabilities']
    )
    
    # Save dashboard
    dashboard.write_html(output_dir / "interactive_dashboard.html")
    logger.info("Saved interactive dashboard")
    
    # Save feature importance data
    fi_df = pd.DataFrame(list(feature_importance.items()), 
                         columns=['feature', 'importance'])
    fi_df.to_csv(output_dir / "feature_importance.csv", index=False)
    
    # Create leaderboard
    leaderboard = evaluator.create_leaderboard([evaluation_results])
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    logger.info("Created leaderboard")
    
    # Save configuration
    save_config(config, output_dir / "config.yaml")
    
    logger.info(f"Training completed. Results saved to {output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Dataset size: {args.data_size}")
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"Test Accuracy: {evaluation_results['classification_metrics']['accuracy']:.4f}")
    print(f"F1 Macro: {evaluation_results['classification_metrics']['f1_macro']:.4f}")
    print(f"Entropy Correlation: {evaluation_results['security_metrics']['entropy_correlation']:.4f}")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
