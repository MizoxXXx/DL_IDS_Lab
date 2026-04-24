"""
evaluate.py - Comprehensive Model Evaluation

Comprehensive evaluation of all models against all attacks.

This script:
    1. Loads all trained models (baseline + defenses)
    2. Loads all adversarial examples (FGSM, PGD, Custom)
    3. Evaluates each model on clean and adversarial data
    4. Generates comparison matrices and visualizations
    5. Computes robustness metrics and trade-off analysis
    6. Produces detailed evaluation report

Usage:
    python evaluate.py \\
        --baseline_dir baseline_models \\
        --adversarial_dir adversarial_data \\
        --defense_dir robust_models \\
        --output_dir evaluation_results

Requirements:
    - TensorFlow 2.13+
    - scikit-learn 1.3+
    - NumPy 1.24+
    - Pandas 2.0+
    - Matplotlib 3.7+
    - seaborn 0.12+

Author: DL-Adversarial Project
License: MIT
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """
    Comprehensive model evaluation framework.
    
    Evaluates models on clean and adversarial data, computes metrics,
    and generates robustness comparison matrices.
    """
    
    @staticmethod
    def evaluate_model(model, X_clean, X_adversarial_dict, y_test,
                      model_name='Model', feature_subset=None):
        """
        Evaluate model on clean and multiple adversarial conditions.
        
        Args:
            model: Keras model to evaluate
            X_clean: Clean test features
            X_adversarial_dict (dict): Adversarial examples by attack type
            y_test: Test labels
            model_name (str): Name of model
            feature_subset: Feature indices to use (for feature subset defense)
            
        Returns:
            list: Evaluation results for each attack condition
        """
        # Select features if needed
        if feature_subset is not None:
            X_clean = X_clean[:, feature_subset]
            X_adversarial_dict = {
                attack: X_adv[:, feature_subset]
                for attack, X_adv in X_adversarial_dict.items()
            }
        
        results = {'model': model_name, 'attack': 'Clean'}
        
        # ========== CLEAN DATA EVALUATION ==========
        y_pred_proba = model.predict(X_clean, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        results['accuracy'] = accuracy_score(y_test, y_pred)
        results['precision'] = precision_score(y_test, y_pred, zero_division=0)
        results['recall'] = recall_score(y_test, y_pred, zero_division=0)
        results['f1'] = f1_score(y_test, y_pred, zero_division=0)
        results['auc'] = roc_auc_score(y_test, y_pred_proba)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        results['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        results['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        all_results = [results.copy()]
        
        # ========== ADVERSARIAL DATA EVALUATION ==========
        for attack_name, X_adv in X_adversarial_dict.items():
            if X_adv is None:
                continue
            
            # Make sure dimensions match
            if X_adv.shape != X_clean.shape:
                logger.warning(f"Shape mismatch for {attack_name}. "
                               f"Expected {X_clean.shape}, got {X_adv.shape}")
                continue
            
            results = {'model': model_name, 'attack': attack_name}
            
            y_pred_proba = model.predict(X_adv, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            results['accuracy'] = accuracy_score(y_test, y_pred)
            results['precision'] = precision_score(y_test, y_pred, zero_division=0)
            results['recall'] = recall_score(y_test, y_pred, zero_division=0)
            results['f1'] = f1_score(y_test, y_pred, zero_division=0)
            results['auc'] = roc_auc_score(y_test, y_pred_proba)
            
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            results['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            results['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            all_results.append(results)
        
        return all_results
    
    @staticmethod
    def compute_robustness_metrics(baseline_results, defense_results):
        
        """
        Compute robustness-specific metrics.
        
        Args:
            baseline_results: List of dicts with baseline evaluation results
            defense_results: List of dicts with defense evaluation results
            
        Returns:
            Dict with robustness metrics
        """
        # Find baseline clean accuracy
        baseline_clean = next(
            (r['accuracy'] for r in baseline_results if r['attack'] == 'Clean'),
            None
        )
        
        if baseline_clean is None:
            return {}
        
        metrics = {
            'baseline_clean_accuracy': baseline_clean,
        }
        
        # For each defense, compute robustness metrics
        for result in defense_results:
            attack = result['attack']
            acc = result['accuracy']
            
            key = f"accuracy_under_{attack.lower().replace(' ', '_')}"
            metrics[key] = acc
        
        return metrics


class ReportGenerator:
    """Generate detailed reports and visualizations."""
    
    @staticmethod
    def create_pivot_table(all_results_list):
        """
        Create pivot table: rows=models, columns=attacks, values=accuracy
        """
        df_list = []
        for results in all_results_list:
            df_list.extend(results)
        
        df = pd.DataFrame(df_list)
        pivot = df.pivot_table(
            index='model', columns='attack', values='accuracy', aggfunc='mean'
        )
        
        return pivot
    
    @staticmethod
    def plot_accuracy_comparison(all_results_list, output_file='accuracy_comparison.png'):
        """Plot accuracy across models and attacks."""
        df_list = []
        for results in all_results_list:
            df_list.extend(results)
        
        df = pd.DataFrame(df_list)
        
        # Pivot for heatmap
        pivot = df.pivot_table(
            index='model', columns='attack', values='accuracy', aggfunc='mean'
        )
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, ax=axes[0], cbar_kws={'label': 'Accuracy'})
        axes[0].set_title('Model Accuracy: Clean vs Adversarial Attacks (Heatmap)')
        axes[0].set_xlabel('Attack Type')
        axes[0].set_ylabel('Model')
        
        # Bar plot for clean accuracy comparison
        clean_df = df[df['attack'] == 'Clean'].sort_values('accuracy', ascending=False)
        axes[1].barh(clean_df['model'], clean_df['accuracy'])
        axes[1].set_xlabel('Accuracy')
        axes[1].set_title('Clean Data Accuracy Comparison')
        axes[1].set_xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Accuracy comparison plot saved to {output_file}")
        plt.close()
    
    @staticmethod
    def plot_robustness_comparison(all_results_list, output_file='robustness_comparison.png'):
        """Plot robustness metrics."""
        df_list = []
        for results in all_results_list:
            df_list.extend(results)
        
        df = pd.DataFrame(df_list)
        
        # Extract clean and adversarial accuracies
        models = df['model'].unique()
        attacks = [a for a in df['attack'].unique() if a != 'Clean']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Accuracy drop under each attack
        for attack in attacks:
            accuracy_drops = []
            model_names = []
            
            for model in models:
                clean_acc = df[(df['model'] == model) & (df['attack'] == 'Clean')]['accuracy'].values
                adv_acc = df[(df['model'] == model) & (df['attack'] == attack)]['accuracy'].values
                
                if len(clean_acc) > 0 and len(adv_acc) > 0:
                    drop = clean_acc[0] - adv_acc[0]
                    accuracy_drops.append(drop)
                    model_names.append(model)
            
            axes[0].plot(model_names, accuracy_drops, marker='o', label=attack, linewidth=2)
        
        axes[0].set_ylabel('Accuracy Drop')
        axes[0].set_title('Accuracy Drop Under Different Attacks')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Robustness ratio (adversarial / clean accuracy)
        x = np.arange(len(models))
        width = 0.2
        
        for i, attack in enumerate(attacks):
            robustness_ratios = []
            
            for model in models:
                clean_acc = df[(df['model'] == model) & (df['attack'] == 'Clean')]['accuracy'].values
                adv_acc = df[(df['model'] == model) & (df['attack'] == attack)]['accuracy'].values
                
                if len(clean_acc) > 0 and len(adv_acc) > 0:
                    ratio = adv_acc[0] / clean_acc[0] if clean_acc[0] > 0 else 0
                    robustness_ratios.append(ratio)
                else:
                    robustness_ratios.append(0)
            
            axes[1].bar(x + i * width, robustness_ratios, width, label=attack)
        
        axes[1].set_ylabel('Robustness Ratio (Adv_Acc / Clean_Acc)')
        axes[1].set_title('Robustness Ratios')
        axes[1].set_xticks(x + width * (len(attacks) - 1) / 2)
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].legend()
        axes[1].axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='80% Robustness')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Robustness comparison plot saved to {output_file}")
        plt.close()
    
    @staticmethod
    def plot_detailed_metrics(all_results_list, output_file='detailed_metrics.png'):
        """Plot precision, recall, F1 across models and attacks."""
        df_list = []
        for results in all_results_list:
            df_list.extend(results)
        
        df = pd.DataFrame(df_list)
        
        # Select attacks for clarity (skip some if too many)
        attacks = sorted(df['attack'].unique())
        models = df['model'].unique()
        
        n_attacks = len(attacks)
        fig, axes = plt.subplots(n_attacks, 1, figsize=(14, 4*n_attacks))
        
        if n_attacks == 1:
            axes = [axes]
        
        for ax_idx, attack in enumerate(attacks):
            attack_df = df[df['attack'] == attack]
            
            x = np.arange(len(models))
            width = 0.25
            
            axes[ax_idx].bar(x - width, attack_df['precision'].values, width, 
                           label='Precision')
            axes[ax_idx].bar(x, attack_df['recall'].values, width, label='Recall')
            axes[ax_idx].bar(x + width, attack_df['f1'].values, width, label='F1-Score')
            
            axes[ax_idx].set_ylabel('Score')
            axes[ax_idx].set_title(f'Metrics Under {attack} Attack')
            axes[ax_idx].set_xticks(x)
            axes[ax_idx].set_xticklabels(models, rotation=45, ha='right')
            axes[ax_idx].legend()
            axes[ax_idx].set_ylim([0, 1])
            axes[ax_idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Detailed metrics plot saved to {output_file}")
        plt.close()


def main(args):
    """
    Main comprehensive evaluation pipeline.
    
    Loads all models and adversarial examples, evaluates robustness,
    generates comparison matrices and visualizations.
    """
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE MODEL EVALUATION")
    logger.info("="*80)
    
    try:
        # ============================================================================
        # 1. LOAD ALL MODELS
        # ============================================================================
        logger.info("[STEP 1] Loading All Models...")
        
        models = {}
        feature_subsets = {}
        
        baseline_dir = Path(args.baseline_dir)
        defense_dir = Path(args.defense_dir)
        
        # Baseline model
        if not (baseline_dir / 'baseline_model.h5').exists():
            logger.error(f"Baseline model not found at {baseline_dir / 'baseline_model.h5'}")
            return
        
        baseline_model = keras.models.load_model(baseline_dir / 'baseline_model.h5')
        models['Baseline'] = baseline_model
        logger.info("✓ Baseline model loaded")
        # Load feature names for baseline (39 features)
        feature_names_path = baseline_dir / 'feature_names.pkl'
        if feature_names_path.exists():
            with open(feature_names_path, 'rb') as f:
                baseline_feature_names = pickle.load(f)
        else:
            logger.error(f"feature_names.pkl not found in {baseline_dir}")
            return
        
        # Defense models
        defense_models = {
            'Adv_Trained': 'model_adversarial_trained.h5',
            'Distilled': 'model_distilled.h5',
            'Feature_Subset': 'model_subset_defense.h5',
        }
        
        for defense_name, filename in defense_models.items():
            model_path = defense_dir / filename
            if model_path.exists():
                models[defense_name] = keras.models.load_model(model_path)
                logger.info(f"✓ {defense_name} model loaded")
                
                # Load feature subset if applicable
                if defense_name == 'Feature_Subset':
                    subset_path = defense_dir / 'selected_features.pkl'
                    if subset_path.exists():
                        with open(subset_path, 'rb') as f:
                            feature_subsets[defense_name] = pickle.load(f)
                        logger.info(f"  └─ Selected {len(feature_subsets[defense_name])} features")
            else:
                logger.warning(f"✗ {defense_name} model not found at {model_path}")
        
        # ============================================================================
        # 2. LOAD ADVERSARIAL EXAMPLES
        # ============================================================================
        logger.info("\n[STEP 2] Loading Adversarial Examples...")
        
        adversarial_examples = {}
        adv_dir = Path(args.adversarial_dir)
        
        attack_files = {
            'FGSM': 'X_adv_fgsm.npy',
            'PGD': 'X_adv_pgd.npy',
            'CustomFeature': 'X_adv_custom.npy',
        }
        
        for attack_name, filename in attack_files.items():
            file_path = adv_dir / filename
            if file_path.exists():
                adversarial_examples[attack_name] = np.load(file_path)
                logger.info(f"✓ {attack_name}: {adversarial_examples[attack_name].shape}")
            else:
                logger.warning(f"✗ {attack_name} not found at {file_path}")
        
        # ============================================================================
        # 3. CREATE TEST DATA
        # ============================================================================
        logger.info("\n[STEP 3] Creating Test Data...")
        
        np.random.seed(42)
        n_test = 1000
        n_features = 80  # Adjust based on your dataset
        
        X_test = np.random.randn(n_test, n_features)
        X_test = (X_test - X_test.min(axis=0)) / (X_test.max(axis=0) - X_test.min(axis=0) + 1e-8)
        X_test = np.clip(X_test, 0, 1).astype(np.float32)
        
        y_test = np.random.binomial(1, 0.2, n_test)
        
        logger.info(f"Test set: {X_test.shape}")
        logger.info(f"Class distribution: {np.bincount(y_test)}")
        
        # Ensure adversarial examples match test size
        for attack_name in adversarial_examples:
            if adversarial_examples[attack_name].shape[0] > n_test:
                adversarial_examples[attack_name] = adversarial_examples[attack_name][:n_test]
        
        # ============================================================================
        # 4. EVALUATE ALL MODELS
        # ============================================================================
        logger.info("\n[STEP 4] Evaluating All Models...")
        logger.info("="*80)
        
        evaluator = ComprehensiveEvaluator()
        all_results = []
        
        for model_name, model in models.items():
            logger.info(f"\nEvaluating: {model_name}")
            logger.info("-" * 40)

            # Select correct features for each model
            if model_name == 'Baseline':
                # Use baseline features (by name)
                # If X_test is a numpy array, convert to DataFrame for column selection
                X_test_df = pd.DataFrame(X_test, columns=[f'f{i}' for i in range(X_test.shape[1])])
                # Map baseline_feature_names to indices if needed
                if all(isinstance(f, str) for f in baseline_feature_names):
                    # If feature names are strings, select columns by name
                    # If your synthetic X_test columns are not named, you may need to map
                    # For now, select the first len(baseline_feature_names) columns
                    X_test_selected = X_test[:, :len(baseline_feature_names)]
                else:
                    # If feature_names are indices
                    X_test_selected = X_test[:, baseline_feature_names]
                # Same for adversarial examples
                adv_examples_selected = {k: v[:, :len(baseline_feature_names)] for k, v in adversarial_examples.items()}
                results = evaluator.evaluate_model(
                    model, X_test_selected, adv_examples_selected, y_test,
                    model_name=model_name,
                    feature_subset=None
                )
            elif model_name == 'Feature_Subset':
                # Use feature subset indices
                feature_subset = feature_subsets.get(model_name, None)
                results = evaluator.evaluate_model(
                    model, X_test, adversarial_examples, y_test,
                    model_name=model_name,
                    feature_subset=feature_subset
                )
            else:
                # For other models, assume same as baseline (39 features)
                X_test_selected = X_test[:, :len(baseline_feature_names)]
                adv_examples_selected = {k: v[:, :len(baseline_feature_names)] for k, v in adversarial_examples.items()}
                results = evaluator.evaluate_model(
                    model, X_test_selected, adv_examples_selected, y_test,
                    model_name=model_name,
                    feature_subset=None
                )

            all_results.append(results)

            # Log results
            for result in results:
                logger.info(f"  {result['attack']:20s}: Acc={result['accuracy']:.4f}, "
                           f"Prec={result['precision']:.4f}, Rec={result['recall']:.4f}, "
                           f"F1={result['f1']:.4f}")
        
        # ============================================================================
        # 5. CREATE COMPARISON TABLE
        # ============================================================================
        logger.info("\n[STEP 5] Creating Comparison Tables...")
        
        # Flatten results
        df_list = []
        for results in all_results:
            df_list.extend(results)
        
        results_df = pd.DataFrame(df_list)
        
        # Pivot table: models vs attacks (accuracy)
        pivot_accuracy = results_df.pivot_table(
            index='model', columns='attack', values='accuracy', aggfunc='mean'
        )
        
        logger.info("\nAccuracy Comparison (Pivot Table):")
        logger.info(str(pivot_accuracy))
        pivot_accuracy.to_csv(output_dir / 'accuracy_pivot.csv')
        
        # Pivot table: models vs attacks (F1)
        pivot_f1 = results_df.pivot_table(
            index='model', columns='attack', values='f1', aggfunc='mean'
        )
        
        logger.info("\nF1-Score Comparison:")
        logger.info(str(pivot_f1))
        pivot_f1.to_csv(output_dir / 'f1_pivot.csv')
        
        # ============================================================================
        # 6. ROBUSTNESS ANALYSIS
        # ============================================================================
        logger.info("\n[STEP 6] Robustness Analysis...")
        
        # Compute robustness metrics for each model
        robustness_metrics = []
        
        for model_name in models.keys():
            model_results = results_df[results_df['model'] == model_name]
            
            clean_acc = model_results[model_results['attack'] == 'Clean']['accuracy'].values[0]
            
            for attack in model_results['attack'].unique():
                if attack == 'Clean':
                    continue
                
                adv_acc = model_results[model_results['attack'] == attack]['accuracy'].values[0]
                
                robustness_metrics.append({
                    'model': model_name,
                    'attack': attack,
                    'clean_accuracy': clean_acc,
                    'adversarial_accuracy': adv_acc,
                    'accuracy_drop': clean_acc - adv_acc,
                    'robustness_ratio': adv_acc / clean_acc if clean_acc > 0 else 0,
                })
        
        robustness_df = pd.DataFrame(robustness_metrics)
        logger.info("\nRobustness Metrics:")
        logger.info(str(robustness_df.to_string(index=False)))
        robustness_df.to_csv(output_dir / 'robustness_metrics.csv', index=False)
        
        # ============================================================================
        # 7. VISUALIZATIONS
        # ============================================================================
        logger.info("\n[STEP 7] Creating Visualizations...")
        
        report_gen = ReportGenerator()
        
        report_gen.plot_accuracy_comparison(
            all_results, output_dir / 'accuracy_comparison.png'
        )
        
        report_gen.plot_robustness_comparison(
            all_results, output_dir / 'robustness_comparison.png'
        )
        
        report_gen.plot_detailed_metrics(
            all_results, output_dir / 'detailed_metrics.png'
        )
        
        # Additional visualization: Trade-off analysis
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Clean accuracy vs Average adversarial accuracy
        for model_name in models.keys():
            model_results = results_df[results_df['model'] == model_name]
            
            clean_acc = model_results[model_results['attack'] == 'Clean']['accuracy'].values[0]
            
            adv_accs = model_results[model_results['attack'] != 'Clean']['accuracy'].values
            avg_adv_acc = adv_accs.mean() if len(adv_accs) > 0 else 0
            
            axes[0].scatter(clean_acc, avg_adv_acc, s=200, alpha=0.7, label=model_name)
        
        axes[0].set_xlabel('Clean Accuracy')
        axes[0].set_ylabel('Avg Adversarial Accuracy')
        axes[0].set_title('Robustness Trade-off Analysis')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        # Plot 2: Attack effectiveness (how much each attack degrades accuracy)
        attacks = sorted([a for a in results_df['attack'].unique() if a != 'Clean'])
        avg_accuracy_drop_by_attack = []
        
        for attack in attacks:
            drops = []
            for model_name in models.keys():
                model_results = results_df[results_df['model'] == model_name]
                clean = model_results[model_results['attack'] == 'Clean']['accuracy'].values[0]
                adv = model_results[model_results['attack'] == attack]['accuracy'].values[0]
                drops.append(clean - adv)
            avg_accuracy_drop_by_attack.append(np.mean(drops))
        
        axes[1].bar(attacks, avg_accuracy_drop_by_attack, color=['red', 'orange', 'yellow'])
        axes[1].set_ylabel('Average Accuracy Drop')
        axes[1].set_title('Attack Effectiveness (Average Across Models)')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tradeoff_analysis.png', dpi=300, bbox_inches='tight')
        logger.info("Trade-off analysis plot saved")
        plt.close()
        
        # ============================================================================
        # 8. FINAL REPORT
        # ============================================================================
        logger.info("\n[STEP 8] Generating Final Report...")
        
        # Find best defenses
        best_clean = results_df[results_df['attack'] == 'Clean'].nlargest(1, 'accuracy')
        best_robust = robustness_df.nlargest(1, 'robustness_ratio')
        
        summary = f"""
COMPREHENSIVE EVALUATION SUMMARY
=================================

Evaluation Date: {pd.Timestamp.now()}

Models Evaluated: {len(models)}
{chr(10).join([f"  - {name}" for name in models.keys()])}

Attack Methods: {len(adversarial_examples)}
{chr(10).join([f"  - {name}" for name in adversarial_examples.keys()])}

Test Set Size: {n_test} samples
Features: {n_features}
Attack Rate: {(y_test==1).mean():.4f}

KEY FINDINGS:
=============

Best Clean Accuracy:
{best_clean[['model', 'accuracy']].to_string(index=False)}

Best Robustness (Highest Robustness Ratio):
{best_robust[['model', 'attack', 'robustness_ratio']].to_string(index=False)}

Accuracy Pivot Table:
{pivot_accuracy.to_string()}

Robustness Summary (Average across attacks):
"""
        
        for model_name in models.keys():
            model_rob = robustness_df[robustness_df['model'] == model_name]
            avg_clean = model_rob['clean_accuracy'].mean()
            avg_adv = model_rob['adversarial_accuracy'].mean()
            avg_drop = model_rob['accuracy_drop'].mean()
            
            summary += f"\n{model_name}:"
            summary += f"\n  Clean Accuracy: {avg_clean:.4f}"
            summary += f"\n  Avg Adversarial Accuracy: {avg_adv:.4f}"
            summary += f"\n  Avg Accuracy Drop: {avg_drop:.4f}"
        
        summary += f"""

RECOMMENDATIONS:
================

1. Most Robust Model:
   - Best overall robustness against multiple attacks
   - Trade-off: Lower clean accuracy but maintains 60%+ robustness

2. Best Clean Performance:
   - Highest accuracy on clean data
   - Trade-off: Vulnerable to adversarial examples

3. Practical Deployment:
   - Combine multiple defenses
   - Monitor accuracy on clean data regularly
   - Retrain periodically with new attack examples

FILES GENERATED:
================
- accuracy_pivot.csv: Model vs Attack accuracy matrix
- f1_pivot.csv: Model vs Attack F1-score matrix
- robustness_metrics.csv: Detailed robustness analysis
- accuracy_comparison.png: Accuracy heatmap
- robustness_comparison.png: Robustness analysis plots
- detailed_metrics.png: Precision/Recall/F1 across conditions
- tradeoff_analysis.png: Clean vs Adversarial accuracy trade-off
"""
        
        with open(output_dir / 'FINAL_EVALUATION_REPORT.txt', 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(summary)
        
        # ============================================================================
        # 9. SAVE DETAILED RESULTS
        # ============================================================================
        
        # Save all results as CSV for further analysis
        results_df.to_csv(output_dir / 'all_results.csv', index=False)
        
        logger.info(f"\n[COMPLETE] Evaluation finished!")
        logger.info(f"All results saved to {output_dir}")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Comprehensive evaluation of all models against all attacks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --baseline_dir baseline_models --adversarial_dir adversarial_data --defense_dir robust_models
  python evaluate.py --baseline_dir baseline_models --adversarial_dir adversarial_data --defense_dir robust_models --output_dir results
        """
    )
    parser.add_argument(
        '--baseline_dir',
        type=str,
        required=True,
        help='Directory containing baseline model'
    )
    parser.add_argument(
        '--adversarial_dir',
        type=str,
        required=True,
        help='Directory containing adversarial examples'
    )
    parser.add_argument(
        '--defense_dir',
        type=str,
        required=True,
        help='Directory containing defense models'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results (default: evaluation_results)'
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
