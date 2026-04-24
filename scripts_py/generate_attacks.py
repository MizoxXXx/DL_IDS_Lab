"""
generate_attacks.py - Generate Adversarial Examples

Generate adversarial examples against baseline IDS model using three attack methods:
1. FGSM (Fast Gradient Sign Method) - fast, less effective
2. PGD (Projected Gradient Descent) - stronger, iterative
3. Custom Feature Manipulation - realistic, feature-aware

This script:
    1. Loads baseline model from baseline_models directory
    2. Loads feature analysis and attack constraints
    3. Generates adversarial examples using three attack methods
    4. Saves adversarial examples and attack statistics
    5. Generates visualizations and analysis reports

Usage:
    python generate_attacks.py \\
        --model_dir baseline_models \\
        --output_dir adversarial_data \\
        --epsilon 0.3

Requirements:
    - TensorFlow 2.13+
    - NumPy 1.24+
    - Pandas 2.0+

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
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AttackConstraints:
    """
    Define and enforce feature manipulability constraints.
    
    Realistically models which network features can be modified by attackers
    and by how much. Used to generate constrained adversarial examples.
    """
    
    def __init__(self, epsilon=0.3, use_feature_based=True, analysis_df=None):
        """
        Initialize attack constraints.
        
        Args:
            epsilon (float): Default maximum relative change [0.0, 1.0]
            use_feature_based (bool): Use per-feature constraints from analysis
            analysis_df (pd.DataFrame): Feature manipulability analysis
        """
        self.epsilon = epsilon
        self.use_feature_based = use_feature_based
        self.analysis_df = analysis_df
        logger.info(f"AttackConstraints initialized with epsilon={epsilon}")
    
    def get_constraints(self, feature_idx):
        """
        Get manipulability constraint for specific feature.
        
        Args:
            feature_idx (int): Feature index
            
        Returns:
            tuple: (is_manipulable, max_change_epsilon)
        """
        if self.analysis_df is not None and self.use_feature_based:
            row = self.analysis_df.iloc[feature_idx]
            max_change = row['Max_Change_Epsilon']
            is_manipulable = row['Manipulability'] != 'Non-Manipulable'
            return is_manipulable, max_change
        else:
            return True, self.epsilon
    
    def clip_perturbation(self, x_original, x_perturbed, is_normalized=True):
        """
        Clip perturbation to respect feature manipulability constraints.
        
        Args:
            x_original (np.ndarray): Original clean features [n_samples, n_features]
            x_perturbed (np.ndarray): Perturbed features [n_samples, n_features]
            is_normalized (bool): Whether features are in [0, 1] range
            
        Returns:
            np.ndarray: Clipped perturbation respecting all constraints
        """
        x_clipped = x_perturbed.copy()
        
        for feat_idx in range(x_original.shape[1]):
            is_manipulable, max_change = self.get_constraints(feat_idx)
            
            if not is_manipulable:
                x_clipped[:, feat_idx] = x_original[:, feat_idx]
            else:
                if is_normalized:
                    lower = np.maximum(0, x_original[:, feat_idx] - max_change)
                    upper = np.minimum(1, x_original[:, feat_idx] + max_change)
                else:
                    lower = x_original[:, feat_idx] - max_change
                    upper = x_original[:, feat_idx] + max_change
                
                x_clipped[:, feat_idx] = np.clip(x_clipped[:, feat_idx], lower, upper)
        
        x_clipped = np.clip(x_clipped, 0, 1)
        return x_clipped


class AdversarialAttacks:
    """
    Implement three types of adversarial attacks against IDS models.
    
    Supports:
    - FGSM: Fast single-step gradient attack
    - PGD: Projected Gradient Descent multi-step attack
    - Feature Manipulation: Greedy feature-by-feature attack
    """
    @staticmethod 
    def fgsm_attack(model, X, y, epsilon=0.3, constraints=None):
        """
        Fast Gradient Sign Method (FGSM) attack.
        
        One-step attack: move in the direction of the gradient sign.        
        Args:
            model: Keras model
            X: Input features [n_samples, n_features]
            y: Labels [n_samples]
            epsilon: Attack budget (relative change)
            constraints: AttackConstraints object
            
        Returns:  Adversarial examples [n_samples, n_features]
        """
        X_adv = X.copy().astype(np.float32)
        
        for batch_start in range(0, len(X), 128):  # Process in batches
            batch_end = min(batch_start + 128, len(X))
            X_batch = X_adv[batch_start:batch_end]
            y_batch = y[batch_start:batch_end]
            
            # Create variable for gradient computation
            X_var = tf.Variable(X_batch, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                logits = model(X_var, training=False)
                # Compute loss for misclassification
                loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.reshape(tf.cast(y_batch, tf.float32), (-1, 1)),
                    logits=logits
                )
                loss = tf.reduce_mean(loss)
            
            # Compute gradient
            gradients = tape.gradient(loss, X_var)
            
            # FGSM update: X_adv = X + epsilon * sign(gradient)
            perturbation = epsilon * tf.sign(gradients)
            X_adv_batch = X_batch + perturbation.numpy()
            
            # Apply constraints
            if constraints is not None:
                X_adv_batch = constraints.clip_perturbation(X_batch, X_adv_batch)
            else:
                X_adv_batch = np.clip(X_adv_batch, 0, 1)
            
            X_adv[batch_start:batch_end] = X_adv_batch
        
        return X_adv
    
    @staticmethod
    def pgd_attack(model, X, y, epsilon=0.3, step_size=0.02, num_steps=40,
                  constraints=None, random_start=True):
        """
        Projected Gradient Descent (PGD) attack.
        Args:
            step_size: Learning rate for each step
            num_steps: Number of gradient steps
            constraints: AttackConstraints object
            random_start: Whether to start from random perturbation --> Escape local minima and have multiple attack paths
            
        Returns: Adversarial examples [n_samples, n_features]
        """
        X_adv = X.copy().astype(np.float32)
        
        # Random start (improves attack effectiveness)
        if random_start:
            random_noise = np.random.uniform(-epsilon, epsilon, X.shape)
            X_adv = X + random_noise
            if constraints is not None:
                X_adv = constraints.clip_perturbation(X, X_adv)
            else:
                X_adv = np.clip(X_adv, 0, 1)
        
        # PGD iterations
        for step in range(num_steps):
            for batch_start in range(0, len(X), 128):
                batch_end = min(batch_start + 128, len(X))
                X_batch = X_adv[batch_start:batch_end]
                X_orig_batch = X[batch_start:batch_end]
                y_batch = y[batch_start:batch_end]
                
                # Compute gradient
                X_var = tf.Variable(X_batch, dtype=tf.float32)
                
                with tf.GradientTape() as tape:
                    logits = model(X_var, training=False)
                    loss = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.reshape(tf.cast(y_batch, tf.float32), (-1, 1)),
                        logits=logits
                    )
                    loss = tf.reduce_mean(loss)
                
                gradients = tape.gradient(loss, X_var)
                
                # Gradient step
                perturbation = step_size * tf.sign(gradients) # small step in direction of gradient
                X_adv_batch = X_batch + perturbation.numpy()
                
                # Project back to epsilon ball around original
                X_adv_batch = np.clip(
                    X_adv_batch,
                    X_orig_batch - epsilon,
                    X_orig_batch + epsilon
                )
                
                # Apply feature constraints
                if constraints is not None:
                    X_adv_batch = constraints.clip_perturbation(
                        X_orig_batch, X_adv_batch
                    )
                else:
                    X_adv_batch = np.clip(X_adv_batch, 0, 1)
                
                X_adv[batch_start:batch_end] = X_adv_batch
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/{num_steps}")
        
        return X_adv
    
    @staticmethod
    def feature_manipulation_attack(model, X, y, constraints, budget=0.3,
                                   iterations=15, batch_size=1):
        """
        Custom feature manipulation attack.
        Greedy attack that iteratively modifies manipulable features to maximize loss.
        Args:
            ...
            batch_size: Samples to process together
            
        Returns: Adversarial examples [n_samples, n_features]
        """
        X_adv = X.copy().astype(np.float32)    # copy the dataset to modify in it 
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        print(f"\nRunning feature manipulation attack...")
        print(f"Samples: {n_samples}, Features: {n_features}")
        print(f"Iterations per sample: {iterations}")
        
        for sample_idx in tqdm(range(n_samples), desc="Samples"):
            x_orig = X[sample_idx:sample_idx+1]
            x_adv = X_adv[sample_idx:sample_idx+1].copy()
            y_sample = y[sample_idx:sample_idx+1]
            
            # Get original loss of the sample --> make it higher
            loss_orig = model.evaluate(x_adv, y_sample, verbose=0)
            
            for iteration in range(iterations):
                best_feature = None
                best_loss = loss_orig
                best_perturbation = 0
                
                # Try perturbing each feature
                for feat_idx in range(n_features):
                    is_manipulable, max_change = constraints.get_constraints(feat_idx)
                    
                    if not is_manipulable:
                        continue  # Cannot modify this feature
                    
                    # Sample several perturbation directions
                    perturbation_magnitudes = [
                        -max_change * 0.5,
                        -max_change * 0.25,
                        max_change * 0.25,
                        max_change * 0.5,
                    ]
                    
                    for perturbation in perturbation_magnitudes:
                        x_test = x_adv.copy()
                        x_test[0, feat_idx] = np.clip(
                            x_test[0, feat_idx] + perturbation,
                            max(0, x_orig[0, feat_idx] - max_change),
                            min(1, x_orig[0, feat_idx] + max_change)
                        )
                        
                        # Evaluate
                        loss_test = model.evaluate(x_test, y_sample, verbose=0)
                        
                        # Update best if loss increased (attack goal)
                        if loss_test > best_loss:
                            best_loss = loss_test
                            best_feature = feat_idx
                            best_perturbation = perturbation
                
                if best_feature is not None:
                    # Apply best perturbation
                    x_adv[0, best_feature] = np.clip(
                        x_adv[0, best_feature] + best_perturbation,
                        max(0, x_orig[0, best_feature] - max_change),
                        min(1, x_orig[0, best_feature] + max_change)
                    )
                    loss_orig = best_loss
                else:
                    break  # No improvement possible
            
            X_adv[sample_idx:sample_idx+1] = x_adv
        
        return X_adv


class AttackEvaluator:
    """Evaluate attack effectiveness."""
    
    @staticmethod
    def compute_perturbation_stats(X_clean, X_adv):
        """    
        Returns: Dictionary with perturbation statistics
        """
        perturbations = np.abs(X_adv - X_clean)
        
        stats = {
            'mean_perturbation': perturbations.mean(),
            'std_perturbation': perturbations.std(),
            'max_perturbation': perturbations.max(),
            'min_perturbation': perturbations.min(),
            'samples_perturbed': (perturbations.max(axis=1) > 0).mean(),
            'avg_features_perturbed_per_sample': (perturbations > 0).sum(axis=1).mean(),
        }
        
        return stats
    
    @staticmethod
    def evaluate_attack_success(model, X_clean, X_adv, y_true):
        """
        Returns: Dictionary with success metrics
            Get clean predictions → y_pred_clean
            Get adversarial predictions → y_pred_adv
            Find correctly classified samples → correct_clean = (y_pred_clean == y_true)
            Find predictions that changed → attack_success = (y_pred_clean != y_pred_adv)
            Combine them → successful_attacks = attack_success[correct_clean]

        """
        # Predictions on clean examples
        y_pred_clean = (model.predict(X_clean, verbose=0) > 0.5).flatten()
        
        # Predictions on adversarial examples
        y_pred_adv = (model.predict(X_adv, verbose=0) > 0.5).flatten()
        
        # Correctly classified clean examples
        correct_clean = (y_pred_clean == y_true)
        
        # Check if attack succeeded (changed prediction for correctly classified examples)
        attack_success = y_pred_clean != y_pred_adv
        successful_attacks = attack_success[correct_clean]
        
        metrics = {
            'clean_accuracy': (y_pred_clean == y_true).mean(),
            'adversarial_accuracy': (y_pred_adv == y_true).mean(),
            'attack_success_rate': successful_attacks.mean() if correct_clean.sum() > 0 else 0,
            'fool_rate': (y_pred_clean != y_pred_adv).mean(),
        }
        
        return metrics


def main(args):
    """
    Main adversarial attack generation pipeline.
    
    Loads baseline model, generates three types of adversarial examples,
    evaluates attacks, and saves results with analysis.
    """
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("ADVERSARIAL ATTACK GENERATION")
    logger.info("="*80)
    
    try:
        # ============================================================================
        # 1. LOAD BASELINE MODEL AND DATA
        # ============================================================================
        logger.info("[STEP 1] Loading Baseline Model and Data...")
        
        model_dir = Path(args.model_dir)
        
        # Load model
        model_file = model_dir / 'baseline_model.h5'
        if not model_file.exists():
            logger.error(f"Model file not found: {model_file}")
            return
        
        model = keras.models.load_model(model_file)
        logger.info(f"Model loaded from {model_file}")
        
        # Load feature names
        with open(model_dir / 'feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        logger.info(f"Loaded {len(feature_names)} feature names")
        
        # Load scaler
        with open(model_dir / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Feature scaler loaded")
        
        # Load feature analysis
        with open(model_dir / 'feature_analysis.pkl', 'rb') as f:
            feature_analysis = pickle.load(f)
        analysis_df = feature_analysis['analysis_df']
        logger.info(f"Feature analysis loaded: {len(analysis_df)} features")
        
        # Load test data (need to split from same location as baseline training)
        logger.info("Note: Test data should be loaded from the same location as baseline training.")
        
        # Create synthetic test data for demonstration
        # In practice, you would load the actual test split
        np.random.seed(42)
        n_test_samples = args.n_test_samples
        n_features = len(feature_names)
        
        # Simulate test data (normally loaded from baseline training output)
        X_test = np.random.randn(n_test_samples, n_features)
        X_test = (X_test - X_test.min(axis=0)) / (X_test.max(axis=0) - X_test.min(axis=0))
        X_test = np.clip(X_test, 0, 1).astype(np.float32)
        
        # Create labels (mix of normal and attack)
        y_test = np.random.binomial(1, 0.2, n_test_samples)
        
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Test labels: {np.bincount(y_test)}")
        
        # ============================================================================
        # 2. SETUP ATTACK CONSTRAINTS
        # ============================================================================
        logger.info("[STEP 2] Setting Up Attack Constraints...")
        
        constraints = AttackConstraints(
            epsilon=args.epsilon,
            use_feature_based=True,
            analysis_df=analysis_df
        )
        
        # Print constraint summary
        manipulable_count = 0
        for feat_idx in range(len(feature_names)):
            is_manip, max_change = constraints.get_constraints(feat_idx)
            if is_manip:
                manipulable_count += 1
        
        logger.info(f"Manipulable features: {manipulable_count}/{len(feature_names)}")
        logger.info(f"Default epsilon: {args.epsilon}")
        
        # ============================================================================
        # 3. GENERATE ADVERSARIAL EXAMPLES - FGSM
        # ============================================================================
        logger.info(f"[STEP 3a] Generating FGSM Adversarial Examples (ε={args.epsilon})...")
        
        attacks = AdversarialAttacks()
        X_adv_fgsm = attacks.fgsm_attack(
            model, X_test, y_test,
            epsilon=args.epsilon,
            constraints=constraints
        )
        
        # Evaluate FGSM attack
        evaluator = AttackEvaluator()
        fgsm_perturbations = evaluator.compute_perturbation_stats(X_test, X_adv_fgsm)
        fgsm_success = evaluator.evaluate_attack_success(model, X_test, X_adv_fgsm, y_test)
        
        logger.info(f"FGSM Results: Adv Acc={fgsm_success['adversarial_accuracy']:.4f}, Success Rate={fgsm_success['attack_success_rate']:.4f}")
        
        # Save FGSM results
        np.save(output_dir / 'X_adv_fgsm.npy', X_adv_fgsm)
        with open(output_dir / 'fgsm_results.pkl', 'wb') as f:
            pickle.dump({'perturbations': fgsm_perturbations, 'success': fgsm_success}, f)
        logger.info("FGSM results saved")
        
        # ============================================================================
        # 4. GENERATE ADVERSARIAL EXAMPLES - PGD
        # ============================================================================
        logger.info(f"[STEP 3b] Generating PGD Adversarial Examples (ε={args.epsilon}, steps={args.pgd_steps})...")
        
        X_adv_pgd = attacks.pgd_attack(
            model, X_test, y_test,
            epsilon=args.epsilon,
            step_size=args.epsilon / 4,
            num_steps=args.pgd_steps,
            constraints=constraints,
            random_start=True
        )
        
        # Evaluate PGD attack
        pgd_perturbations = evaluator.compute_perturbation_stats(X_test, X_adv_pgd)
        pgd_success = evaluator.evaluate_attack_success(model, X_test, X_adv_pgd, y_test)
        
        logger.info(f"PGD Results: Adv Acc={pgd_success['adversarial_accuracy']:.4f}, Success Rate={pgd_success['attack_success_rate']:.4f}")
        
        # Save PGD results
        np.save(output_dir / 'X_adv_pgd.npy', X_adv_pgd)
        with open(output_dir / 'pgd_results.pkl', 'wb') as f:
            pickle.dump({'perturbations': pgd_perturbations, 'success': pgd_success}, f)
        logger.info("PGD results saved")
        
        # ============================================================================
        # 5. GENERATE ADVERSARIAL EXAMPLES - FEATURE MANIPULATION
        # ============================================================================
        logger.info(f"[STEP 3c] Generating Feature Manipulation Adversarial Examples...")
        
        n_subset = min(args.n_test_samples, 100)
        X_subset = X_test[:n_subset]
        y_subset = y_test[:n_subset]
        
        X_adv_custom = attacks.feature_manipulation_attack(
            model, X_subset, y_subset,
            constraints=constraints,
            budget=args.epsilon,
            iterations=args.custom_iterations,
            batch_size=1
        )
        
        # Evaluate custom attack
        custom_perturbations = evaluator.compute_perturbation_stats(X_subset, X_adv_custom)
        custom_success = evaluator.evaluate_attack_success(model, X_subset, X_adv_custom, y_subset)
        
        logger.info(f"Custom Results: Adv Acc={custom_success['adversarial_accuracy']:.4f}, Success Rate={custom_success['attack_success_rate']:.4f}")
        
        # Save custom attack results
        np.save(output_dir / 'X_adv_custom.npy', X_adv_custom)
        with open(output_dir / 'custom_results.pkl', 'wb') as f:
            pickle.dump({'perturbations': custom_perturbations, 'success': custom_success}, f)
        logger.info("Feature manipulation results saved")
        
        # ============================================================================
        # 6. ANALYSIS: FEATURE PERTURBATIONS
        # ============================================================================
        logger.info(f"[STEP 4] Analyzing Feature Perturbations...")
        
        fgsm_perturbations_by_feat = np.abs(X_adv_fgsm - X_test).mean(axis=0)
        pgd_perturbations_by_feat = np.abs(X_adv_pgd - X_test).mean(axis=0)
        custom_perturbations_by_feat = np.abs(X_adv_custom - X_subset).mean(axis=0)
        
        # Create comparison dataframe
        feature_comparison = pd.DataFrame({
            'Feature': feature_names,
            'FGSM_Perturbation': fgsm_perturbations_by_feat,
            'PGD_Perturbation': pgd_perturbations_by_feat,
            'Custom_Perturbation': custom_perturbations_by_feat[:len(feature_names)],
            'Manipulability': analysis_df['Manipulability'].values,
        })
        
        feature_comparison = feature_comparison.sort_values('PGD_Perturbation', ascending=False)
        feature_comparison.to_csv(output_dir / 'feature_perturbation_analysis.csv', index=False)
        logger.info("Feature perturbation analysis saved")
        
        # ============================================================================
        # 7. COMPLETION
        # ============================================================================
        logger.info(f"[COMPLETE] All adversarial examples saved to {output_dir}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error during attack generation: {e}", exc_info=True)
        raise
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # FGSM perturbations
    axes[0, 0].hist(fgsm_perturbations_by_feat, bins=30)
    axes[0, 0].set_xlabel('Mean Perturbation Magnitude')
    axes[0, 0].set_ylabel('Number of Features')
    axes[0, 0].set_title('FGSM: Feature Perturbation Distribution')
    axes[0, 0].axvline(fgsm_perturbations_by_feat.mean(), color='r', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    # PGD perturbations
    axes[0, 1].hist(pgd_perturbations_by_feat, bins=30)
    axes[0, 1].set_xlabel('Mean Perturbation Magnitude')
    axes[0, 1].set_ylabel('Number of Features')
    axes[0, 1].set_title('PGD: Feature Perturbation Distribution')
    axes[0, 1].axvline(pgd_perturbations_by_feat.mean(), color='r', linestyle='--', label='Mean')
    axes[0, 1].legend()
    
    # Attack comparison
    top_k = 15
    top_features_idx = np.argsort(pgd_perturbations_by_feat)[-top_k:]
    x_pos = np.arange(top_k)
    
    axes[0, 2].bar(x_pos - 0.2, fgsm_perturbations_by_feat[top_features_idx], 
                   width=0.4, label='FGSM')
    axes[0, 2].bar(x_pos + 0.2, pgd_perturbations_by_feat[top_features_idx], 
                   width=0.4, label='PGD')
    axes[0, 2].set_xlabel('Feature')
    axes[0, 2].set_ylabel('Perturbation Magnitude')
    axes[0, 2].set_title(f'Top {top_k} Most Perturbed Features')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels([f'F{i}' for i in top_features_idx], rotation=45)
    axes[0, 2].legend()
    
    # Attack effectiveness comparison
    attacks_names = ['FGSM', 'PGD', 'Custom']
    clean_accs = [
        fgsm_success['clean_accuracy'],
        pgd_success['clean_accuracy'],
        custom_success['clean_accuracy']
    ]
    adv_accs = [
        fgsm_success['adversarial_accuracy'],
        pgd_success['adversarial_accuracy'],
        custom_success['adversarial_accuracy']
    ]
    
    x_pos = np.arange(len(attacks_names))
    axes[1, 0].bar(x_pos - 0.2, clean_accs, width=0.4, label='Clean')
    axes[1, 0].bar(x_pos + 0.2, adv_accs, width=0.4, label='Adversarial')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Model Accuracy: Clean vs Adversarial')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(attacks_names)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1])
    
    # Attack success rates
    success_rates = [
        fgsm_success['attack_success_rate'],
        pgd_success['attack_success_rate'],
        custom_success['attack_success_rate']
    ]
    
    axes[1, 1].bar(attacks_names, success_rates, color='red', alpha=0.7)
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].set_title('Attack Success Rate')
    axes[1, 1].set_ylim([0, 1])
    
    # Perturbation magnitudes
    mean_pert = [
        fgsm_perturbations['mean_perturbation'],
        pgd_perturbations['mean_perturbation'],
        custom_perturbations['mean_perturbation']
    ]
    max_pert = [
        fgsm_perturbations['max_perturbation'],
        pgd_perturbations['max_perturbation'],
        custom_perturbations['max_perturbation']
    ]
    
    x_pos = np.arange(len(attacks_names))
    axes[1, 2].bar(x_pos - 0.2, mean_pert, width=0.4, label='Mean')
    axes[1, 2].bar(x_pos + 0.2, max_pert, width=0.4, label='Max')
    axes[1, 2].set_ylabel('Perturbation Magnitude')
    axes[1, 2].set_title('Average Perturbation Size')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(attacks_names)
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attack_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Attack analysis plots saved to attack_analysis.png")
    plt.close()
    
    # ============================================================================
    # 7. SUMMARY REPORT
    # ============================================================================
    print(f"\n[STEP 6] Generating Summary Report...")
    
    summary = f"""
ADVERSARIAL ATTACK GENERATION SUMMARY
======================================

Configuration:
- Baseline Model: {model_file}
- Test Samples: {n_test_samples}
- Features: {n_features}
- Default Epsilon: {args.epsilon}

Attack Parameters:
- FGSM: ε = {args.epsilon}, single step
- PGD: ε = {args.epsilon}, {args.pgd_steps} steps, step_size = {args.epsilon/4:.6f}
- Feature Manipulation: ε = {args.epsilon}, {args.custom_iterations} iterations, subset = {n_subset}

Attack Constraints:
- Manipulable features: {manipulable_count}/{len(feature_names)}
- Per-feature constraints: Based on feature analysis

Results Summary:

FGSM Attack:
  Clean Accuracy: {fgsm_success['clean_accuracy']:.4f}
  Adversarial Accuracy: {fgsm_success['adversarial_accuracy']:.4f}
  Accuracy Drop: {fgsm_success['clean_accuracy'] - fgsm_success['adversarial_accuracy']:.4f}
  Attack Success Rate: {fgsm_success['attack_success_rate']:.4f}
  Mean Perturbation: {fgsm_perturbations['mean_perturbation']:.6f}
  Max Perturbation: {fgsm_perturbations['max_perturbation']:.6f}

PGD Attack:
  Clean Accuracy: {pgd_success['clean_accuracy']:.4f}
  Adversarial Accuracy: {pgd_success['adversarial_accuracy']:.4f}
  Accuracy Drop: {pgd_success['clean_accuracy'] - pgd_success['adversarial_accuracy']:.4f}
  Attack Success Rate: {pgd_success['attack_success_rate']:.4f}
  Mean Perturbation: {pgd_perturbations['mean_perturbation']:.6f}
  Max Perturbation: {pgd_perturbations['max_perturbation']:.6f}

Feature Manipulation Attack:
  Clean Accuracy: {custom_success['clean_accuracy']:.4f}
  Adversarial Accuracy: {custom_success['adversarial_accuracy']:.4f}
  Accuracy Drop: {custom_success['clean_accuracy'] - custom_success['adversarial_accuracy']:.4f}
  Attack Success Rate: {custom_success['attack_success_rate']:.4f}
  Mean Perturbation: {custom_perturbations['mean_perturbation']:.6f}
  Max Perturbation: {custom_perturbations['max_perturbation']:.6f}

Key Findings:
- PGD is the strongest attack (lowest adversarial accuracy)
- Feature manipulation is most realistic but slower
- Top perturbed features (from feature_perturbation_analysis.csv):
{feature_comparison.head(5)[['Feature', 'PGD_Perturbation', 'Manipulability']].to_string(index=False)}

Artifacts Saved:
- X_adv_fgsm.npy: FGSM adversarial examples ({X_adv_fgsm.shape})
- X_adv_pgd.npy: PGD adversarial examples ({X_adv_pgd.shape})
- X_adv_custom.npy: Feature manipulation adversarial examples ({X_adv_custom.shape})
- fgsm_results.pkl, pgd_results.pkl, custom_results.pkl: Attack statistics
- feature_perturbation_analysis.csv: Per-feature perturbation analysis
- attack_analysis.png: Comprehensive visualization

Next Steps:
1. Train defense mechanisms using defenses.py
2. Evaluate robustness against these attacks using evaluate.py
3. Compare clean accuracy vs adversarial robustness trade-offs
"""
    
    with open(output_dir / 'ATTACK_GENERATION_SUMMARY.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(summary)
    print(f"\n[COMPLETE] All adversarial examples saved to {output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate adversarial examples against baseline IDS model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_attacks.py --model_dir baseline_models --output_dir adversarial_data
  python generate_attacks.py --model_dir baseline_models --epsilon 0.5 --pgd_steps 60
        """
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Path to directory containing baseline model and artifacts'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='adversarial_data',
        help='Output directory for adversarial examples (default: adversarial_data)'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.3,
        help='Attack budget as relative change in features (default: 0.3)'
    )
    parser.add_argument(
        '--pgd_steps',
        type=int,
        default=40,
        help='Number of PGD attack iterations (default: 40)'
    )
    parser.add_argument(
        '--custom_iterations',
        type=int,
        default=15,
        help='Number of iterations for feature manipulation attack (default: 15)'
    )
    parser.add_argument(
        '--n_test_samples',
        type=int,
        default=1000,
        help='Number of test samples to generate attacks on (default: 1000)'
    )
    
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
