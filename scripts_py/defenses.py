"""
defenses.py - Train Robust Defense Models

Implement robust defense mechanisms against adversarial examples using three strategies:
1. Adversarial Training - Train on mix of clean and adversarial examples
2. Defensive Distillation - Train student model on soft targets from teacher  
3. Feature Subset Defense - Train on only robust (non-manipulable) features

This script:
    1. Loads baseline model and adversarial examples
    2. Trains three robust models using different defense mechanisms
    3. Evaluates robust models on clean and adversarial data
    4. Compares defense effectiveness
    5. Saves all trained models and evaluation results

Usage:
    python defenses.py \\
        --baseline_model baseline_models/baseline_model.h5 \\
        --adversarial_dir adversarial_data \\
        --output_dir robust_models

Requirements:
    - TensorFlow 2.13+
    - scikit-learn 1.3+
    - NumPy 1.24+

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdversarialTraining:
    """
    Defense: Train model on mixed clean and adversarial examples.
    
    Strategy: Expose model to adversarial examples during training to improve robustness.
    Mixes 50% clean examples with 50% adversarial examples in each epoch.
    """
    
    def __init__(self, baseline_model, attack_function=None):
        """
        Initialize adversarial training defense.
        
        Args:
            baseline_model: Keras model to train
            attack_function: Function to generate adversarial examples during training
        """
        self.baseline_model = baseline_model
        self.attack_function = attack_function
        logger.info("AdversarialTraining initialized")
    
    @staticmethod
    def generate_fgsm_batch(model, X_batch, y_batch, epsilon=0.3):
        """
        Generate FGSM adversarial examples for a batch.
        
        Args:
            model: Keras model
            X_batch: Feature batch
            y_batch: Label batch
            epsilon: Attack perturbation magnitude
            
        Returns:
            np.ndarray: Adversarial examples
        """
        X_batch = X_batch.astype(np.float32)
        X_var = tf.Variable(X_batch)
        
        with tf.GradientTape() as tape:
            logits = model(X_var, training=False)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.reshape(tf.cast(y_batch, tf.float32), (-1, 1)),
                logits=logits
            )
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, X_var)
        perturbation = epsilon * tf.sign(gradients)
        X_adv = X_batch + perturbation.numpy()
        X_adv = np.clip(X_adv, 0, 1)
        
        return X_adv
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100,
              batch_size=32, epsilon=0.3, adversarial_fraction=0.5):
        """
        Train model using mix of clean and adversarial examples.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
            epochs (int): Training epochs
            batch_size (int): Batch size
            epsilon (float): Adversarial perturbation magnitude
            adversarial_fraction (float): Fraction of adversarial examples in each batch
            
        Returns:
            dict: Training history with loss and accuracy
        """
        logger.info(f"Starting adversarial training: {epochs} epochs, batch_size={batch_size}")
        
        # Custom training loop to mix clean and adversarial examples
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        n_batches = len(X_train) // batch_size
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            epoch_acc = 0
            
            for batch_idx in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{epochs}"):
                batch_start = batch_idx * batch_size
                batch_end = batch_start + batch_size
                
                X_clean = X_shuffled[batch_start:batch_end]
                y_clean = y_shuffled[batch_start:batch_end]
                
                #Takes the clean batch and creates an adversarial version using FGSM.
                X_adv = self.generate_fgsm_batch(
                    self.baseline_model, X_clean, y_clean, epsilon=epsilon
                )
                
                # Mix clean and adversarial
                n_adv = int(len(X_clean) * adversarial_fraction)
                X_mixed = np.vstack([X_clean[n_adv:], X_adv[:n_adv]])
                y_mixed = np.hstack([y_clean[n_adv:], y_clean[:n_adv]])
                
                # Shuffle mixed batch
                perm = np.random.permutation(len(X_mixed))
                X_mixed = X_mixed[perm]
                y_mixed = y_mixed[perm]
                
                # Train step
                metrics = self.baseline_model.train_on_batch(X_mixed, y_mixed)
                epoch_loss += metrics[0]
                epoch_acc += metrics[1]
            
            # Validation
            val_metrics = self.baseline_model.evaluate(X_val, y_val, verbose=0)
            
            history['loss'].append(epoch_loss / n_batches)
            history['accuracy'].append(epoch_acc / n_batches)
            history['val_loss'].append(val_metrics[0])
            history['val_accuracy'].append(val_metrics[1])
            
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}: Loss={history['loss'][-1]:.4f}, "
                      f"Acc={history['accuracy'][-1]:.4f}, "
                      f"Val_Loss={val_metrics[0]:.4f}, Val_Acc={val_metrics[1]:.4f}")
        
        return history


class DefensiveDistillation:
    """Defense: Train student model on soft targets from teacher."""
    
    @staticmethod
    def build_student_model(input_dim):
        """Build student model (same architecture as baseline)."""
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_dim=input_dim,
                             kernel_regularizer=keras.regularizers.l2(1e-4)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(1e-4)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(1e-4)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    @staticmethod
    def distill(teacher_model, X_train, y_train, X_val, y_val,
               temperature=10, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Train student model with knowledge distillation.
        Returns: Trained student model and training history
        """
        # Get soft targets from teacher
        print("Generating soft targets from teacher model...")
        teacher_logits = teacher_model.predict(X_train, verbose=0)
        soft_targets = tf.nn.sigmoid(teacher_logits / temperature).numpy()

        # Validation soft targets
        val_teacher_logits = teacher_model.predict(X_val, verbose=0)
        val_soft_targets = tf.nn.sigmoid(val_teacher_logits / temperature).numpy()

        # Build student
        student = DefensiveDistillation.build_student_model(X_train.shape[1])

        # Compile student with standard binary crossentropy
        student.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print(f"Training student model with temperature={temperature}...")
        history = student.fit(
            X_train, soft_targets,
            validation_data=(X_val, val_soft_targets),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                )
            ],
            verbose=1
        )

        return student, history


class FeatureSubsetDefense:
    """Defense: Train on robust (non-manipulable) features only.
    
    All 80 features                    Selected features (50)
┌─────────────────────────┐        ┌─────────────────────┐
│ Duration (can't touch)  │────────│ ✓ Keep              │
│ Source Port (can change)│────────│ ✗ Discard           │
│ Packet Len (limited)    │────────│ ✓ Keep              │
│ Timestamp (can't touch) │────────│ ✓ Keep              │
│ ...                     │        │ ...                 │
└─────────────────────────┘        └─────────────────────┘
    
    """
    
    @staticmethod
    def select_robust_features(X_train, y_train, analysis_df, top_k=None):
        """
        Select non-manipulable and important features.
        Args:
            X_train: Training features
            y_train: Training labels
            analysis_df: Feature analysis DataFrame
            top_k: Number of top features to select (if None, use manipulability)
            
        Returns: Selected feature indices
        """
        if top_k is not None:
            # Select top-k important features using mutual information
            mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
            selected_features = np.argsort(mi_scores)[-top_k:]
            print(f"Selected top {top_k} features by mutual information")
        else:
            # Select only non-manipulable features
            non_manipulable = analysis_df['Manipulability'] == 'Non-Manipulable'
            selected_features = np.where(non_manipulable)[0]
            print(f"Selected {len(selected_features)} non-manipulable features")
        
        return np.sort(selected_features)
    
    @staticmethod
    def build_subset_model(input_dim):
        """Build model for subset of features."""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=input_dim,
                             kernel_regularizer=keras.regularizers.l2(1e-4)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(1e-4)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    @staticmethod
    def train(X_train, y_train, X_val, y_val, selected_features,
             epochs=100, batch_size=32, learning_rate=0.001):
        """
        Train model on subset of features.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            selected_features: Indices of features to use
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Trained model and training history
        """
        # Select features
        X_train_subset = X_train[:, selected_features]
        X_val_subset = X_val[:, selected_features]
        
        print(f"Training on {len(selected_features)} features (subset)")
        
        # Build and train model
        model = FeatureSubsetDefense.build_subset_model(X_train_subset.shape[1])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        history = model.fit(
            X_train_subset, y_train,
            validation_data=(X_val_subset, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                )
            ],
            verbose=1
        )
        
        return model, history, selected_features


class DefenseEvaluator:
    """Evaluate defense effectiveness."""
    
    @staticmethod
    def evaluate_defense(model, X_clean, X_adv, y_test, defense_name='Defense'):
        """
        Evaluate defense against adversarial examples.
        
        Args:
            model: Keras model
            X_clean: Clean test examples
            X_adv: Adversarial test examples
            y_test: Test labels
            defense_name: Name of defense
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Predictions
        y_pred_clean = (model.predict(X_clean, verbose=0) > 0.5).flatten()
        y_pred_adv = (model.predict(X_adv, verbose=0) > 0.5).flatten()
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'clean_accuracy': accuracy_score(y_test, y_pred_clean),
            'adversarial_accuracy': accuracy_score(y_test, y_pred_adv),
            'accuracy_drop': accuracy_score(y_test, y_pred_clean) - accuracy_score(y_test, y_pred_adv),
            'clean_precision': precision_score(y_test, y_pred_clean, zero_division=0),
            'adversarial_precision': precision_score(y_test, y_pred_adv, zero_division=0),
            'clean_recall': recall_score(y_test, y_pred_clean, zero_division=0),
            'adversarial_recall': recall_score(y_test, y_pred_adv, zero_division=0),
            'clean_f1': f1_score(y_test, y_pred_clean, zero_division=0),
            'adversarial_f1': f1_score(y_test, y_pred_adv, zero_division=0),
        }
        
        print(f"\n{'='*80}")
        print(f"DEFENSE EVALUATION: {defense_name}")
        print(f"{'='*80}")
        print(f"Clean Accuracy:        {metrics['clean_accuracy']:.4f}")
        print(f"Adversarial Accuracy:  {metrics['adversarial_accuracy']:.4f}")
        print(f"Accuracy Drop:         {metrics['accuracy_drop']:.4f}")
        print(f"Clean Precision:       {metrics['clean_precision']:.4f}")
        print(f"Adversarial Precision: {metrics['adversarial_precision']:.4f}")
        print(f"Clean Recall:          {metrics['clean_recall']:.4f}")
        print(f"Adversarial Recall:    {metrics['adversarial_recall']:.4f}")
        print(f"{'='*80}\n")
        
        return metrics


def main(args):
    """Main defense training pipeline."""
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("DEFENSE MECHANISM TRAINING")
    print("="*80 + "\n")
    
    # ============================================================================
    # 1. LOAD BASELINE MODEL AND DATA
    # ============================================================================
    print("[STEP 1] Loading Baseline Model and Training Data...")
    
    baseline_model = keras.models.load_model(args.baseline_model)
    # Re-compile the loaded model to avoid optimizer variable mismatch errors
    baseline_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    print(f"Baseline model loaded from {args.baseline_model}")
    
    # Load feature analysis
    model_dir = Path(args.baseline_model).parent
    with open(model_dir / 'feature_analysis.pkl', 'rb') as f:
        feature_analysis = pickle.load(f)
    analysis_df = feature_analysis['analysis_df']
    
    with open(model_dir / 'feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    # Create synthetic training data (in practice, load actual data)
    np.random.seed(42)
    n_train = 5000
    n_features = len(feature_names)
    
    X_train = np.random.randn(n_train, n_features)
    X_train = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0) + 1e-8)
    X_train = np.clip(X_train, 0, 1).astype(np.float32)
    y_train = np.random.binomial(1, 0.2, n_train)
    
    # Validation data
    X_val = np.random.randn(1000, n_features)
    X_val = (X_val - X_val.min(axis=0)) / (X_val.max(axis=0) - X_val.min(axis=0) + 1e-8)
    X_val = np.clip(X_val, 0, 1).astype(np.float32)
    y_val = np.random.binomial(1, 0.2, 1000)
    
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    
    # Load adversarial examples (if available)
    adv_dir = Path(args.adversarial_dir)
    if adv_dir.exists():
        print(f"\nLoading adversarial examples from {adv_dir}...")
        try:
            X_adv_test_pgd = np.load(adv_dir / 'X_adv_pgd.npy')
            print(f"Loaded PGD adversarial examples: {X_adv_test_pgd.shape}")
        except:
            print("Could not load adversarial examples")
            X_adv_test_pgd = None
    else:
        X_adv_test_pgd = None
    
    # ============================================================================
    # 2. DEFENSE 1: ADVERSARIAL TRAINING
    # ============================================================================
    print("\n[STEP 2] Training Defense 1: Adversarial Training...")
    
    # Copy baseline model for adversarial training
    adv_trainer = AdversarialTraining(baseline_model)
    
    print("Starting adversarial training (mixing clean + adversarial examples)...")
    history_adv = adv_trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=32,
        epsilon=args.epsilon,
        adversarial_fraction=0.5
    )
    
    model_adv_trained = baseline_model
    model_adv_trained.save(output_dir / 'model_adversarial_trained.h5')
    print(f"Adversarial training model saved")
    
    # Save history
    with open(output_dir / 'history_adversarial_training.pkl', 'wb') as f:
        pickle.dump(history_adv, f)
    
    # ============================================================================
    # 3. DEFENSE 2: DEFENSIVE DISTILLATION
    # ============================================================================
    print("\n[STEP 3] Training Defense 2: Defensive Distillation...")
    
    # Rebuild baseline for distillation (don't modify original)
    teacher_model = keras.models.load_model(args.baseline_model)
    teacher_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    model_distilled, history_distill = DefensiveDistillation.distill(
        teacher_model, X_train, y_train, X_val, y_val,
        temperature=args.temperature,
        epochs=args.epochs,
        batch_size=32,
        learning_rate=0.001
    )
    
    model_distilled.save(output_dir / 'model_distilled.h5')
    print(f"Distilled model saved")
    
    # Save history
    history_distill_dict = {
        'loss': history_distill.history['loss'],
        'accuracy': history_distill.history['accuracy'],
        'val_loss': history_distill.history['val_loss'],
        'val_accuracy': history_distill.history['val_accuracy'],
    }
    with open(output_dir / 'history_distillation.pkl', 'wb') as f:
        pickle.dump(history_distill_dict, f)
    
    # ============================================================================
    # 4. DEFENSE 3: FEATURE SUBSET DEFENSE
    # ============================================================================
    print("\n[STEP 4] Training Defense 3: Feature Subset Defense...")
    
    selected_features = FeatureSubsetDefense.select_robust_features(
        X_train, y_train, analysis_df, top_k=args.feature_subset_k
    )
    
    model_subset, history_subset, selected_feat = FeatureSubsetDefense.train(
        X_train, y_train, X_val, y_val,
        selected_features=selected_features,
        epochs=args.epochs,
        batch_size=32,
        learning_rate=0.001
    )
    
    model_subset.save(output_dir / 'model_subset_defense.h5')
    print(f"Feature subset defense model saved")
    
    # Save history
    history_subset_dict = {
        'loss': history_subset.history['loss'],
        'accuracy': history_subset.history['accuracy'],
        'val_loss': history_subset.history['val_loss'],
        'val_accuracy': history_subset.history['val_accuracy'],
    }
    with open(output_dir / 'history_subset_defense.pkl', 'wb') as f:
        pickle.dump(history_subset_dict, f)
    
    # Save selected features
    with open(output_dir / 'selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)
    
    # ============================================================================
    # 5. EVALUATION (if adversarial data available)
    # ============================================================================
    print("\n[STEP 5] Evaluating Defenses...")
    
    if X_adv_test_pgd is not None:
        # Create synthetic test data
        X_test = np.random.randn(1000, n_features)
        X_test = (X_test - X_test.min(axis=0)) / (X_test.max(axis=0) - X_test.min(axis=0) + 1e-8)
        X_test = np.clip(X_test, 0, 1).astype(np.float32)
        y_test = np.random.binomial(1, 0.2, 1000)
        
        # Truncate adversarial examples to match test size
        X_adv_test_pgd = X_adv_test_pgd[:len(X_test)]
        
        evaluator = DefenseEvaluator()
        
        # Evaluate baseline (for comparison)
        baseline = keras.models.load_model(args.baseline_model)
        baseline.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        baseline_metrics = evaluator.evaluate_defense(
            baseline, X_test, X_adv_test_pgd, y_test,
            defense_name='Baseline (No Defense)'
        )
        
        # Evaluate defenses
        adv_train_metrics = evaluator.evaluate_defense(
            model_adv_trained, X_test, X_adv_test_pgd, y_test,
            defense_name='Adversarial Training'
        )
        
        distill_metrics = evaluator.evaluate_defense(
            model_distilled, X_test, X_adv_test_pgd, y_test,
            defense_name='Defensive Distillation'
        )
        
        # Feature subset (need to select same features from test data)
        X_test_subset = X_test[:, selected_features]
        X_adv_test_subset = X_adv_test_pgd[:, selected_features]
        
        subset_metrics = evaluator.evaluate_defense(
            model_subset, X_test_subset, X_adv_test_subset, y_test,
            defense_name='Feature Subset Defense'
        )
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            'Defense': ['Baseline', 'Adversarial Training', 'Distillation', 'Feature Subset'],
            'Clean_Accuracy': [
                baseline_metrics['clean_accuracy'],
                adv_train_metrics['clean_accuracy'],
                distill_metrics['clean_accuracy'],
                subset_metrics['clean_accuracy']
            ],
            'Adversarial_Accuracy': [
                baseline_metrics['adversarial_accuracy'],
                adv_train_metrics['adversarial_accuracy'],
                distill_metrics['adversarial_accuracy'],
                subset_metrics['adversarial_accuracy']
            ],
            'Accuracy_Drop': [
                baseline_metrics['accuracy_drop'],
                adv_train_metrics['accuracy_drop'],
                distill_metrics['accuracy_drop'],
                subset_metrics['accuracy_drop']
            ],
            'Robustness_Ratio': [
                baseline_metrics['adversarial_accuracy'] / baseline_metrics['clean_accuracy'] if baseline_metrics['clean_accuracy'] > 0 else 0,
                adv_train_metrics['adversarial_accuracy'] / adv_train_metrics['clean_accuracy'] if adv_train_metrics['clean_accuracy'] > 0 else 0,
                distill_metrics['adversarial_accuracy'] / distill_metrics['clean_accuracy'] if distill_metrics['clean_accuracy'] > 0 else 0,
                subset_metrics['adversarial_accuracy'] / subset_metrics['clean_accuracy'] if subset_metrics['clean_accuracy'] > 0 else 0,
            ]
        })
        
        print("\nDefense Comparison:")
        print(comparison_df.to_string(index=False))
        comparison_df.to_csv(output_dir / 'defense_comparison.csv', index=False)
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Accuracy comparison
        x = np.arange(len(comparison_df))
        width = 0.35
        
        axes[0].bar(x - width/2, comparison_df['Clean_Accuracy'], width, label='Clean')
        axes[0].bar(x + width/2, comparison_df['Adversarial_Accuracy'], width, label='Adversarial')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Clean vs Adversarial Accuracy')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(comparison_df['Defense'], rotation=45, ha='right')
        axes[0].legend()
        axes[0].set_ylim([0, 1])
        
        # Accuracy drop
        colors = ['red' if drop > 0.3 else 'orange' if drop > 0.1 else 'green' 
                 for drop in comparison_df['Accuracy_Drop']]
        axes[1].bar(comparison_df['Defense'], comparison_df['Accuracy_Drop'], color=colors)
        axes[1].set_ylabel('Accuracy Drop')
        axes[1].set_title('Accuracy Drop Under Attack')
        axes[1].set_xticklabels(comparison_df['Defense'], rotation=45, ha='right')
        
        # Robustness ratio
        axes[2].bar(comparison_df['Defense'], comparison_df['Robustness_Ratio'])
        axes[2].set_ylabel('Robustness Ratio')
        axes[2].set_title('Robustness Ratio (Adv_Acc / Clean_Acc)')
        axes[2].set_xticklabels(comparison_df['Defense'], rotation=45, ha='right')
        axes[2].axhline(y=0.8, color='g', linestyle='--', label='Good Robustness (80%)')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'defense_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Defense comparison plot saved")
        plt.close()
    
    else:
        print("Adversarial examples not found. Skipping evaluation.")
    
    # ============================================================================
    # 6. SUMMARY REPORT
    # ============================================================================
    print("\n[STEP 6] Generating Summary Report...")
    
    summary = f"""
DEFENSE MECHANISM TRAINING SUMMARY
===================================

Configuration:
- Baseline Model: {args.baseline_model}
- Epsilon: {args.epsilon}
- Epochs: {args.epochs}
- Temperature (Distillation): {args.temperature}
- Feature Subset Size: {args.feature_subset_k}

Defense 1: Adversarial Training
- Method: Train on mix of clean and adversarial examples
- Adversarial Fraction: 0.5 (50% adversarial in each batch)
- Attack Used: FGSM (epsilon={args.epsilon})
- Training Epochs: {len(history_adv['loss'])}
- Final Train Loss: {history_adv['loss'][-1]:.4f}
- Final Train Acc: {history_adv['accuracy'][-1]:.4f}
- Final Val Acc: {history_adv['val_accuracy'][-1]:.4f}

Defense 2: Defensive Distillation
- Method: Train student on soft targets from teacher
- Temperature: {args.temperature}
- Training Epochs: {len(history_distill.history['loss'])}
- Final Train Acc: {history_distill.history['accuracy'][-1]:.4f}
- Final Val Acc: {history_distill.history['val_accuracy'][-1]:.4f}

Defense 3: Feature Subset Defense
- Method: Train on robust (non-manipulable) features only
- Selected Features: {len(selected_features)}/{n_features}
- Training Epochs: {len(history_subset.history['loss'])}
- Final Train Acc: {history_subset.history['accuracy'][-1]:.4f}
- Final Val Acc: {history_subset.history['val_accuracy'][-1]:.4f}

Models Saved:
- model_adversarial_trained.h5
- model_distilled.h5
- model_subset_defense.h5
- selected_features.pkl

Histories Saved:
- history_adversarial_training.pkl
- history_distillation.pkl
- history_subset_defense.pkl

Next Steps:
1. Use evaluate.py to comprehensively compare all defenses
2. Test against FGSM, PGD, and custom attacks
3. Analyze trade-offs between clean and adversarial accuracy
4. Generate final robustness report
"""
    
    with open(output_dir / 'DEFENSE_TRAINING_SUMMARY.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(summary)
    print(f"\n[COMPLETE] All defense models saved to {output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train robust defense mechanisms'
    )
    parser.add_argument(
        '--baseline_model',
        type=str,
        required=True,
        help='Path to baseline model'
    )
    parser.add_argument(
        '--adversarial_dir',
        type=str,
        default='./adversarial_data',
        help='Directory containing adversarial examples'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./robust_models',
        help='Directory to save robust models'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.3,
        help='Attack budget'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=10.0,
        help='Temperature for distillation'
    )
    parser.add_argument(
        '--feature_subset_k',
        type=int,
        default=50,
        help='Number of features to select for feature subset defense'
    )
    
    args = parser.parse_args()
    main(args)
