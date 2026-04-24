"""
train_baseline.py - Train Baseline IDS Model

Train a baseline intrusion detection system (IDS) model on clean network traffic data.

This script:
    1. Loads and preprocesses the UNSW-NB15 or CIC-IDS2017 dataset
    2. Performs feature analysis and manipulability assessment for threat modeling
    3. Creates stratified train/validation/test splits
    4. Trains a baseline MLP neural network classifier
    5. Evaluates performance on clean data
    6. Saves model and preprocessing artifacts for later use

Usage:
    python train_baseline.py \\
        --dataset unsw-nb15 \\
        --data_path data/UNSW_NB15_training-set.csv \\
        --output_dir baseline_models \\
        --epochs 100

Requirements:
    - TensorFlow 2.13+
    - scikit-learn 1.3+
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Shows the progess bar for loops 

# Import dataset utilities
from dataset_utils import DatasetManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")



class DataLoader:
    """Load and preprocess IDS datasets."""
    
    @staticmethod
    def load_dataset(dataset_type, filepath):
        """
        Load dataset using DatasetManager.
        Args:
            dataset_type: 'cic-ids2017' or 'unsw-nb15'
            filepath: Path to dataset CSV file
            
        Returns:
            X: Feature matrix
            y: Label array
            feature_names: List of feature names
        """
        dm = DatasetManager(dataset_type)
        return dm.load_dataset(filepath)


class FeatureAnalyzer:
    """Analyze features for manipulability."""
    
    @staticmethod
    def analyze_features(X, y, feature_names):
        """
        Analyze features and create manipulability assessment.
        
        Returns:
            DataFrame with feature statistics and manipulability assessment
        """
        print("\n" + "="*80)
        print("FEATURE ANALYSIS & MANIPULABILITY ASSESSMENT")
        print("="*80 + "\n")
        
        n_features = X.shape[1]
        analysis = []
        
        for i in range(n_features):
            feature = X[:, i]
            
            # Statistics
            mean_val = feature.mean()
            std_val = feature.std()
            min_val = feature.min()
            max_val = feature.max()
            
            # Attack vs Normal
            attack_mean = X[y==1, i].mean()
            normal_mean = X[y==0, i].mean()
            
            # Variance ratio (higher = more different in attacks)
            attack_var = X[y==1, i].var()
            normal_var = X[y==0, i].var()
            variance_ratio = attack_var / (normal_var + 1e-8)
            
            # Correlation with class
            correlation = np.corrcoef(feature, y)[0, 1]
            
            # Manipulability heuristic:
            # Features that vary significantly in attacks are less manipulable
            # (because modifying them may break the attack)
            if variance_ratio > 1.5:
                # More variance in attacks = important to attack structure
                manipulability = 'Semi-Manipulable'
                max_change = 0.15
            elif abs(correlation) < 0.1:
                # Low correlation with class = less critical
                manipulability = 'Manipulable'
                max_change = 0.3
            else:
                # Moderate correlation
                manipulability = 'Semi-Manipulable'
                max_change = 0.15
            
            analysis.append({
                'Feature_Index': i,
                'Feature_Name': feature_names[i],
                'Mean': mean_val,
                'Std': std_val,
                'Min': min_val,
                'Max': max_val,
                'Attack_Mean': attack_mean,
                'Normal_Mean': normal_mean,
                'Variance_Ratio': variance_ratio,
                'Correlation_with_Class': correlation,
                'Manipulability': manipulability,
                'Max_Change_Epsilon': max_change,
            })
        
        analysis_df = pd.DataFrame(analysis)
        
        # Summary statistics
        print(f"Total features: {n_features}")
        print(f"Manipulable features: {(analysis_df['Manipulability'] == 'Manipulable').sum()}")
        print(f"Semi-Manipulable features: {(analysis_df['Manipulability'] == 'Semi-Manipulable').sum()}")
        
        print("\nTop 10 Features by Correlation with Class:")
        print(analysis_df.nlargest(10, 'Correlation_with_Class')[
            ['Feature_Name', 'Correlation_with_Class', 'Manipulability']
        ].to_string(index=False))
        
        return analysis_df


class BaselineModel:
    """Build and train baseline IDS model."""
    
    @staticmethod
    def build_model(input_dim, name='baseline_model'):
        """
        Build standard MLP for binary classification.
        
        Args:
            input_dim: Number of input features
            name: Model name
            
        Returns:
            Uncompiled Keras model
        """
        model = keras.Sequential([
            keras.layers.Dense(
                256, activation='relu',
                input_dim=input_dim,
                kernel_regularizer=keras.regularizers.l2(1e-4),
                name='dense_1'
            ),
            keras.layers.Dropout(0.3),  # Disable 30% of neurons to prevent overfitting
            
            keras.layers.Dense(
                128, activation='relu',
                kernel_regularizer=keras.regularizers.l2(1e-4),
                name='dense_2'
            ),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(
                64, activation='relu',
                kernel_regularizer=keras.regularizers.l2(1e-4),
                name='dense_3'
            ),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(32, activation='relu', name='dense_4'),
            keras.layers.Dense(1, activation='sigmoid', name='output') # Output: 0 = normal, 1 = attack
        ], name=name)
        
        return model
    
    @staticmethod
    def compile_model(model, learning_rate=0.001):
        """Compile model with appropriate optimizer and loss."""
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        return model
    
    @staticmethod 
    def train_model(model, X_train, y_train, X_val, y_val,
                   epochs=100, batch_size=32, verbose=1):
        """
        Train model with early stopping and learning rate reduction.
        
        Returns:
            Training history
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',        # monitors the loss over validation
                patience=15,               # waits 15 epochs without improvement before stopping
                restore_best_weights=True,
                verbose=1                  
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',      
                factor=0.5,                # Divide LR by 2
                patience=5,             
                min_lr=1e-7,               # LR minimum
                verbose=1                
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history


class Evaluator:
    """Evaluate model performance."""
    
    @staticmethod
    def evaluate_model(model, X_test, y_test, model_name='Model'):
        """
        Comprehensive evaluation on test set.
        
        Returns:
            Dictionary with all metrics
        """
        # Predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'fpr': fpr,
            'fnr': fnr,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
        }
        
        # Print report
        print(f"\n{'='*80}")
        print(f"EVALUATION REPORT: {model_name}")
        print(f"{'='*80}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc_score:.4f}")
        print(f"\nFalse Positive Rate: {fpr:.4f}")
        print(f"False Negative Rate: {fnr:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"TP: {tp} | TN: {tn}")
        print(f"FP: {fp} | FN: {fn}")
        print(f"{'='*80}\n")
        
        return metrics, y_pred, y_pred_proba
    
    @staticmethod
    def plot_training_history(history, output_file='training_history.png'):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train')
        axes[0, 0].plot(history.history['val_accuracy'], label='Val')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train')
        axes[0, 1].plot(history.history['val_loss'], label='Val')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Train')
        axes[1, 0].plot(history.history['val_precision'], label='Val')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Train')
        axes[1, 1].plot(history.history['val_recall'], label='Val')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {output_file}")
        plt.close()
    
    @staticmethod
    def plot_roc_curve(y_test, y_pred_proba, output_file='roc_curve.png'):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {output_file}")
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, output_file='confusion_matrix.png'):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_file}")
        plt.close()


def main(args):
    """Main training pipeline."""
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*50)
    print("BASELINE IDS MODEL TRAINING")
    print("="*50 + "\n")
    
    # ============================================================================
    # 1. LOAD DATASET
    # ============================================================================
    print("\n[STEP 1] Loading Dataset...")
    
    loader = DataLoader()
    X, y, feature_names = loader.load_dataset(args.dataset.lower(), args.data_path)
    
    # ============================================================================
    # 2. FEATURE ANALYSIS
    # ============================================================================
    print("\n[STEP 2] Analyzing Features...")
    analyzer = FeatureAnalyzer()
    analysis_df = analyzer.analyze_features(X, y, feature_names)
    analysis_df.to_csv(output_dir / 'feature_analysis.csv', index=False)
    print(f"Feature analysis saved to feature_analysis.csv")
    
    # ============================================================================
    # 3. NORMALIZE DATA
    # ============================================================================
    print("\n[STEP 3] Normalizing Data...")
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    print(f"Feature ranges after normalization: [0, 1]")
    
    # Save scaler
    scaler_file = output_dir / 'scaler.pkl'
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_file}")
    
    # ============================================================================
    # 4. TRAIN/VAL/TEST SPLIT
    # ============================================================================
    print("\n[STEP 4] Splitting Data...")
    
    # First split: 70% train, 30% temp (for val/test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_normalized, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Second split: 50-50 split of temp into val and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(y)*100:.1f}%)")
    print(f"Val set:   {X_val.shape[0]} samples ({X_val.shape[0]/len(y)*100:.1f}%)")
    print(f"Test set:  {X_test.shape[0]} samples ({X_test.shape[0]/len(y)*100:.1f}%)")
    print(f"\nClass distribution:")
    print(f"Train - Normal: {(y_train==0).sum()}, Attack: {(y_train==1).sum()}")
    print(f"Val   - Normal: {(y_val==0).sum()}, Attack: {(y_val==1).sum()}")
    print(f"Test  - Normal: {(y_test==0).sum()}, Attack: {(y_test==1).sum()}")
    
    # Save split indices for reproducibility
    split_info = {
        'train_size': X_train.shape[0],
        'val_size': X_val.shape[0],
        'test_size': X_test.shape[0],
        'train_attack_rate': (y_train==1).mean(),
        'val_attack_rate': (y_val==1).mean(),
        'test_attack_rate': (y_test==1).mean(),
    }
    with open(output_dir / 'split_info.pkl', 'wb') as f:
        pickle.dump(split_info, f)
    
    # ============================================================================
    # 5. BUILD MODEL
    # ============================================================================
    print("\n[STEP 5] Building Baseline Model...")
    
    baseline = BaselineModel()
    model = baseline.build_model(input_dim=X_train.shape[1], name='baseline_ids')
    model = baseline.compile_model(model, learning_rate=0.001)
    
    model.summary()
    
    # ============================================================================
    # 6. TRAIN MODEL
    # ============================================================================
    print("\n[STEP 6] Training Model...")
    print(f"Training for maximum {args.epochs} epochs with early stopping (patience=15)...\n")
    
    history = baseline.train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=32,
        verbose=1
    )
    
    # Save trained model
    model_file = output_dir / 'baseline_model.h5'
    model.save(model_file)
    print(f"\nModel saved to {model_file}")
    
    # ============================================================================
    # 7. EVALUATE ON CLEAN TEST DATA
    # ============================================================================
    print("\n[STEP 7] Evaluating on Clean Test Data...")
    
    evaluator = Evaluator()
    metrics, y_pred, y_pred_proba = evaluator.evaluate_model(
        model, X_test, y_test, model_name='Baseline Model (Clean Data)'
    )
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / 'baseline_metrics_clean.csv', index=False)
    
    # ============================================================================
    # 8. VISUALIZATIONS
    # ============================================================================
    print("\n[STEP 8] Creating Visualizations...")
    
    evaluator.plot_training_history(history, output_dir / 'training_history.png')
    evaluator.plot_roc_curve(y_test, y_pred_proba.flatten(), output_dir / 'roc_curve.png')
    evaluator.plot_confusion_matrix(y_test, y_pred, output_dir / 'confusion_matrix.png')
    
    # Feature statistics by class
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top features by variance in attacks
    attack_var = X[y==1].var(axis=0)
    top_var_idx = np.argsort(attack_var)[-10:]
    
    axes[0, 0].barh([feature_names[i] for i in top_var_idx], attack_var[top_var_idx])
    axes[0, 0].set_xlabel('Variance')
    axes[0, 0].set_title('Top 10 Features by Variance in Attacks')
    
    # Top features by correlation
    correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
    top_corr_idx = np.argsort(np.abs(correlations))[-10:]
    
    axes[0, 1].barh([feature_names[i] for i in top_corr_idx], correlations[top_corr_idx])
    axes[0, 1].set_xlabel('Correlation with Class')
    axes[0, 1].set_title('Top 10 Features by Correlation with Class')
    
    # Attack rate by feature manipulability
    manipulability_counts = analysis_df['Manipulability'].value_counts()
    axes[1, 0].bar(manipulability_counts.index, manipulability_counts.values)
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Feature Distribution by Manipulability')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Class distribution
    class_counts = np.bincount(y)
    axes[1, 1].bar(['Normal', 'Attack'], class_counts)
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Dataset Class Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Feature analysis plots saved to feature_analysis.png")
    plt.close()
    
    # ============================================================================
    # 9. SAVE TRAINING ARTIFACTS
    # ============================================================================
    print("\n[STEP 9] Saving Artifacts...")
    
    # Save feature names
    with open(output_dir / 'feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save feature analysis
    feature_analysis = {
        'feature_names': feature_names,
        'num_features': X.shape[1],
        'analysis_df': analysis_df,
    }
    with open(output_dir / 'feature_analysis.pkl', 'wb') as f:
        pickle.dump(feature_analysis, f)
    
    # Summary report
    summary = f"""
BASELINE IDS MODEL TRAINING SUMMARY
====================================

Dataset: {args.dataset}
Date: {pd.Timestamp.now()}

Data Statistics:
- Total samples: {len(y)}
- Attack samples: {(y==1).sum()} ({(y==1).mean()*100:.2f}%)
- Normal samples: {(y==0).sum()} ({(y==0).mean()*100:.2f}%)
- Features: {X.shape[1]}

Train/Val/Test Split:
- Train: {X_train.shape[0]} samples
- Val:   {X_val.shape[0]} samples
- Test:  {X_test.shape[0]} samples

Model Architecture:
- Input: {X_train.shape[1]} features
- Hidden: 256 → 128 → 64 → 32 neurons
- Output: Binary classification (sigmoid)
- Regularization: L2 + Dropout

Training Configuration:
- Optimizer: Adam (lr=0.001)
- Loss: Binary Crossentropy
- Batch Size: 32
- Epochs: {len(history.history['loss'])} (early stopped)
- Early Stopping Patience: 15

Baseline Performance (Clean Test Data):
- Accuracy:  {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall:    {metrics['recall']:.4f}
- F1-Score:  {metrics['f1']:.4f}
- ROC-AUC:   {metrics['auc']:.4f}
- FPR:       {metrics['fpr']:.4f}
- FNR:       {metrics['fnr']:.4f}

Artifacts Saved:
- Model: baseline_model.h5
- Scaler: scaler.pkl
- Feature Names: feature_names.pkl
- Feature Analysis: feature_analysis.csv, feature_analysis.pkl
- Metrics: baseline_metrics_clean.csv
- Plots: training_history.png, roc_curve.png, confusion_matrix.png, feature_analysis.png

Next Steps:
1. Generate adversarial examples using generate_attacks.py
2. Train robust models using defenses.py
3. Evaluate all models using evaluate.py
"""
    
    with open(output_dir / 'TRAINING_SUMMARY.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    print(summary)
    
    print(f"\n[COMPLETE] All artifacts saved to {output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train baseline IDS model'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='cic-ids2017',
        choices=['cic-ids2017', 'unsw-nb15'],
        help='Dataset to use'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to dataset CSV file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./baseline_models',
        help='Directory to save models and artifacts'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum number of epochs'
    )
    
    args = parser.parse_args()
    main(args)
