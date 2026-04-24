"""
dataset_utils.py - Dataset Loading and Preprocessing

Load and preprocess the UNSW-NB15 intrusion detection dataset.
"""

import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetManager:
    """Unified dataset manager for UNSW-NB15 intrusion detection dataset."""
    
    def __init__(self, dataset_type='unsw-nb15'):
        if dataset_type.lower() != 'unsw-nb15':
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        self.dataset_type = dataset_type.lower()
        logger.info(f"DatasetManager initialized for {dataset_type}")
    
    def load_dataset(self, filepath):
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        logger.info(f"Loading {self.dataset_type} from {filepath}")
        return self._load_unsw_nb15(filepath)
    
    @staticmethod
    def _load_unsw_nb15(filepath):  
        """Load UNSW-NB15 dataset."""
        
        print(f"Loading UNSW-NB15 from {filepath}...")
        df = pd.read_csv(filepath)
        
        print(f"Initial rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        
        # Trouver la colonne label
        label_col = 'label'
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found")
        
        print(f"First 5 labels: {df['label'].head().tolist()}")
        
        # Identifier les colonnes non-numériques (sauf label)
        non_numeric_cols = []
        for col in df.columns:
            if col not in [label_col, 'attack_cat']:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    non_numeric_cols.append(col)
                    print(f"  Non-numeric column: {col}")
        
        if non_numeric_cols:
            print(f"Dropping {len(non_numeric_cols)} non-numeric columns")
            df = df.drop(columns=non_numeric_cols)
        
        # Supprimer les NaN
        initial_len = len(df)
        df = df.dropna()
        print(f"After removing NaN: {initial_len} → {len(df)} rows")
        
        if len(df) == 0:
            print("WARNING: All rows removed! Using fallback...")
            df = pd.read_csv(filepath, low_memory=False)
            
            # Garder seulement les colonnes numériques
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if label_col in numeric_cols:
                numeric_cols.remove(label_col)
            if 'attack_cat' in numeric_cols:
                numeric_cols.remove('attack_cat')
            
            df = df[[label_col] + numeric_cols]
            df = df.dropna()
            print(f"After fallback: {len(df)} rows")
        
        # Supprimer les doublons
        df = df.drop_duplicates()
        print(f"After removing duplicates: {len(df)} rows")
        
        if len(df) == 0:
            raise ValueError("Dataset is empty after preprocessing")
        
        # Colonnes à supprimer
        cols_to_drop = ['id', 'attack_cat', label_col]
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        
        # Features
        X = df.drop(columns=cols_to_drop).values.astype(np.float32)
        
        # Labels
        if df[label_col].dtype == 'object':
            y = (df[label_col] != 'Normal').astype(int).values
        else:
            y = df[label_col].values.astype(int)
        
        feature_names = df.drop(columns=cols_to_drop).columns.tolist()
        
        print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Label distribution: 0: {(y==0).sum()}, 1: {(y==1).sum()}")
        
        return X, y, feature_names