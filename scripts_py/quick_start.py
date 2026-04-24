#!/usr/bin/env python
"""
quick_start.py - Quick Start Verification Script

Verify environment and dataset availability before running DL-Adversarial pipeline.

This script:
    1. Checks that datasets exist and have appropriate sizes
    2. Verifies all required Python dependencies are installed
    3. Provides step-by-step commands for the workflow
    4. Helps troubleshoot common issues

Usage:
    python quick_start.py

Requirements:
    - Python 3.8+
    - All packages listed in readmes/requirements.txt

Author: DL-Adversarial Project
License: MIT
"""

import os
import sys
import logging
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_datasets():
    """
    Check if datasets are present and accessible.
    
    Returns:
        bool: True if all datasets found, False otherwise
    """
    logger.info("="*80)
    logger.info("CHECKING DATASETS")
    logger.info("="*80)
    
    issues = []
    
    # Check UNSW-NB15 (using pathlib for cross-platform compatibility)
    unsw_train = Path("data") / "UNSW_NB15_training-set.csv"
    unsw_test = Path("data") / "UNSW_NB15_testing-set.csv"
    
    logger.info("UNSW-NB15 Dataset:")
    if unsw_train.exists():
        size_mb = unsw_train.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ Training set: {unsw_train} ({size_mb:.1f} MB)")
    else:
        logger.warning(f"  ✗ Training set missing: {unsw_train}")
        issues.append(f"Missing: {unsw_train}")
    
    if unsw_test.exists():
        size_mb = unsw_test.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ Test set: {unsw_test} ({size_mb:.1f} MB)")
    else:
        logger.warning(f"  ✗ Test set missing: {unsw_test}")
        issues.append(f"Missing: {unsw_test}")
    
    # Check CIC-IDS2017
    cic_dir = Path("data") / "MachineLearningCVE"
    logger.info("CIC-IDS2017 Dataset:")
    
    if cic_dir.exists():
        csv_files = list(cic_dir.glob("*.csv"))
        if csv_files:
            logger.info(f"  ✓ Found {len(csv_files)} CSV files")
            for f in csv_files[:3]:  # Show first 3 files
                size_mb = f.stat().st_size / (1024 * 1024)
                logger.info(f"    - {f.name} ({size_mb:.1f} MB)")
            if len(csv_files) > 3:
                logger.info(f"    ... and {len(csv_files)-3} more files")
        else:
            logger.warning(f"  ✗ No CSV files in {cic_dir}")
            issues.append(f"No CSV files in: {cic_dir}")
    else:
        logger.warning(f"  ✗ Directory missing: {cic_dir}")
        issues.append(f"Missing: {cic_dir}")
    
    if issues:
        logger.warning("\n⚠ Issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    else:
        logger.info("\n✓ All datasets found!")
        return True


def check_dependencies():
    """
    Check if required Python packages are installed.
    
    Returns:
        bool: True if all dependencies installed, False otherwise
    """
    logger.info("\n" + "="*80)
    logger.info("CHECKING DEPENDENCIES")
    logger.info("="*80)
    
    required = {
        'numpy': '1.24+',
        'pandas': '2.0+',
        'sklearn': '1.3+',
        'tensorflow': '2.13+',
        'keras': '2.13+',
        'matplotlib': '3.7+',
        'seaborn': '0.12+',
        'tqdm': 'latest'
    }
    
    missing = []
    
    for package, version in required.items():
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            logger.info(f"  ✓ {package} ({version})")
        except ImportError:
            logger.warning(f"  ✗ {package} (MISSING)")
            missing.append(package)
    
    if missing:
        logger.warning(f"\n⚠ Missing packages: {', '.join(missing)}")
        logger.warning("Install with:")
        logger.warning("  pip install -r readmes/requirements.txt")
        return False
    else:
        logger.info("\n✓ All dependencies installed!")
        return True


def print_workflow():
    """Print the recommended workflow."""
    print("\n" + "="*80)
    print("RECOMMENDED WORKFLOW")
    print("="*80 + "\n")
    
    workflow = """
STEP 1: Merge CIC-IDS2017 files
================================
If using CIC-IDS2017, first merge all CSV files:

  cd "py files"
  python merge_cic.py
  cd ..

Output: data/cic_ids_2017_full.csv


STEP 2: Train Baseline Model
=============================
Choose your dataset and run:

For UNSW-NB15:
  cd "py files"
  python train_baseline.py ^
    --dataset unsw-nb15 ^
    --data_path "../data/UNSW_NB15_training-set.csv" ^
    --output_dir "../baseline_models" ^
    --epochs 100
  cd ..

For CIC-IDS2017:
  cd "py files"
  python train_baseline.py ^
    --dataset cic-ids2017 ^
    --data_path "../data/cic_ids_2017_full.csv" ^
    --output_dir "../baseline_models" ^
    --epochs 100
  cd ..

Outputs:
  - baseline_models/baseline_model.h5
  - baseline_models/scaler.pkl
  - baseline_models/feature_analysis.pkl
  - baseline_models/TRAINING_SUMMARY.txt


STEP 3: Generate Adversarial Examples
======================================
  cd "py files"
  python generate_attacks.py ^
    --model_dir "../baseline_models" ^
    --output_dir "../adversarial_data" ^
    --epsilon 0.3 ^
    --pgd_steps 40 ^
    --n_test_samples 1000
  cd ..

Outputs:
  - adversarial_data/X_adv_fgsm.npy
  - adversarial_data/X_adv_pgd.npy
  - adversarial_data/X_adv_custom.npy
  - adversarial_data/attack_analysis.png


STEP 4: Train Robust Models
============================
  cd "py files"
  python defenses.py ^
    --baseline_model "../baseline_models/baseline_model.h5" ^
    --adversarial_dir "../adversarial_data" ^
    --output_dir "../robust_models" ^
    --defense_type adversarial_training
  cd ..

Outputs:
  - robust_models/defense_*.h5 (trained models)
  - robust_models/DEFENSE_SUMMARY.txt


STEP 5: Evaluate All Models
============================
  cd "py files"
  python evaluate.py ^
    --baseline_dir "../baseline_models" ^
    --robust_models "../robust_models" ^
    --adversarial_data "../adversarial_data" ^
    --output_dir "../evaluation_results"
  cd ..

Outputs:
  - evaluation_results/comparison_tables.csv
  - evaluation_results/evaluation_summary.png
  - evaluation_results/EVALUATION_REPORT.txt
"""
    
    print(workflow)


def main():
    """
    Run environment verification and display workflow instructions.
    
    Checks datasets and dependencies, then prints recommended workflow.
    """
    logger.info("="*80)
    logger.info("DL-ADVERSARIAL PROJECT - QUICK START VERIFICATION")
    logger.info("="*80)
    
    datasets_ok = check_datasets()
    deps_ok = check_dependencies()
    
    if not datasets_ok:
        logger.warning("\n⚠ WARNING: Some datasets are missing!")
        logger.warning("Please download the required datasets and place them in the data/ directory.")
    
    if not deps_ok:
        logger.warning("\n⚠ WARNING: Some dependencies are missing!")
        logger.warning("Please install them before running the training scripts.")
    
    if datasets_ok and deps_ok:
        logger.info("\n✓ Environment is ready! You can start the workflow.")
    
    print_workflow()
    
    logger.info("="*80)
    logger.info("QUICK TIPS")
    logger.info("="*80)
    logger.info("""
1. Dataset Format:
   - UNSW-NB15: Standard CSV with 'label' column (lowercase)
   - CIC-IDS2017: Must merge files first using merge_cic.py

2. First Time Setup:
   - Install dependencies: pip install -r readmes/requirements.txt
   - Check datasets exist in data/ folder
   - Start with STEP 1 above

3. Common Parameters:
   - --dataset: Choose 'unsw-nb15' or 'cic-ids2017'
   - --epochs: Increase for better accuracy (default: 100)
   - --epsilon: Attack strength (default: 0.3, range: 0.0-1.0)

4. Output Directories:
   - baseline_models/: Trained baseline model + artifacts
   - adversarial_data/: Generated adversarial examples
   - robust_models/: Models trained with defenses
   - evaluation_results/: Comparison results

5. For More Details:
   - Read USAGE_GUIDE.md for detailed instructions
   - Check generated SUMMARY files in output directories
   - Review code comments in individual scripts
""")
    
    logger.info("="*80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
