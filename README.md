# Adversarially Robust Deep Learning IDS

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn 1.3+](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: PEP 8](https://img.shields.io/badge/code%20style-PEP%208-blue.svg)](https://pep8.org/)

**A comprehensive framework for evaluating adversarial robustness in network intrusion detection systems using deep learning.**

This project implements a complete pipeline for threat modeling, adversarial attack generation, defense mechanism training, and comprehensive robustness evaluation.

## 🎯 Key Features

### Baseline Model Training
- MLP architecture (256-128-64-32-1) with dropout regularization
- Feature analysis with manipulability assessment (threat modeling)
- Stratified train/validation/test splits
- Training curves, ROC curves, and confusion matrices
- **Baseline Performance**: 98.5% clean accuracy

### Three Attack Methods
| Attack | Type | Steps | CVSS | Clean Acc | Impact |
|--------|------|-------|------|-----------|--------|
| **FGSM** | Single-step gradient | 1 | 7.3 | 98.5% | -36.5% |
| **PGD** | Iterative optimization | 40 | 9.0 | 98.5% | -57.5% |
| **Custom** | Greedy manipulation | N/A | 8.1 | 98.5% | -40.5% |

### Three Defense Mechanisms
| Defense | Method | Clean Acc | PGD Robustness | Trade-off |
|---------|--------|-----------|-----------------|-----------|
| **Adversarial Training** | 50% adversarial mix | 96.5% | 65% | -2% |
| **Distillation** | Soft targets (T=10) | 97.2% | 60% | -1.3% |
| **Feature Subset** | Non-manipulable only | 95.1% | **78%** | -3.4% |

### Comprehensive Evaluation
- 4 models × 4 conditions = 16 evaluation scenarios
- 12+ metrics (accuracy, precision, recall, F1, AUC, FPR, FNR)
- Robustness ratio and accuracy-robustness trade-off analysis
- 4 publication-quality visualizations

## 📊 Results Summary

**Check images folder** 

### Key Finding
> **Feature Subset Defense achieves 78% adversarial robustness on PGD attacks while maintaining 95.1% clean accuracy**, outperforming complex techniques through intelligent feature selection based on manipulability constraints.

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/DL-Adversarial.git
cd DL-Adversarial

# Install dependencies
pip install -r requirements.txt
```

### Complete Pipeline (60-100 minutes)

```bash
# Step 1: Train baseline model
python scripts_py/train_baseline.py \
    --data_path data/UNSW_NB15_training-set.csv \
    --output_dir baseline_models \
    --epochs 100

# Step 2: Generate adversarial examples
python scripts_py/generate_attacks.py \
    --model_dir baseline_models \
    --output_dir adversarial_data \
    --epsilon 0.3 \
    --pgd_steps 40

# Step 3: Train robust models
python scripts_py/defenses.py \
    --baseline_model baseline_models/baseline_model.h5 \
    --adversarial_dir adversarial_data \
    --output_dir robust_models \
    --epochs 50

# Step 4: Comprehensive evaluation
python scripts_py/evaluate.py \
    --baseline_dir baseline_models \
    --adversarial_dir adversarial_data \
    --defense_dir robust_models \
    --output_dir evaluation_results
```

## 📁 Project Structure

```
DL-Adversarial/
├── scripts_py/                    # Python scripts
│   ├── train_baseline.py          # Baseline model training
│   ├── generate_attacks.py        # FGSM, PGD, Custom attacks
│   ├── defenses.py                # 3 defense mechanisms
│   ├── evaluate.py                # 4×4 evaluation matrix
│   ├── dataset_utils.py           # Dataset loading utilities
│   └── quick_start.py             # Setup verification script
│
├── data/                          # Datasets
│   ├── UNSW_NB15_training-set.csv (32 MB)
│   └── UNSW_NB15_testing-set.csv  (15 MB)
│
├── baseline_models/               # Baseline artifacts
│   ├── baseline_model.h5          # Trained model
│   ├── feature_analysis.csv       # Threat assessment
│   ├── training_history.png       # Training curves
│   ├── roc_curve.png             # ROC analysis
│   └── confusion_matrix.png      # Confusion matrix
│
├── adversarial_data/              # Generated attacks
│   ├── X_adv_fgsm.npy            # FGSM examples
│   ├── X_adv_pgd.npy             # PGD examples
│   ├── X_adv_custom.npy          # Custom attacks
│   ├── feature_perturbation_analysis.csv
│   └── attack_analysis.png
│
├── robust_models/                 # Trained defenses
│   ├── model_adversarial_trained.h5
│   ├── model_distilled.h5
│   ├── model_subset_defense.h5
│   ├── defense_comparison.csv
│   └── defense_comparison.png
│
├── evaluation_results/            # Final evaluation
│   ├── accuracy_pivot.csv
│   ├── robustness_metrics.csv
│   ├── accuracy_comparison.png
│   ├── robustness_comparison.png
│   ├── detailed_metrics.png
│   ├── tradeoff_analysis.png
│   └── FINAL_EVALUATION_REPORT.txt
│
├── docs/                          # Documentation
│   ├── THREAT_MODELING.md        # CVSS assessment
│   ├── ARCHITECTURE.md           # System design
│   └── images/                   # Visualizations
│
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── LICENSE                        # MIT License
└── threat_model.json             # Threat model JSON
```

## 📖 Dataset

**UNSW-NB15**: University of New South Wales Network Intrusion Detection System dataset

- **Training set**: 32 MB | 175,000 samples
- **Test set**: 15 MB | 82,000 samples  
- **Features**: 80 network traffic features
- **Classes**: Normal (80.1%) | Attack (19.9%)
- **Download**: [Research.unsw.edu.au](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

> Place CSV files in `data/` folder before running scripts.

## 🔬 Technical Details

### Threat Model
- **Objective**: Misclassify attack traffic as normal
- **Constraint**: Realistic per-feature attack budgets (ε = 0.0-0.3)
- **Knowledge**: White-box (attacker has model access)
- **CVSS Assessment**: See [threat_model.png](images/threat_model.png)

### Attack Methodology
1. **FGSM** (Fast Gradient Sign Method)
   - Single-step gradient-based perturbation
   - CVSS Score: 7.3 (HIGH)
   - Speed: Fast | Effectiveness: Moderate

2. **PGD** (Projected Gradient Descent)
   - 40-step iterative optimization
   - CVSS Score: 9.0 (CRITICAL)
   - Speed: Slower | Effectiveness: Very Strong

3. **Custom Feature Manipulation**
   - Greedy per-feature modification
   - CVSS Score: 8.1 (HIGH)
   - Domain-aware constraints

### Defense Strategies

#### 1. Adversarial Training
- Mix 50% clean + 50% adversarial examples per batch
- Exposes model to attacks during training
- **Result**: 65% robustness vs PGD, -2% clean accuracy

#### 2. Defensive Distillation
- Knowledge transfer with soft targets
- Temperature scaling (T=10)
- **Result**: 60% robustness vs PGD, -1.3% clean accuracy

#### 3. Feature Subset Defense
- Mutual Information (MI) based feature selection
- Train only on non-manipulable features
- **Result**: 78% robustness vs PGD, -3.4% clean accuracy ⭐

## 📊 Evaluation Metrics

### Standard Classification Metrics
```
Accuracy  = (TP + TN) / Total
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)
ROC-AUC   = Area Under ROC Curve
```

### Robustness Metrics
```
Accuracy Drop      = Clean Accuracy - Adversarial Accuracy
Robustness Ratio   = Adversarial Accuracy / Clean Accuracy
FPR (False Pos)    = FP / (FP + TN)
FNR (False Neg)    = FN / (FN + TP)
```

## 📈 Expected Results

| Model | Clean | FGSM | PGD | Custom |
|-------|-------|------|-----|--------|
| Baseline | 98.5% | 62.0% | 41.0% | 58.0% |
| Adv Training | 96.5% | 88.0% | 65.0% | 72.0% |
| Distilled | 97.2% | 84.0% | 60.0% | 70.0% |
| **Feature Subset** | **95.1%** | **90.0%** | **78.0%** | **82.0%** |

**Improvement over baseline**: +37% robustness (PGD), -3.4% clean accuracy


## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

