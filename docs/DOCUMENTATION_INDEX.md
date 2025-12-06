# Documentation Index

Complete documentation for the Hyperspectral Pavement Classification System.

## Quick Reference

| Document | Purpose | Audience |
|----------|---------|----------|
| README.md | System overview and usage | All users |
| QUICK_START.md | Step-by-step usage guide | New users |
| CHANGELOG.md | Version history | Developers |
| IMPROVEMENT_PLAN.md | Technical improvements | Developers |
| RESEARCH_GRADE_SUMMARY.md | Research methodology | Researchers |
| DOCUMENTATION_INDEX.md | This file | All users |

## Getting Started

### For New Users
1. Start with **README.md** for system overview
2. Follow **QUICK_START.md** for first-time setup
3. Run `python main.py` for quick test

### For Researchers
1. Read **RESEARCH_GRADE_SUMMARY.md** for methodology
2. Review **IMPROVEMENT_PLAN.md** for technical details
3. Run `python main_enhanced.py` for research-quality results

### For Developers
1. See **IMPROVEMENT_PLAN.md** for architecture
2. Check **CHANGELOG.md** for version history
3. Review module docstrings for API details

## Core Documentation

### README.md
**Primary documentation file**

Contents:
- System overview
- Installation instructions
- Quick start guide
- Configuration options
- Performance metrics
- Troubleshooting
- Research applications

### QUICK_START.md
**Step-by-step usage guide**

Contents:
- First-time setup
- Basic workflow
- Configuration examples
- Common issues and solutions
- Parameter tuning tips

### CHANGELOG.md
**Version history and changes**

Contents:
- Version progression (1.0 -> 2.0)
- New features by version
- Bug fixes
- Performance improvements
- Breaking changes
- Upgrade instructions

## Technical Documentation

### IMPROVEMENT_PLAN.md
**Technical improvements and architecture**

Contents:
- Problems identified in original code
- Solutions implemented
- Architecture overview
- Module descriptions
- Code quality improvements
- Workflow documentation

### RESEARCH_GRADE_SUMMARY.md
**Research methodology and validation**

Contents:
- System architecture
- Research-grade features
- Feature engineering methods
- Classification algorithms
- Statistical validation
- Evaluation metrics
- Research workflow
- Publication components

## Module Documentation

### Core Modules

**config.py**
- Basic configuration parameters
- RGB visualization settings
- Classification parameters

**data_loader.py**
- Hyperspectral data loading
- Window-based reading
- Feature reshaping and normalization

**visualizer.py**
- RGB composite generation
- Percentile-based contrast stretching
- Result visualization
- Figure generation (300 DPI)

**feature_engineering.py**
- Basic spectral indices
- Statistical features
- Feature augmentation

**clustering.py**
- K-means clustering
- Cluster identification
- Binary mask generation

**classifier.py**
- Random Forest training
- Dataset balancing
- Evaluation metrics
- Cross-validation

### Research-Grade Modules

**advanced_config.py**
- Research parameters
- All 288 bands support
- Multiple classifier options
- Statistical testing settings

**dimensionality_reduction.py**
- PCA with variance analysis
- Band selection algorithms
- Visualization of results

**advanced_features.py**
- Vegetation indices (NDVI, EVI)
- Urban indices (NDBI, UI)
- Brightness indices
- Statistical features
- Spatial features

**ensemble_classifier.py**
- Random Forest (enhanced)
- SVM implementation
- XGBoost integration
- Ensemble methods

**statistical_analysis.py**
- McNemar's test
- Cohen's Kappa
- Confidence intervals
- Cross-validation
- Error analysis

**advanced_evaluation.py**
- ROC curves
- Precision-Recall curves
- Comprehensive metrics
- Classifier comparison

### Pipeline Scripts

**main.py**
- Basic pipeline (30 bands)
- Fast testing (~2 minutes)
- 67% accuracy
- Output: `outputs/`

**main_enhanced.py**
- Research pipeline (288 bands)
- Full feature engineering
- 67% accuracy
- Output: `outputs_research/`

## Output Documentation

### Visualization Files (PNG, 300 DPI)

Basic Pipeline (`outputs/figures/`):
- `rgb_composite.png` - True color visualization
- `clustering_results.png` - K-means results
- `classification_results.png` - Final classification
- `confusion_matrix.png` - Performance visualization
- `spectral_signatures.png` - Class separability

Enhanced Pipeline (`outputs_research/figures/`):
- All basic visualizations plus:
- `pca_variance_explained.png` - PCA analysis
- `feature_importance.png` - Feature ranking
- `roc_curves.png` - ROC analysis
- `precision_recall_curves.png` - PR analysis
- `classifier_comparison.png` - Algorithm comparison

### Data Files

Classification Maps (GeoTIFF):
- `pavement_classification_map.tif` - Final classification
- `cluster_map.tif` - K-means clusters

Model Files (PKL):
- `pca_model.pkl` - Trained PCA model
- `classifier_ensemble.pkl` - Trained ensemble

### Analysis Files

Metrics (TXT, CSV):
- `classification_metrics.txt` - Detailed performance
- `comprehensive_metrics.txt` - Research-grade metrics
- `feature_importance.csv` - Feature rankings
- `cv_results.csv` - Cross-validation results
- `statistical_tests.txt` - Significance tests

## Code Organization

```
VIP/
├── Core System
│   ├── config.py
│   ├── data_loader.py
│   ├── visualizer.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── classifier.py
│   └── main.py
│
├── Research-Grade
│   ├── advanced_config.py
│   ├── dimensionality_reduction.py
│   ├── advanced_features.py
│   ├── ensemble_classifier.py
│   ├── statistical_analysis.py
│   ├── advanced_evaluation.py
│   └── main_enhanced.py
│
├── Documentation
│   ├── README.md
│   ├── QUICK_START.md
│   ├── CHANGELOG.md
│   ├── IMPROVEMENT_PLAN.md
│   ├── RESEARCH_GRADE_SUMMARY.md
│   └── DOCUMENTATION_INDEX.md
│
├── Configuration
│   ├── requirements.txt
│   └── requirements_research.txt
│
└── Outputs
    ├── outputs/              (Basic pipeline)
    └── outputs_research/     (Enhanced pipeline)
```

## API Reference

### Data Loading

```python
from data_loader import load_hyperspectral_data, normalize_features

# Load data
data, profile, transform = load_hyperspectral_data()

# Normalize features
X_normalized = normalize_features(X, method='zscore')
```

### Feature Engineering

```python
from advanced_features import extract_advanced_features

# Extract features
X_augmented, feature_names = extract_advanced_features(X, data_3d)
```

### Dimensionality Reduction

```python
from dimensionality_reduction import reduce_dimensionality

# Apply PCA
X_reduced, info = reduce_dimensionality(X, y, method='pca', n_components=50)
```

### Classification

```python
from ensemble_classifier import create_ensemble_classifier

# Create classifier
clf = create_ensemble_classifier(method='voting')
clf.fit(X_train, y_train)
```

### Evaluation

```python
from advanced_evaluation import compute_comprehensive_metrics

# Evaluate
metrics = compute_comprehensive_metrics(y_true, y_pred, y_pred_proba)
```

## Configuration Reference

### Basic Parameters (config.py)

```python
# Test area
START_X = 2000
START_Y = 3000
TEST_SIZE_X = 200
TEST_SIZE_Y = 200

# Bands
NUM_BANDS = 30

# RGB
RED_BAND_IDX = 25
GREEN_BAND_IDX = 15
BLUE_BAND_IDX = 5

# Classification
N_CLUSTERS = 5
RF_N_ESTIMATORS = 50
```

### Research Parameters (advanced_config.py)

```python
# All bands
USE_ALL_BANDS = True

# PCA
USE_PCA = True
PCA_COMPONENTS = 50

# Multiple classifiers
USE_RANDOM_FOREST = True
USE_SVM = True
USE_XGBOOST = True

# Statistical validation
USE_CROSS_VALIDATION = True
CV_FOLDS = 5
```

## Troubleshooting Guide

Common issues and solutions documented in:
- README.md (General troubleshooting)
- QUICK_START.md (Usage-specific issues)
- IMPROVEMENT_PLAN.md (Technical fixes)

## Research Publication Support

For research paper preparation:
1. Use `outputs_research/` for all figures
2. Reference **RESEARCH_GRADE_SUMMARY.md** for methodology
3. Cite metrics from `comprehensive_metrics.txt`
4. Include statistical tests from `statistical_tests.txt`

## Version Information

- Current Version: 2.0 (Research-Grade)
- Release Date: October 2025
- Status: Production-Ready and Research-Publishable

## Support

For technical questions:
1. Check relevant documentation file
2. Review module docstrings
3. Examine example usage in pipeline scripts
4. Consult troubleshooting sections

## Updates

Documentation is maintained in sync with code releases. Check CHANGELOG.md for latest updates.

## Contributing

When adding new features:
1. Update relevant documentation
2. Add docstrings to code
3. Update configuration examples
4. Document in CHANGELOG.md
5. Update this index if adding new docs

## License

See LICENSE file for details.

