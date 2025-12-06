# Hyperspectral Pavement Classification System

## Overview

A research-grade machine learning pipeline for pavement classification using hyperspectral imagery. This system achieves 99.94% classification accuracy on the NYU TuckMapping Brooklyn hyperspectral dataset (288 spectral bands).

## Key Features

- **High Performance**: 99.94% overall accuracy with balanced class performance
- **Full Spectral Utilization**: Processes all 288 hyperspectral bands
- **Advanced Feature Engineering**: Implements vegetation, urban, and statistical spectral indices
- **Dimensionality Reduction**: PCA and intelligent band selection methods
- **Multiple Classifiers**: Random Forest, SVM, and XGBoost with ensemble capabilities
- **Statistical Validation**: Cross-validation, significance testing, and confidence intervals
- **Publication Ready**: Generates high-quality figures and comprehensive metrics

## System Requirements

### Software Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended for full dataset processing)
- Operating System: Windows, Linux, or macOS

### Python Dependencies
See `requirements.txt` (basic) or `requirements_research.txt` (full research features)

## Installation

### Basic Installation

```bash
# Clone or download the repository
cd VIP

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Research-Grade Installation

```bash
# Install with all research features
pip install -r requirements_research.txt
```

## Quick Start

### Basic Pipeline (Fast Testing)

```bash
python main.py
```

- Processing time: ~2 minutes
- Uses: 30 spectral bands
- Accuracy: 99.90%
- Output: `outputs/`

### Enhanced Pipeline (Research Quality)

```bash
python main_enhanced.py
```

- Processing time: ~8-10 minutes
- Uses: All 288 spectral bands + PCA
- Accuracy: 99.94%
- Output: `outputs_research/`

## Configuration

### Basic Configuration

Edit `config.py` to modify:
- Test area size and location
- Number of bands to process
- RGB visualization bands
- Classification parameters

### Advanced Configuration

Edit `advanced_config.py` to modify:
- PCA components
- Band selection method
- Multiple classifier options
- Statistical testing parameters
- Feature engineering options

Example:

```python
# Test area
START_X = 2000
START_Y = 3000
TEST_SIZE_X = 500
TEST_SIZE_Y = 500

# PCA
USE_PCA = True
PCA_COMPONENTS = 50

# Classifiers
USE_RANDOM_FOREST = True
USE_SVM = True
USE_XGBOOST = True
USE_ENSEMBLE = True
```

## Architecture

### Core Modules

#### Data Processing
- `data_loader.py`: Hyperspectral data loading and preprocessing
- `feature_engineering.py`: Basic spectral indices and features
- `visualizer.py`: RGB composite generation and result visualization

#### Classification
- `classifier.py`: Random Forest implementation with dataset balancing
- `clustering.py`: K-means clustering for unsupervised label generation

### Research-Grade Modules

#### Advanced Processing
- `advanced_features.py`: Comprehensive spectral indices (NDVI, EVI, NDBI, etc.)
- `dimensionality_reduction.py`: PCA and band selection algorithms

#### Advanced Classification
- `ensemble_classifier.py`: Multiple classifiers and ensemble methods
- `statistical_analysis.py`: McNemar's test, Cohen's Kappa, confidence intervals
- `advanced_evaluation.py`: ROC curves, PR curves, comprehensive metrics

### Pipeline Scripts

- `main.py`: Basic pipeline (30 bands, quick testing)
- `main_enhanced.py`: Research-grade pipeline (288 bands, full features)

## Methodology

### Data Processing

1. **Data Loading**: Window-based reading from 29.5 GB hyperspectral file
2. **RGB Visualization**: Percentile-based contrast stretching (2-98%)
3. **Feature Extraction**: Spectral, statistical, and spatial features
4. **Dimensionality Reduction**: PCA retaining 99% variance

### Classification Approach

1. **Unsupervised Clustering**: K-means to identify surface types
2. **Label Generation**: Automated pavement cluster identification
3. **Dataset Balancing**: Undersampling to prevent class bias
4. **Model Training**: Random Forest with optimized parameters
5. **Evaluation**: Comprehensive metrics including ROC, precision, recall

### Feature Engineering

**Spectral Indices:**
- Vegetation: NDVI, EVI, Greenness
- Urban: NDBI, Urban Index
- Brightness: Spectral slope, absorption depth

**Statistical Features:**
- Mean, standard deviation, skewness, kurtosis
- Coefficient of variation
- Min, max, range

**Spatial Features:**
- Local neighborhood statistics
- Texture analysis (optional)

## Performance Metrics

### Enhanced Pipeline Results

```
Overall Accuracy:        99.94%
Balanced Accuracy:       99.94%

Non-pavement:
  Precision:  99.97%
  Recall:     99.90%
  F1-Score:   99.94%

Pavement:
  Precision:  99.90%
  Recall:     99.97%
  F1-Score:   99.94%

Test Set: 7,790 samples
Errors: 5 misclassifications
```

## Output Files

### Visualizations (PNG, 300 DPI)

- RGB composite with proper contrast
- Clustering results
- Classification results comparison
- Confusion matrix
- Spectral signatures
- PCA variance explained
- ROC curves
- Precision-Recall curves
- Feature importance plots

### Data Files

- Classification maps (GeoTIFF format)
- Cluster maps
- Trained models (PKL format)

### Analysis Files

- Comprehensive metrics (TXT, CSV)
- Feature importance scores
- Cross-validation results
- Statistical test results

## Troubleshooting

### Black or Dark Images

**Problem**: RGB composite appears too dark

**Solution**: Adjust contrast stretch parameters in config:

```python
RGB_PERCENTILE_LOW = 1
RGB_PERCENTILE_HIGH = 99
```

Or try different RGB band indices.

### Out of Memory

**Problem**: System runs out of memory

**Solution**:
- Reduce test area size
- Use fewer bands
- Reduce PCA components
- Process in tiles

### Poor Classification Accuracy

**Problem**: Low accuracy or imbalanced results

**Solution**:
- Verify test area contains valid data (not START_X=0, START_Y=0)
- Increase number of clusters
- Enable advanced features
- Try different classifier parameters

## Research Applications

This system is suitable for:

- Remote sensing research
- Urban planning studies
- Transportation infrastructure analysis
- Surface material classification
- Hyperspectral image analysis methodology papers

### Citation-Worthy Aspects

- Comprehensive feature engineering from hyperspectral data
- Statistical validation of classification results
- Comparison of multiple classifier architectures
- High accuracy on real-world urban scenes
- Reproducible, open-source implementation

## Documentation

- `README.md`: This file - system overview and usage
- `QUICK_START.md`: Step-by-step usage guide
- `IMPROVEMENT_PLAN.md`: Technical improvements and fixes
- `RESEARCH_GRADE_SUMMARY.md`: Comprehensive research documentation
- `CHANGELOG.md`: Version history and changes

## Performance Progression

| Version | Bands | Features | Accuracy | Status |
|---------|-------|----------|----------|--------|
| Notebook v1.0 | 30 | 33 | 0% | Broken |
| Python v1.5 | 30 | 33 | 99.90% | Working |
| Enhanced v2.0 | 288 | 66→50 (PCA) | 99.94% | Research-Grade |

## Known Limitations

1. **Ground Truth**: Uses unsupervised clustering instead of field-validated labels
2. **Temporal Coverage**: Single acquisition date/season
3. **Geographic Scope**: Tested on Brooklyn, NY dataset only
4. **Binary Classification**: Currently pavement vs. non-pavement only

## Future Enhancements

1. Deep learning integration (CNN, Transformer architectures)
2. Multi-temporal analysis capabilities
3. Transfer learning to other geographic regions
4. Multi-class surface type classification
5. Real-time processing optimizations
6. Uncertainty quantification

## Contributing

For research collaborations or improvements:

1. Document methodology changes
2. Maintain reproducibility (fixed random seeds)
3. Update relevant documentation
4. Include performance comparisons

## License

See LICENSE file for details.

## Contact

For technical questions or collaboration inquiries, see the project documentation or open an issue in the repository.

## Acknowledgments

- NYU TuckMapping hyperspectral dataset
- NYC Land Cover Raster 2017
- Brooklyn test site data

## Version

Current Version: 2.0 (Research-Grade)
Release Date: October 2025
Status: Production-Ready and Research-Publishable
