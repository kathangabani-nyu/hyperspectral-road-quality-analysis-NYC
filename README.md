# Hyperspectral Road Quality Analysis - NYC

A research-grade machine learning pipeline for pavement classification using hyperspectral imagery. This system achieves **88.81% overall accuracy** on hyperspectral data (288 spectral bands) using LiDAR-derived ground truth validation with advanced multi-stage filtering and signature matching techniques.

**Note:** This project was developed using NYU TuckMapping hyperspectral dataset. This dataset is classified and requires special permissions granted only to NYU researchers. Users must provide their own hyperspectral data to use this pipeline.

## Project Structure

```
hyperspectral-road-quality-analysis-NYC/
├── src/                    # Source code modules
│   ├── main_enhanced.py   # Main research pipeline
│   ├── main.py            # Basic pipeline
│   ├── data_loader.py     # Data loading utilities
│   ├── feature_engineering.py
│   ├── classifier.py
│   ├── clustering.py
│   └── ...                # Additional modules
├── docs/                  # Documentation
│   ├── README.md
│   ├── PROJECT_REPORT.txt
│   ├── SETUP_INSTRUCTIONS.md
│   └── ...                # Additional documentation
├── scripts/               # Utility scripts
│   ├── run_classification.bat
│   └── ...
├── data/                  # Data directory (excluded from git)
├── tests/                 # Test files
├── requirements.txt       # Basic dependencies
└── requirements_research.txt  # Full research dependencies
```

## Key Features

- **Validated Performance**: 88.81% overall accuracy with LiDAR-derived ground truth validation
- **Full Spectral Utilization**: Processes all 288 hyperspectral bands with PCA dimensionality reduction
- **Advanced Feature Engineering**: 332 features including spectral indices, spatial features, and sidewalk-specific signatures
- **Multi-Stage Filtering**: Sequential spectral masking, height-based filtering, geometric filtering, and signature matching
- **Ground Truth Integration**: Uses NYC Land Cover Raster (2017) LiDAR data for independent validation
- **Height-Based Filtering**: Integrates LiDAR elevation data to distinguish rooftops from ground-level pavement
- **Signature Matching**: Novel approach using ground truth sidewalk signatures to expand detection
- **Random Forest Classifier**: 300-tree ensemble with comprehensive feature importance analysis
- **Publication Ready**: Generates high-quality figures (400 DPI) and comprehensive metrics

## System Requirements

### Software Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended for full dataset processing)
- Operating System: Windows, Linux, or macOS

### Python Dependencies
See `requirements.txt` (basic) or `requirements_research.txt` (full research features)

## Installation

### Basic Installation

1. Clone the repository:
```bash
git clone https://github.com/kathangabani-nyu/hyperspectral-road-quality-analysis-NYC.git
cd hyperspectral-road-quality-analysis-NYC
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements_research.txt
```

## Usage

### Basic Pipeline

```bash
python src/main.py
```

### Enhanced Research Pipeline

```bash
python src/main_enhanced.py
```

The enhanced pipeline includes:
- Full 288-band processing with PCA dimensionality reduction (99.5% variance retained)
- Advanced feature engineering (332 features: spectral indices, spatial, statistical)
- Ground truth integration with coordinate system reprojection
- Height-based filtering using LiDAR elevation data
- Multi-stage spatial filtering (rooftop removal, geometric constraints)
- Sidewalk signature matching for expanded detection
- Comprehensive evaluation metrics (accuracy, precision, recall, F1, Kappa, IoU)

## Data

This project was developed using:
- NYU TuckMapping hyperspectral imagery (288 bands) - **Restricted access, NYU researchers only**
- NYC Land Cover Raster Data (2017) - 6-inch resolution LiDAR-derived ground truth (publicly available)

**Important:** The NYU TuckMapping dataset is classified and requires special permissions granted only to NYU researchers. This repository does not provide access to this dataset. Users must provide their own hyperspectral data to use this pipeline.

Note: Large data files are excluded from the repository.

## Results

The system achieves (latest run):
- **Overall Accuracy**: 88.81%
- **Kappa Coefficient**: 0.7762 (substantial agreement)
- **Mean IoU**: 79.84%
- **Precision (Pavement)**: 85.99%
- **Recall (Pavement)**: 92.72%
- **F1-Score (Pavement)**: 89.23%

### Processing Pipeline
- Input: 640,000 pixels (800×800 window)
- After spectral masking: 98,919 pixels (15.5%)
- Feature engineering: 288 bands → 332 features → 19 PCA components
- Training: 28,000 samples (balanced)
- Test: 12,000 samples
- Final classification: 52,383 pavement pixels (8.18% of total)

### Key Innovations
1. **Ground Truth Integration**: Independent LiDAR-derived labels eliminate circular dependency
2. **Multi-Stage Filtering**: Spectral → Height → Geometric → Signature matching
3. **Signature Matching**: 75.7% expansion using ground truth sidewalk signatures
4. **Height-Based Filtering**: LiDAR DEM distinguishes rooftops from ground-level pavement

Results are saved to `outputs_research/` directory (excluded from git).

## Documentation

- `docs/METHODOLOGY_REPORT.txt` - **NEW**: Comprehensive methodology report with step-by-step pipeline
- `docs/PROJECT_REPORT.txt` - Comprehensive project report
- `docs/SETUP_INSTRUCTIONS.md` - Detailed setup guide
- `docs/QUICK_START.md` - Quick start guide
- `docs/DOCUMENTATION_INDEX.md` - Full documentation index

The methodology report details all 19 processing steps, rationale for methodology choices, and performance metrics.

## Acknowledgments

- NYU TuckMapping hyperspectral dataset (restricted access, NYU researchers only)
- NYC Department of Information Technology and Telecommunications (DoITT) for Land Cover data
- Research collaborators and advisors

