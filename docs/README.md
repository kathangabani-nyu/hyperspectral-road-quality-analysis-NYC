# Hyperspectral Road Quality Analysis - NYC

A research-grade machine learning pipeline for pavement classification using hyperspectral imagery. This system achieves 65-70% classification accuracy on the    hyperspectral dataset (288 spectral bands) using LiDAR-derived ground truth validation.

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

- **Validated Performance**: 65-70% overall accuracy with LiDAR-derived ground truth validation
- **Full Spectral Utilization**: Processes all 288 hyperspectral bands
- **Advanced Feature Engineering**: Implements vegetation, urban, and statistical spectral indices
- **Dimensionality Reduction**: PCA and intelligent band selection methods
- **Ground Truth Integration**: Uses NYC Land Cover Raster (2017) LiDAR data for validation
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

1. Clone the repository:
```bash
git clone https://github.com/kathangabani-
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
- Full 288-band processing
- Advanced feature engineering
- Ground truth integration
- Comprehensive evaluation metrics

## Data

This project uses:
-   hyperspectral imagery (288 bands)
- NYC Land Cover Raster Data (2017) - 6-inch resolution LiDAR-derived ground truth

Note: Large data files are excluded from the repository. See `docs/SETUP_INSTRUCTIONS.md` for data acquisition details.

## Results

The system achieves:
- Overall Accuracy: 67.2%
- Kappa Coefficient: 0.34
- Mean IoU: 50.8%
- Precision (Pavement): 81.6%
- Recall (Pavement): 87.0%
- F1-Score (Pavement): 84.2%

Results are saved to `outputs_research/` directory (excluded from git).

## Documentation

- `docs/PROJECT_REPORT.txt` - Comprehensive project report
- `docs/SETUP_INSTRUCTIONS.md` - Detailed setup guide
- `docs/QUICK_START.md` - Quick start guide
- `docs/DOCUMENTATION_INDEX.md` - Full documentation index

## Contributing

This is a research project. For questions or contributions, please open an issue or contact the repository maintainers.

## License

[Specify license if applicable]

## Citation

If you use this code in your research, please cite:

```
[Add citation information]
```

## Acknowledgments

-   dataset
- NYC Department of Information Technology and Telecommunications (DoITT) for Land Cover data
- Research collaborators and advisors

