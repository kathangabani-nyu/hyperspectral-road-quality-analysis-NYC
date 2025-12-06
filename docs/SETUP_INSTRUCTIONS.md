# Setup Instructions

## Installation

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements_research.txt
```

## Data Setup

### Required Files:

1. **Hyperspectral data**: Provide your own hyperspectral imagery file
   - **Note:** This project was developed using NYU TuckMapping dataset, which is classified and restricted to NYU researchers only. This repository does not provide access to this dataset.
   - Format: ERDAS Imagine (.pix) or GeoTIFF (.tif)
   - Must contain multiple spectral bands
   - Update the path in `src/advanced_config.py` (HYPERSPECTRAL_PATH)

2. **Land Cover data**: NYC Land Cover Raster (2017) from NYC Open Data
   - Download from: https://data.cityofnewyork.us/Environment/Land-Cover-Raster-Data-2017-6in-Resolution/he6d-2qns/about_data
   - Format: ERDAS Imagine (.img) or GeoTIFF (.tif)
   - Update the path in `src/advanced_config.py` (LAND_COVER_PATH)

### Important: Identify Pavement Codes

Before running, you MUST identify the pavement class codes:
1. Open the land cover data in a GIS software (QGIS, ArcGIS)
2. Check the attribute table or documentation
3. Find codes for "Roads", "Other Impervious", or similar pavement classes
4. Update `PAVEMENT_CLASSES` in `src/advanced_config.py` (default: [6, 7])

## Usage

### Basic Pipeline

```bash
python src/main.py
```

### Enhanced Research Pipeline

```bash
python src/main_enhanced.py
```

## Configuration

Edit `src/advanced_config.py` to configure:
- Data file paths
- Test area size and location
- Classification parameters
- Ground truth settings

## Output

Results are saved to `outputs_research/` directory:
- `figures/` - Visualization images
- `results/` - Classification maps (GeoTIFF)
- `analysis/` - Metrics and statistics
