# Quick Start Instructions

## Installation

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Setup

### Required Files:
1. **Hyperspectral data**: `
   - Download from Google Drive: https://drive.google.com/drive/u/0/folders/1PoHgH77VZTlXPSB-esOxD8XMSKeO8VfZ
   - Place in your project directory or specify path in the notebook

2. **Land Cover data**:  land cover raster from NYC Open Data
   - Download from: https://data.cityofnewyork.us/Environment/Land-Cover-Raster-Data-2017-6in-Resolution/he6d-2qns/about_data
   - Should be in GeoTIFF format (.tif)
   - Name it `land_cover_ or update the path in the code

### Important: Identify Pavement Codes
Before running, you MUST identify the pavement class codes:
1. Open the land cover data in a GIS software (QGIS, ArcGIS)
2. Check the attribute table or documentation
3. Find codes for "Impervious", "Paved", "Roads", or similar
4. Update line with `pavement_codes = [1]` in the notebook/script

## Usage

### Option 1: Jupyter Notebook (Recommended for exploration)
```bash
jupyter notebook pavement_classification.ipynb
```

### Option 2: Python Script
```bash
python pavement_classification.py
```

## Expected Outputs

1. **Classification map**: `pavement_classification_map.tif` - Binary classification (pavement vs non-pavement)
2. **Visualization**: `pavement_classification_results.png` - Side-by-side comparison of ground truth and results
3. **Console output**: Accuracy metrics and confusion matrix

## Troubleshooting

### "Cannot open .pix file"
- .pix files may require conversion to GeoTIFF
- Use GDAL: `gdal_translate -of GTiff input.pix output.tif`
- Update the path in your code

### "Memory error"
- Reduce the test area size (currently 500x500 pixels)
- Use fewer bands (currently using first 100 bands)
- Adjust `max_samples` in the training data preparation

### "Coordinate system mismatch"
- The data may have different CRS
- Use QGIS to reproject one dataset to match the other
- Or modify the window/extent parameters in the code

### "No pavement pixels found"
- Verify you have the correct pavement codes
- Check that the land cover data overlaps with the hyperspectral area
- Adjust the `pavement_codes` list

