# Data Preparation Guide

## Required Data

### 1. Hyperspectral Data
- Format: ERDAS Imagine (.pix) or GeoTIFF (.tif)
- Must contain multiple spectral bands (288 bands recommended)
- Update path in `src/advanced_config.py`

### 2. NYC Land Cover Data
Download from NYC Open Data:
1. Go to: https://data.cityofnewyork.us/Environment/Land-Cover-Raster-Data-2017-6in-Resolution/he6d-2qns/about_data
2. Click "Export" or "Download"
3. Download the dataset
4. Format: ERDAS Imagine (.img) or GeoTIFF (.tif)

## Data Organization

Recommended directory structure:
```
project_root/
├── data/
│   ├── hyperspectral_data.pix  # Your hyperspectral imagery
│   └── land_cover_data.img      # NYC Land Cover Raster data
├── src/
└── outputs_research/
```

## Data Conversion

If you need to convert between formats:

### Using GDAL (command line):
```bash
gdal_translate -of GTiff input.pix output.tif
```

### Using Python:
```python
import rasterio

with rasterio.open('input.pix') as src:
    profile = src.profile
    profile.update(driver='GTiff')
    
    with rasterio.open('output.tif', 'w', **profile) as dst:
        dst.write(src.read())
```

## Coordinate System

Ensure your hyperspectral data and land cover data have compatible coordinate systems. The pipeline will handle reprojection automatically, but matching systems improve performance.

## Configuration

Update paths in `src/advanced_config.py`:
```python
HYPERSPECTRAL_PATH = "path/to/your/hyperspectral_data.pix"
LAND_COVER_PATH = "path/to/land_cover_data.img"
PAVEMENT_CLASSES = [6, 7]  # Roads and Other Impervious
```
