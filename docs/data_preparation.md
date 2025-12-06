# Data Preparation Guide

## Downloading Data

### 1. Hyperspectral Data
The hyperspectral data files are in Google Drive:
- Files: `NYU_TuckMapping_20190511_Mission1_NE-SW_288Bands.*`
- Format: PCI Geomatica (.pix)
- You need the .pix and .hdr files

### 2. NYC Land Cover Data
Download from NYC Open Data:
1. Go to: https://data.cityofnewyork.us/Environment/Land-Cover-Raster-Data-2017-6in-Resolution/he6d-2qns/about_data
2. Click "Export" or "Download"
3. Download the Brooklyn area or full city dataset
4. Look for the following file formats:
   - GeoTIFF (.tif)
   - Or other raster formats

## Data Conversion

If you need to convert the .pix file to GeoTIFF:

### Using GDAL (command line):
```bash
gdal_translate -of GTiff input.pix output.tif
```

### Using Python:
```python
import gdal

# Open .pix file
src_ds = gdal.Open("input.pix", gdal.GA_ReadOnly)

# Get all subdatasets (bands)
subdatasets = src_ds.GetSubDatasets()

# Convert (adjust based on your .pix structure)
# You may need to extract individual bands
```

### Alternative: Use ENVI or PCI Geomatica
- ENVI can read .pix files and export as GeoTIFF
- PCI Geomatica can export to various formats

## Identify Pavement Codes

Before running the notebook, you need to identify the pavement class codes in the land cover data:

1. Open the land cover raster in QGIS, ArcGIS, or similar
2. Check the attribute table or metadata
3. Look for codes like:
   - Impervious surfaces
   - Paved surfaces
   - Roads/pavement
4. Update the `pavement_codes` variable in the notebook

## File Organization

Organize your files like this:
```
VIP/
├── pavement_classification.ipynb
├── pavement_classification.py
├── requirements.txt
├── README.md
├── data/
│   ├── NYU_TuckMapping_20190511_Mission1_NE-SW_288Bands.pix
│   ├── NYU_TuckMapping_20190511_Mission1_NE-SW_288Bands.pix.hdr
│   ├── NYU_TuckMapping_20190511_Mission1_NE-SW_288Bands.pix.aux.xml
│   └── land_cover_brooklyn.tif
└── outputs/
    ├── pavement_classification_map.tif
    └── pavement_classification_results.png
```

## Coordinate System

The hyperspectral and land cover data may have different coordinate systems. You may need to:
1. Reproject one dataset to match the other
2. Adjust the transform parameters in the notebook

Use QGIS or Python with rasterio to check and adjust coordinate systems.

## Next Steps

1. Download both datasets
2. If needed, convert .pix to GeoTIFF
3. Identify pavement class codes from land cover metadata
4. Update paths in the notebook/script
5. Run the classification

