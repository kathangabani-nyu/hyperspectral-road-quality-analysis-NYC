# Quick Start Guide

## Running the Classification Pipeline

### Option 1: Using the Batch Script (Windows)
Double-click `run_classification.bat` or run from command line:
```bash
run_classification.bat
```

This will:
- Activate the virtual environment
- Run the classification pipeline
- Open the results folder when complete

### Option 2: Manual Execution

1. **Activate virtual environment:**
```bash
venv\Scripts\activate
```

2. **Run the pipeline:**
```bash
python main.py
```

3. **View results:**
Results are saved in the `outputs/` directory:
- `outputs/figures/` - Visualizations (PNG images)
- `outputs/results/` - GeoTIFF maps and metrics

---

## First Time Setup

If you haven't set up the environment yet:

1. **Create virtual environment:**
```bash
python -m venv venv
```

2. **Activate it:**
```bash
venv\Scripts\activate
```

3. **Install requirements:**
```bash
pip install -r requirements.txt
```

4. **Run the pipeline:**
```bash
python main.py
```

---

## Customizing Parameters

Edit `config.py` to change:
- Test area size and location
- Number of bands to use
- RGB visualization bands
- Classification parameters
- Output paths

Example changes in `config.py`:
```python
# Change test area location
START_X = 1000
START_Y = 2000

# Change area size
TEST_SIZE_X = 800
TEST_SIZE_Y = 800

# Use different bands for RGB
RED_BAND_IDX = 80
GREEN_BAND_IDX = 50
BLUE_BAND_IDX = 10

# Adjust number of clusters
N_CLUSTERS = 7
```

Save `config.py` and run `python main.py` again.

---

## Checking Results

### 1. RGB Composite
File: `outputs/figures/rgb_composite.png`
- Verify this shows a clear image (NOT black)
- Check that the area looks correct

### 2. Clustering Results
File: `outputs/figures/clustering_results.png`
- Left: RGB composite
- Right: Cluster map with different colors for each cluster
- Identify which clusters represent pavement (roads, parking lots)

### 3. Classification Results
File: `outputs/figures/classification_results.png`
- Left: RGB composite
- Middle: Reference labels from clustering
- Right: Final classification

### 4. Metrics
File: `outputs/results/classification_metrics.txt`
- Check accuracy, precision, recall for each class
- Look for balanced performance

### 5. Spectral Signatures
File: `outputs/figures/spectral_signatures.png`
- Should show distinct curves for pavement vs. non-pavement
- If curves are identical, classification may be poor

---

## Troubleshooting

### Black or Dark Images
**Problem**: RGB composite is too dark to see

**Solutions**:
1. Try different band indices in `config.py`:
   ```python
   RED_BAND_IDX = 60
   GREEN_BAND_IDX = 30
   BLUE_BAND_IDX = 10
   ```

2. Adjust contrast stretch percentiles:
   ```python
   RGB_PERCENTILE_LOW = 1
   RGB_PERCENTILE_HIGH = 99
   ```

### Poor Classification Accuracy
**Problem**: Low accuracy or one class has 0% precision

**Solutions**:
1. Check if pavement clusters were correctly identified in clustering results
2. Try a different test area with clearer pavement examples
3. Increase number of clusters:
   ```python
   N_CLUSTERS = 8
   ```
4. Enable spectral indices:
   ```python
   USE_SPECTRAL_INDICES = True
   ```

### Out of Memory Error
**Problem**: Script crashes with memory error

**Solutions**:
1. Reduce test area size:
   ```python
   TEST_SIZE_X = 300
   TEST_SIZE_Y = 300
   ```

2. Use fewer bands:
   ```python
   NUM_BANDS = 50
   ```

3. Reduce clustering samples:
   ```python
   MAX_CLUSTERING_SAMPLES = 20000
   ```

### Script Takes Too Long
**Problem**: Processing is very slow

**Solutions**:
1. Start with a smaller test area (e.g., 200x200)
2. Reduce number of bands
3. Reduce Random Forest trees:
   ```python
   RF_N_ESTIMATORS = 50
   ```

---

## Processing Different Areas

To classify a different region of the hyperspectral image:

1. **Find coordinates**: Determine the pixel coordinates you want to process

2. **Update config.py**:
   ```python
   START_X = 2000  # Your X coordinate
   START_Y = 3000  # Your Y coordinate
   TEST_SIZE_X = 500
   TEST_SIZE_Y = 500
   ```

3. **Run again**:
   ```bash
   python main.py
   ```

4. **Compare results**: Check if classification quality is consistent

---

## Processing Full Image

⚠️ **Warning**: Processing the full 6419x8001 pixel image will require significant memory and time.

**Recommended approach**:

1. **Test first**: Verify good results on 500x500 area

2. **Process in tiles**: Modify `main.py` to loop over tiles:
   ```python
   tile_size = 1000
   for x in range(0, full_width, tile_size):
       for y in range(0, full_height, tile_size):
           # Process tile at (x, y)
           # Save results
   ```

3. **Mosaic results**: Combine tiles into final classification map

---

## Next Steps

1. ✅ Run initial classification
2. ✅ Review all output figures
3. ✅ Check classification metrics
4. ✅ Adjust parameters if needed
5. ✅ Re-run and compare
6. ✅ When satisfied, process larger areas
7. ✅ Export final results for GIS

For detailed information, see `IMPROVEMENT_PLAN.md`.

---

## Getting Help

If you encounter issues:

1. Check the console output for error messages
2. Look for clues in `outputs/processing_log.txt`
3. Review `IMPROVEMENT_PLAN.md` for technical details
4. Try simplifying parameters (smaller area, fewer bands)

Common issues are usually:
- Wrong file paths
- Insufficient memory
- Incorrect band indices
- Poor cluster selection

