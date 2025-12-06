# Sidewalk Detection and Manual Masking Guide

This guide explains how to use the new sidewalk-specific detection and manual masking features to improve pavement classification and exclude building rooftops.

## Overview

Two new features have been added:

1. **Manual Masking Tool** - Interactively mark pavement areas
2. **Sidewalk-Specific Filtering** - Automatically focus on sidewalks (narrow linear features) while excluding rooftops

## Manual Masking Tool

### Usage

Run the manual masking tool:

```bash
python manual_mask_tool.py
```

### Features

- **Interactive Drawing**: Click and drag to mark pavement areas
- **Add/Remove Modes**: Switch between adding and removing pavement marks
- **Adjustable Brush Size**: Use +/- buttons to change brush size
- **Visual Feedback**: Red overlay shows marked pavement areas
- **Save/Load**: Save your mask and load it later to continue editing

### Workflow

1. Launch the tool: `python manual_mask_tool.py`
2. The RGB composite image will be displayed
3. Click "Add" mode to mark pavement areas
4. Click and drag to draw on the image
5. Use "Remove" mode to unmark incorrect areas
6. Adjust brush size as needed
7. Click "Save Mask" when done
8. The mask will be saved to `outputs_research/manual_pavement_mask.npy`

### Integration

To use the manual mask in the main pipeline:

1. Edit `advanced_config.py`:
   ```python
   USE_MANUAL_MASK = True
   ```

2. Run the enhanced pipeline:
   ```bash
   python main_enhanced.py
   ```

The manual mask will be integrated into the classification results.

## Sidewalk-Specific Filtering

### Purpose

This feature focuses on detecting sidewalks (narrow linear features) while aggressively filtering out building rooftops using geometric properties.

### Key Characteristics

**Sidewalks:**
- Narrow (1-8 pixels wide at typical resolution)
- Elongated (high length/width ratio)
- Form connected paths
- Often adjacent to roads

**Building Rooftops:**
- Square/rectangular (low aspect ratio)
- Isolated (not connected to road network)
- Medium to large size
- High compactness (fill their bounding box)

### Configuration

Edit `advanced_config.py`:

```python
USE_SIDEWALK_FILTERING = True  # Enable sidewalk-specific filtering
SIDEWALK_MIN_WIDTH = 1         # Minimum sidewalk width (pixels)
SIDEWALK_MAX_WIDTH = 8         # Maximum sidewalk width (pixels)
SIDEWALK_MIN_LENGTH = 10       # Minimum sidewalk length (pixels)
USE_ROAD_ADJACENCY = True      # Use road adjacency information
```

### How It Works

1. **Aggressive Rooftop Filtering**: Removes square/rectangular regions with high compactness
2. **Narrow Linear Feature Detection**: Identifies features within the width range (sidewalks)
3. **Road Adjacency**: Optionally identifies pavement adjacent to roads
4. **Geometric Constraints**: Applies length/width ratio constraints to validate sidewalks

### Usage

Simply enable the feature in `advanced_config.py` and run:

```bash
python main_enhanced.py
```

## Combining Both Features

You can use both features together:

1. **First**: Create a manual mask to mark known sidewalk areas
   ```bash
   python manual_mask_tool.py
   ```

2. **Then**: Enable both features in `advanced_config.py`:
   ```python
   USE_MANUAL_MASK = True
   USE_SIDEWALK_FILTERING = True
   ```

3. **Run**: Execute the enhanced pipeline
   ```bash
   python main_enhanced.py
   ```

The pipeline will:
- Apply sidewalk-specific filtering to focus on narrow linear features
- Integrate your manual mask to ensure known sidewalk areas are included
- Aggressively filter rooftops using geometric properties

## Tips for Best Results

### Manual Masking

- **Start Small**: Mark a few clear sidewalk areas first
- **Use Zoom**: If the image is large, focus on specific regions
- **Iterate**: Run the pipeline, check results, and refine the mask
- **Brush Size**: Use smaller brushes (1-3 pixels) for precise marking

### Sidewalk Filtering

- **Adjust Width Range**: If sidewalks are being missed, increase `SIDEWALK_MAX_WIDTH`
- **Adjust Length**: If too many small features are detected, increase `SIDEWALK_MIN_LENGTH`
- **Road Adjacency**: Enable `USE_ROAD_ADJACENCY` if sidewalks are typically next to roads
- **Iterate**: Fine-tune parameters based on your specific data

## Output Files

- **Manual Mask**: `outputs_research/manual_pavement_mask.npy`
- **Mask Visualization**: `outputs_research/manual_pavement_mask_visualization.png`
- **Final Classification**: `outputs_research/results/pavement_classification_advanced.tif`
- **Classification Visualization**: `outputs_research/figures/classification_results_enhanced.png`

## Troubleshooting

### Manual Mask Not Being Used

- Check that `USE_MANUAL_MASK = True` in `advanced_config.py`
- Verify the mask file exists: `outputs_research/manual_pavement_mask.npy`
- Ensure the mask dimensions match the image dimensions

### Too Many/Few Sidewalks Detected

- Adjust `SIDEWALK_MIN_WIDTH` and `SIDEWALK_MAX_WIDTH`
- Modify `SIDEWALK_MIN_LENGTH` to filter small features
- Toggle `USE_ROAD_ADJACENCY` on/off

### Rooftops Still Appearing

- The sidewalk filter should remove most rooftops
- If some remain, they may be:
  - Very large (excluded by max_area)
  - Very small (excluded by min_area)
  - Elongated (not square/rectangular)
- Consider manually marking these as non-pavement in the manual mask tool

