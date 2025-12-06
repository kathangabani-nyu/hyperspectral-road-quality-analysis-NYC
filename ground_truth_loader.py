"""
Ground truth data loading and alignment for hyperspectral classification.

This module handles loading NYC Land Cover Raster (2017) data and aligning it
with hyperspectral imagery for use as ground truth labels.
"""

import numpy as np
import os
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.windows import Window, from_bounds
from rasterio.crs import CRS
import warnings
warnings.filterwarnings('ignore')

try:
    import advanced_config as config
except ImportError:
    import config


def load_land_cover_data(land_cover_path, window=None, transform=None, bounds=None, max_size=10000):
    """
    Load land cover data from ERDAS Imagine (.img) format.
    
    Parameters:
    -----------
    land_cover_path : str
        Path to land cover raster file (will try to resolve if not found)
    window : rasterio.windows.Window, optional
        Window to extract (in land cover coordinates)
    transform : Affine, optional
        Transform for window extraction
    bounds : tuple, optional
        (minx, miny, maxx, maxy) bounds for extraction
        
    Returns:
    --------
    land_cover_array : ndarray
        Land cover data (height, width)
    profile : dict
        Rasterio profile
    transform : Affine
        Geospatial transform
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("LOADING LAND COVER GROUND TRUTH DATA")
        print(f"{'='*60}")
        print(f"File: {land_cover_path}")
    
    # Try to resolve path if it doesn't exist
    if not os.path.exists(land_cover_path):
        import glob
        # Try to find the file
        filename = os.path.basename(land_cover_path)
        matches = glob.glob(f"**/{filename}", recursive=True)
        if matches:
            land_cover_path = matches[0]
            if config.VERBOSE:
                print(f"Resolved path to: {land_cover_path}")
        else:
            raise FileNotFoundError(f"Land cover file not found: {land_cover_path}")
    
    with rasterio.open(land_cover_path) as src:
        if config.VERBOSE:
            print(f"Land Cover Dataset Info:")
            print(f"  Size: {src.width} x {src.height} pixels")
            print(f"  CRS: {src.crs}")
            print(f"  Bounds: {src.bounds}")
            print(f"  Transform: {src.transform}")
            print(f"  Data type: {src.dtypes[0]}")
        
        # Extract window if provided
        if bounds is not None:
            try:
                # Create window from bounds
                window = from_bounds(*bounds, src.transform)
            except Exception as e:
                if config.VERBOSE:
                    print(f"Warning: Could not create window from bounds: {e}")
                    print(f"  Using full dataset instead")
                window = None
        
        if window is not None:
            if config.VERBOSE:
                print(f"\nExtracting window: {window}")
            # Limit window size to avoid memory issues
            if window.width > max_size or window.height > max_size:
                if config.VERBOSE:
                    print(f"Window too large ({window.width}x{window.height}), limiting to {max_size}x{max_size}")
                window = Window(
                    window.col_off,
                    window.row_off,
                    min(window.width, max_size),
                    min(window.height, max_size)
                )
            data = src.read(1, window=window)
            transform = src.window_transform(window)
        else:
            # Don't load full dataset - it's too large
            # Instead, load a reasonable sample or raise error
            if src.width * src.height > max_size * max_size:
                raise ValueError(
                    f"Land cover dataset is too large ({src.width}x{src.height}). "
                    f"Please provide a window parameter to extract a subset."
                )
            data = src.read(1)
            transform = src.transform
        
        profile = src.profile.copy()
        profile.update({
            'height': data.shape[0],
            'width': data.shape[1],
            'count': 1,
            'transform': transform
        })
        
        if config.VERBOSE:
            print(f"\nLoaded land cover data:")
            print(f"  Shape: {data.shape}")
            print(f"  Unique values: {np.unique(data)}")
            print(f"  Data range: [{data.min()}, {data.max()}]")
    
    return data, profile, transform


def extract_pavement_classes(land_cover_array, pavement_classes=None):
    """
    Extract pavement classes from land cover data.
    
    Parameters:
    -----------
    land_cover_array : ndarray
        Land cover classification (height, width)
    pavement_classes : list, optional
        List of class IDs to include as pavement
        Default: [6, 7] (Roads + Other Impervious)
        
    Returns:
    --------
    binary_pavement_mask : ndarray
        Binary mask (1 = pavement, 0 = non-pavement)
    """
    if pavement_classes is None:
        pavement_classes = getattr(config, 'PAVEMENT_CLASSES', [6, 7])
    
    if config.VERBOSE:
        print(f"\nExtracting pavement classes: {pavement_classes}")
        print(f"  Class 6: Roads")
        print(f"  Class 7: Other Impervious (sidewalks, parking lots, etc.)")
    
    # Create binary mask
    binary_mask = np.zeros_like(land_cover_array, dtype=np.uint8)
    
    for class_id in pavement_classes:
        class_mask = (land_cover_array == class_id)
        binary_mask[class_mask] = 1
        
        if config.VERBOSE:
            n_pixels = class_mask.sum()
            print(f"  Class {class_id}: {n_pixels} pixels")
    
    total_pavement = binary_mask.sum()
    total_pixels = binary_mask.size
    
    if config.VERBOSE:
        print(f"\nPavement mask summary:")
        print(f"  Total pavement pixels: {total_pavement}")
        print(f"  Total pixels: {total_pixels}")
        print(f"  Pavement percentage: {100 * total_pavement / total_pixels:.2f}%")
    
    return binary_mask


def align_with_hyperspectral(land_cover_data, land_cover_profile, 
                             hyperspectral_profile, hyperspectral_transform,
                             hyperspectral_bounds=None):
    """
    Align land cover data with hyperspectral data.
    
    Handles:
    - Coordinate system reprojection
    - Resolution resampling
    - Spatial window extraction
    
    Parameters:
    -----------
    land_cover_data : ndarray
        Land cover array (height, width)
    land_cover_profile : dict
        Land cover rasterio profile
    hyperspectral_profile : dict
        Hyperspectral rasterio profile
    hyperspectral_transform : Affine
        Hyperspectral transform
    hyperspectral_bounds : tuple, optional
        (minx, miny, maxx, maxy) bounds of hyperspectral window
        
    Returns:
    --------
    aligned_data : ndarray
        Aligned land cover data matching hyperspectral dimensions
    aligned_profile : dict
        Updated profile matching hyperspectral
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("ALIGNING LAND COVER WITH HYPERSPECTRAL DATA")
        print(f"{'='*60}")
    
    lc_crs = land_cover_profile['crs']
    hs_crs = hyperspectral_profile['crs']
    hs_height = hyperspectral_profile['height']
    hs_width = hyperspectral_profile['width']
    
    if config.VERBOSE:
        print(f"Land Cover CRS: {lc_crs}")
        print(f"Hyperspectral CRS: {hs_crs}")
        print(f"Target size: {hs_width} x {hs_height} pixels")
    
    # Check if reprojection is needed
    needs_reprojection = (lc_crs != hs_crs)
    
    # Always reproject/resample to match hyperspectral dimensions and CRS
    # This ensures proper alignment regardless of original CRS
    if True:  # Always do alignment
        if config.VERBOSE:
            print(f"\nReprojecting land cover from {lc_crs} to {hs_crs}...")
        
        # Calculate transform for reprojection
        if hyperspectral_bounds is not None:
            dst_bounds = hyperspectral_bounds
        else:
            # Calculate bounds from transform and dimensions
            from rasterio.transform import xy
            minx, miny = xy(hyperspectral_transform, 0, hs_height)
            maxx, maxy = xy(hyperspectral_transform, hs_width, 0)
            dst_bounds = (minx, miny, maxx, maxy)
        
        # Reproject/resample to match hyperspectral CRS and dimensions
        if config.VERBOSE:
            if needs_reprojection:
                print(f"Reprojecting from {lc_crs} to {hs_crs}...")
            else:
                print(f"Resampling to match hyperspectral dimensions...")
        
        reprojected_data = np.zeros((hs_height, hs_width), dtype=land_cover_data.dtype)
        
        reproject(
            source=land_cover_data,
            destination=reprojected_data,
            src_transform=land_cover_profile['transform'],
            src_crs=lc_crs,
            dst_transform=hyperspectral_transform,
            dst_crs=hs_crs,
            resampling=Resampling.nearest  # Nearest neighbor for categorical data
        )
        
        aligned_data = reprojected_data
    
    # Update profile
    aligned_profile = hyperspectral_profile.copy()
    aligned_profile.update({
        'count': 1,
        'dtype': aligned_data.dtype,
        'crs': hs_crs,
        'transform': hyperspectral_transform
    })
    
    if config.VERBOSE:
        print(f"\nAlignment complete:")
        print(f"  Aligned shape: {aligned_data.shape}")
        print(f"  Unique values: {np.unique(aligned_data)}")
    
    return aligned_data, aligned_profile


def load_and_align_ground_truth(land_cover_path, hyperspectral_profile, 
                                 hyperspectral_transform, hyperspectral_bounds=None,
                                 pavement_classes=None):
    """
    Convenience function to load and align ground truth in one step.
    
    Parameters:
    -----------
    land_cover_path : str
        Path to land cover raster
    hyperspectral_profile : dict
        Hyperspectral profile
    hyperspectral_transform : Affine
        Hyperspectral transform
    hyperspectral_bounds : tuple, optional
        Bounds of hyperspectral window
    pavement_classes : list, optional
        Pavement class IDs
        
    Returns:
    --------
    pavement_mask : ndarray
        Binary pavement mask aligned with hyperspectral data
    land_cover_classes : ndarray
        Full land cover classification (for reference)
    """
    # Calculate the window we need from land cover data
    # First, we need to find the matching area in land cover coordinates
    # We'll use reproject to extract the matching window directly
    
    # Open land cover to get its info
    with rasterio.open(land_cover_path) as lc_src:
        lc_crs = lc_src.crs
        lc_transform = lc_src.transform
        
        # Calculate target bounds in hyperspectral CRS
        hs_height = hyperspectral_profile['height']
        hs_width = hyperspectral_profile['width']
        
        from rasterio.transform import xy
        if hyperspectral_bounds is None:
            minx, miny = xy(hyperspectral_transform, 0, hs_height)
            maxx, maxy = xy(hyperspectral_transform, hs_width, 0)
            hyperspectral_bounds = (minx, miny, maxx, maxy)
        
        # Reproject bounds to land cover CRS to find matching window
        from rasterio.warp import transform_bounds
        lc_bounds = transform_bounds(
            hyperspectral_profile['crs'],
            lc_crs,
            *hyperspectral_bounds
        )
        
        # Create window in land cover coordinates
        try:
            lc_window = from_bounds(*lc_bounds, lc_transform)
            # Expand window slightly to ensure coverage
            lc_window = Window(
                max(0, int(lc_window.col_off) - 10),
                max(0, int(lc_window.row_off) - 10),
                min(lc_src.width - int(lc_window.col_off), int(lc_window.width) + 20),
                min(lc_src.height - int(lc_window.row_off), int(lc_window.height) + 20)
            )
        except Exception:
            # If window creation fails, use a reasonable subset
            lc_window = Window(0, 0, min(5000, lc_src.width), min(5000, lc_src.height))
    
    # Load only the window from land cover
    land_cover_data, lc_profile, lc_transform = load_land_cover_data(
        land_cover_path,
        window=lc_window
    )
    
    # Align with hyperspectral
    aligned_land_cover, aligned_profile = align_with_hyperspectral(
        land_cover_data,
        lc_profile,
        hyperspectral_profile,
        hyperspectral_transform,
        hyperspectral_bounds
    )
    
    # Extract pavement classes
    pavement_mask = extract_pavement_classes(aligned_land_cover, pavement_classes)
    
    return pavement_mask, aligned_land_cover

