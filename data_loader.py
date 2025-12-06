"""
Data loading and preprocessing for hyperspectral imagery.
"""

import numpy as np
import rasterio
from rasterio.windows import Window
import config

# Try to use advanced_config if available (for enhanced pipeline)
try:
    import advanced_config
    # Use advanced_config if it has USE_ALL_BANDS (indicates it's being used)
    if hasattr(advanced_config, 'USE_ALL_BANDS'):
        config = advanced_config
except ImportError:
    pass  # Use basic config


def load_hyperspectral_data(window=None, band_subset=None):
    """
    Load hyperspectral data from file.
    
    Parameters:
    -----------
    window : rasterio.windows.Window, optional
        Window to read (subset of image)
    band_subset : list, optional
        List of band indices to read (1-indexed for rasterio)
        
    Returns:
    --------
    data : ndarray
        Hyperspectral data with shape (bands, height, width)
    profile : dict
        Rasterio profile metadata
    transform : Affine
        Geospatial transform for the window
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("LOADING HYPERSPECTRAL DATA")
        print(f"{'='*60}")
    
    with rasterio.open(config.HYPERSPECTRAL_PATH) as src:
        if config.VERBOSE:
            print(f"Dataset info:")
            print(f"  Total bands: {src.count}")
            print(f"  Full size: {src.width} x {src.height} pixels")
            print(f"  CRS: {src.crs}")
            print(f"  Data type: {src.dtypes[0]}")
        
        # Determine which bands to read
        if band_subset is None:
            # Check if USE_ALL_BANDS is set (from advanced_config)
            use_all_bands = getattr(config, 'USE_ALL_BANDS', False)
            
            if use_all_bands:
                # Use all bands
                band_subset = list(range(1, src.count + 1))
            elif config.NUM_BANDS is not None:
                band_subset = list(range(1, min(config.NUM_BANDS + 1, src.count + 1)))
            else:
                band_subset = list(range(1, src.count + 1))
        
        # Determine window to read
        if window is None:
            # Get window parameters (works with both config and advanced_config)
            start_x = getattr(config, 'START_X', 0)
            start_y = getattr(config, 'START_Y', 0)
            test_size_x = getattr(config, 'TEST_SIZE_X', src.width)
            test_size_y = getattr(config, 'TEST_SIZE_Y', src.height)
            
            window = Window(
                start_x, 
                start_y, 
                test_size_x, 
                test_size_y
            )
        
        if config.VERBOSE:
            print(f"\nReading data:")
            print(f"  Bands: {len(band_subset)} (indices {band_subset[0]} to {band_subset[-1]})")
            print(f"  Window: ({window.col_off}, {window.row_off}) "
                  f"size {window.width}x{window.height} pixels")
        
        # Read the data
        data = src.read(band_subset, window=window)
        profile = src.profile.copy()
        
        # Update profile for the window
        transform = src.window_transform(window)
        profile.update({
            'height': window.height,
            'width': window.width,
            'count': len(band_subset),
            'transform': transform
        })
        
        if config.VERBOSE:
            print(f"\nLoaded data shape: {data.shape}")
            print(f"Data range: {data.min()} to {data.max()}")
            print(f"Data type: {data.dtype}")
            
            # Check for invalid values
            n_nan = np.isnan(data).sum()
            n_inf = np.isinf(data).sum()
            if n_nan > 0:
                print(f"WARNING: {n_nan} NaN values found")
            if n_inf > 0:
                print(f"WARNING: {n_inf} Inf values found")
    
    return data, profile, transform


def reshape_for_ml(hyperspectral_data):
    """
    Reshape hyperspectral data for machine learning.
    
    Parameters:
    -----------
    hyperspectral_data : ndarray
        Shape (bands, height, width)
        
    Returns:
    --------
    X : ndarray
        Shape (n_pixels, n_bands)
    valid_mask : ndarray
        Boolean mask indicating valid pixels (no NaN/Inf)
    """
    n_bands = hyperspectral_data.shape[0]
    n_height = hyperspectral_data.shape[1]
    n_width = hyperspectral_data.shape[2]
    n_pixels = n_height * n_width
    
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("RESHAPING DATA FOR ML")
        print(f"{'='*60}")
        print(f"Original shape: {hyperspectral_data.shape}")
    
    # Reshape: (bands, height, width) -> (height, width, bands) -> (n_pixels, n_bands)
    X = hyperspectral_data.transpose(1, 2, 0).reshape(n_pixels, n_bands)
    
    if config.VERBOSE:
        print(f"Reshaped to: {X.shape}")
    
    # Create mask for valid pixels
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
    
    if config.VERBOSE:
        print(f"Valid pixels: {valid_mask.sum()} / {n_pixels} ({100*valid_mask.sum()/n_pixels:.1f}%)")
        
        if valid_mask.sum() < n_pixels:
            print(f"WARNING: {n_pixels - valid_mask.sum()} pixels contain invalid values")
    
    return X, valid_mask


def normalize_features(X, method='zscore'):
    """
    Normalize feature matrix.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix (n_samples, n_features)
    method : str
        'zscore' - zero mean, unit variance
        'minmax' - scale to [0, 1]
        
    Returns:
    --------
    X_normalized : ndarray
        Normalized features
    """
    if config.VERBOSE:
        print(f"\nNormalizing features using {method} method...")
    
    if method == 'zscore':
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        # Avoid division by zero
        std[std == 0] = 1.0
        X_normalized = (X - mean) / std
    elif method == 'minmax':
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        # Avoid division by zero
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        X_normalized = (X - min_val) / range_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if config.VERBOSE:
        print(f"Normalized data range: {X_normalized.min():.3f} to {X_normalized.max():.3f}")
    
    return X_normalized

