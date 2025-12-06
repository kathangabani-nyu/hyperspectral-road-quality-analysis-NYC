"""
Feature engineering for hyperspectral data.
Includes spectral indices and spatial features.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import config


def compute_ndvi(X, red_idx=50, nir_idx=70):
    """
    Compute Normalized Difference Vegetation Index (NDVI).
    NDVI = (NIR - Red) / (NIR + Red)
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix (n_samples, n_bands)
    red_idx : int
        Index of red band
    nir_idx : int
        Index of near-infrared band
        
    Returns:
    --------
    ndvi : ndarray
        NDVI values (n_samples,)
    """
    if red_idx >= X.shape[1] or nir_idx >= X.shape[1]:
        if config.VERBOSE:
            print(f"Warning: Cannot compute NDVI - band indices out of range")
        return np.zeros(X.shape[0])
    
    red = X[:, red_idx]
    nir = X[:, nir_idx]
    
    # Avoid division by zero
    denominator = nir + red
    ndvi = np.zeros_like(red)
    valid = denominator != 0
    ndvi[valid] = (nir[valid] - red[valid]) / denominator[valid]
    
    return ndvi


def compute_spectral_indices(X):
    """
    Compute various spectral indices.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix (n_samples, n_bands)
        
    Returns:
    --------
    indices : ndarray
        Spectral indices (n_samples, n_indices)
    """
    if not config.USE_SPECTRAL_INDICES:
        return np.array([]).reshape(X.shape[0], 0)
    
    if config.VERBOSE:
        print(f"\nComputing spectral indices...")
    
    indices_list = []
    
    # NDVI (if we have enough bands)
    if X.shape[1] >= 71:
        ndvi = compute_ndvi(X, red_idx=50, nir_idx=70)
        indices_list.append(ndvi.reshape(-1, 1))
        if config.VERBOSE:
            print(f"  - NDVI computed")
    
    # Mean reflectance
    mean_reflectance = X.mean(axis=1).reshape(-1, 1)
    indices_list.append(mean_reflectance)
    if config.VERBOSE:
        print(f"  - Mean reflectance computed")
    
    # Standard deviation of reflectance (spectral variability)
    std_reflectance = X.std(axis=1).reshape(-1, 1)
    indices_list.append(std_reflectance)
    if config.VERBOSE:
        print(f"  - Std reflectance computed")
    
    # Ratio of first and last band
    if X.shape[1] >= 2:
        ratio = (X[:, 0] / (X[:, -1] + 1e-10)).reshape(-1, 1)
        indices_list.append(ratio)
        if config.VERBOSE:
            print(f"  - Band ratio computed")
    
    if len(indices_list) == 0:
        return np.array([]).reshape(X.shape[0], 0)
    
    indices = np.hstack(indices_list)
    
    if config.VERBOSE:
        print(f"Total indices computed: {indices.shape[1]}")
    
    return indices


def apply_spatial_smoothing(data_3d, sigma=1.0):
    """
    Apply Gaussian smoothing to each band.
    
    Parameters:
    -----------
    data_3d : ndarray
        Hyperspectral data (bands, height, width)
    sigma : float
        Standard deviation for Gaussian kernel
        
    Returns:
    --------
    smoothed : ndarray
        Smoothed data (bands, height, width)
    """
    if not config.USE_SPATIAL_SMOOTHING:
        return data_3d
    
    if config.VERBOSE:
        print(f"\nApplying spatial smoothing (sigma={sigma})...")
    
    smoothed = np.zeros_like(data_3d)
    
    for i in range(data_3d.shape[0]):
        smoothed[i] = gaussian_filter(data_3d[i], sigma=sigma)
    
    if config.VERBOSE:
        print(f"Spatial smoothing applied to {data_3d.shape[0]} bands")
    
    return smoothed


def augment_features(X):
    """
    Augment feature matrix with additional derived features.
    
    Parameters:
    -----------
    X : ndarray
        Original feature matrix (n_samples, n_bands)
        
    Returns:
    --------
    X_augmented : ndarray
        Augmented feature matrix (n_samples, n_features)
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("FEATURE ENGINEERING")
        print(f"{'='*60}")
        print(f"Original features: {X.shape[1]}")
    
    features = [X]
    
    # Add spectral indices
    if config.USE_SPECTRAL_INDICES:
        indices = compute_spectral_indices(X)
        if indices.shape[1] > 0:
            features.append(indices)
    
    # Combine all features
    X_augmented = np.hstack(features)
    
    if config.VERBOSE:
        print(f"Augmented features: {X_augmented.shape[1]}")
        print(f"Added {X_augmented.shape[1] - X.shape[1]} derived features")
    
    return X_augmented


def compute_band_statistics(X):
    """
    Compute statistics for each band across all samples.
    Useful for understanding data distribution.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix (n_samples, n_bands)
        
    Returns:
    --------
    stats : dict
        Dictionary with band statistics
    """
    stats = {
        'mean': X.mean(axis=0),
        'std': X.std(axis=0),
        'min': X.min(axis=0),
        'max': X.max(axis=0),
        'median': np.median(X, axis=0)
    }
    
    return stats

