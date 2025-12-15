"""
Advanced feature engineering for hyperspectral data.

Implements:
- Comprehensive spectral indices (vegetation, urban, brightness)
- Spatial features (local statistics, texture)
- Statistical features (higher-order moments)
- Band ratios and derivatives
"""

import numpy as np
from scipy.ndimage import uniform_filter, generic_filter
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import advanced_config as config


# ============================================================================
# SPECTRAL INDICES
# ============================================================================

def compute_ndvi(X, red_idx=50, nir_idx=70):
    """
    Normalized Difference Vegetation Index.
    NDVI = (NIR - Red) / (NIR + Red)
    """
    if red_idx >= X.shape[1] or nir_idx >= X.shape[1]:
        return np.zeros(X.shape[0])
    
    red = X[:, red_idx]
    nir = X[:, nir_idx]
    denominator = nir + red
    ndvi = np.zeros_like(red)
    valid = denominator != 0
    ndvi[valid] = (nir[valid] - red[valid]) / denominator[valid]
    return ndvi.reshape(-1, 1)


def compute_evi(X, red_idx=50, nir_idx=70, blue_idx=20):
    """
    Enhanced Vegetation Index.
    EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    """
    if max(red_idx, nir_idx, blue_idx) >= X.shape[1]:
        return np.zeros(X.shape[0])
    
    red = X[:, red_idx]
    nir = X[:, nir_idx]
    blue = X[:, blue_idx]
    denominator = nir + 6*red - 7.5*blue + 1
    evi = np.zeros_like(red)
    valid = denominator != 0
    evi[valid] = 2.5 * (nir[valid] - red[valid]) / denominator[valid]
    return evi.reshape(-1, 1)


def compute_ndbi(X, nir_idx=70, swir_idx=150):
    """
    Normalized Difference Built-up Index.
    NDBI = (SWIR - NIR) / (SWIR + NIR)
    Useful for identifying urban/built areas.
    """
    if max(nir_idx, swir_idx) >= X.shape[1]:
        return np.zeros(X.shape[0])
    
    nir = X[:, nir_idx]
    swir = X[:, swir_idx]
    denominator = swir + nir
    ndbi = np.zeros_like(nir)
    valid = denominator != 0
    ndbi[valid] = (swir[valid] - nir[valid]) / denominator[valid]
    return ndbi.reshape(-1, 1)


def compute_ui(X, nir_idx=70, swir_idx=150):
    """
    Urban Index.
    UI = (SWIR - NIR) / (SWIR + NIR)
    Similar to NDBI, emphasizes urban areas.
    """
    return compute_ndbi(X, nir_idx, swir_idx)


def compute_brightness(X):
    """
    Brightness - mean reflectance across all bands.
    """
    return X.mean(axis=1).reshape(-1, 1)


def compute_greenness(X, green_idx=40, red_idx=50):
    """
    Greenness - ratio of green to red.
    """
    if max(green_idx, red_idx) >= X.shape[1]:
        return np.zeros(X.shape[0])
    
    green = X[:, green_idx]
    red = X[:, red_idx]
    greenness = np.zeros_like(green)
    valid = red != 0
    greenness[valid] = green[valid] / red[valid]
    return greenness.reshape(-1, 1)


def compute_spectral_slope(X):
    """
    Slope of spectral curve (linear regression across bands).
    Indicates overall trend in reflectance.
    """
    n_bands = X.shape[1]
    band_indices = np.arange(n_bands)
    
    slopes = []
    for i in range(X.shape[0]):
        spectrum = X[i, :]
        slope, _ = np.polyfit(band_indices, spectrum, 1)
        slopes.append(slope)
    
    return np.array(slopes).reshape(-1, 1)


def compute_absorption_depth(X, band_idx=50, window=5):
    """
    Depth of absorption feature around a specific band.
    """
    if band_idx >= X.shape[1]:
        return np.zeros(X.shape[0])
    
    start = max(0, band_idx - window)
    end = min(X.shape[1], band_idx + window + 1)
    
    local_min = X[:, start:end].min(axis=1)
    local_mean = X[:, start:end].mean(axis=1)
    
    depth = local_mean - local_min
    return depth.reshape(-1, 1)


def compute_concrete_index(X, visible_idx=50, nir_idx=70):
    """
    Concrete Index - distinguishes concrete from other materials.
    Concrete typically has: high visible reflectance, moderate NIR.
    CI = (Visible - NIR) / (Visible + NIR)
    """
    if max(visible_idx, nir_idx) >= X.shape[1]:
        return np.zeros(X.shape[0])
    
    visible = X[:, visible_idx]
    nir = X[:, nir_idx]
    denominator = visible + nir + 1e-10
    ci = (visible - nir) / denominator
    return ci.reshape(-1, 1)


def compute_asphalt_index(X, red_idx=50, swir_idx=150):
    """
    Asphalt Index - distinguishes asphalt from other materials.
    Asphalt typically has: low overall reflectance, specific red/SWIR ratio.
    AI = (Red - SWIR) / (Red + SWIR)
    If SWIR not available, uses NIR instead.
    """
    if red_idx >= X.shape[1]:
        return np.zeros(X.shape[0])
    
    red = X[:, red_idx]
    # Use SWIR if available, otherwise use a band in SWIR range
    if swir_idx >= X.shape[1]:
        # Use last band as proxy for SWIR
        swir_idx = X.shape[1] - 1
    
    swir = X[:, swir_idx]
    denominator = red + swir + 1e-10
    ai = (red - swir) / denominator
    return ai.reshape(-1, 1)


def compute_pavement_spectral_signature(X, red_idx=50, nir_idx=70, swir_idx=150):
    """
    Compute comprehensive pavement (sidewalk) spectral signature features.
    
    Sidewalks (concrete/asphalt) have distinct characteristics:
    - Low NDVI (not vegetation)
    - Moderate brightness (not shadow, not extreme)
    - Specific band ratios
    - Low spectral variance (uniform surface)
    - Specific absorption features
    
    Returns multiple features that help identify sidewalks.
    """
    features = []
    
    # 1. NDVI (should be low for pavement)
    ndvi = compute_ndvi(X, red_idx, nir_idx)
    features.append(ndvi)
    
    # 2. Brightness (mean reflectance)
    brightness = compute_brightness(X)
    features.append(brightness)
    
    # 3. Spectral variance (should be low for uniform pavement)
    spectral_std = X.std(axis=1).reshape(-1, 1)
    features.append(spectral_std)
    
    # 4. Concrete index
    ci = compute_concrete_index(X, visible_idx=red_idx, nir_idx=nir_idx)
    features.append(ci)
    
    # 5. Asphalt index (if SWIR available)
    if swir_idx < X.shape[1]:
        ai = compute_asphalt_index(X, red_idx=red_idx, swir_idx=swir_idx)
        features.append(ai)
    
    # 6. Visible/NIR ratio (pavement has specific ratio)
    if nir_idx < X.shape[1] and red_idx < X.shape[1]:
        visible_nir_ratio = (X[:, red_idx] / (X[:, nir_idx] + 1e-10)).reshape(-1, 1)
        features.append(visible_nir_ratio)
    
    # 7. Spectral slope (overall trend)
    slope = compute_spectral_slope(X)
    features.append(slope)
    
    # 8. Coefficient of variation (uniformity measure)
    cv = (X.std(axis=1) / (X.mean(axis=1) + 1e-10)).reshape(-1, 1)
    features.append(cv)
    
    return np.hstack(features)


# ============================================================================
# STATISTICAL FEATURES
# ============================================================================

def compute_statistical_features(X):
    """
    Compute statistical features for each pixel's spectrum.
    Returns: mean, std, skewness, kurtosis, min, max, range
    """
    features = []
    
    # Mean and std
    features.append(X.mean(axis=1).reshape(-1, 1))
    features.append(X.std(axis=1).reshape(-1, 1))
    
    # Skewness and kurtosis
    features.append(skew(X, axis=1).reshape(-1, 1))
    features.append(kurtosis(X, axis=1).reshape(-1, 1))
    
    # Min, max, range
    features.append(X.min(axis=1).reshape(-1, 1))
    features.append(X.max(axis=1).reshape(-1, 1))
    features.append((X.max(axis=1) - X.min(axis=1)).reshape(-1, 1))
    
    # Coefficient of variation
    cv = X.std(axis=1) / (X.mean(axis=1) + 1e-10)
    features.append(cv.reshape(-1, 1))
    
    return np.hstack(features)


# ============================================================================
# SPATIAL FEATURES
# ============================================================================

def compute_spatial_mean(data_3d, window_size=3):
    """
    Compute local spatial mean in a window around each pixel.
    
    Parameters:
    -----------
    data_3d : ndarray
        3D array (bands, height, width)
    window_size : int
        Size of spatial window
        
    Returns:
    --------
    spatial_features : ndarray
        Flattened spatial mean features
    """
    n_bands, height, width = data_3d.shape
    spatial_means = np.zeros_like(data_3d)
    
    for b in range(n_bands):
        spatial_means[b] = uniform_filter(data_3d[b], size=window_size, mode='reflect')
    
    # Reshape to (n_pixels, n_bands)
    spatial_means = spatial_means.transpose(1, 2, 0).reshape(-1, n_bands)
    
    return spatial_means


def compute_spatial_std(data_3d, window_size=3):
    """
    Compute local spatial standard deviation.
    """
    n_bands, height, width = data_3d.shape
    spatial_stds = np.zeros_like(data_3d)
    
    for b in range(n_bands):
        # Compute local std using a sliding window
        local_mean = uniform_filter(data_3d[b], size=window_size, mode='reflect')
        local_mean_sq = uniform_filter(data_3d[b]**2, size=window_size, mode='reflect')
        spatial_stds[b] = np.sqrt(np.maximum(local_mean_sq - local_mean**2, 0))
    
    spatial_stds = spatial_stds.transpose(1, 2, 0).reshape(-1, n_bands)
    
    return spatial_stds


def compute_spatial_features(data_3d, window_size=3):
    """
    Compute spatial statistics for a subset of bands.
    
    To avoid memory issues, only compute for key bands or PCA components.
    """
    if config.VERBOSE:
        print(f"\nComputing spatial features (window={window_size})...")
    
    # Use only a subset of bands to save computation
    n_bands = min(10, data_3d.shape[0])  # Use first 10 bands or all if less
    indices = np.linspace(0, data_3d.shape[0]-1, n_bands, dtype=int)
    
    spatial_features = []
    
    # Spatial mean
    spatial_mean = compute_spatial_mean(data_3d[indices], window_size)
    spatial_features.append(spatial_mean)
    
    # Spatial std
    spatial_std = compute_spatial_std(data_3d[indices], window_size)
    spatial_features.append(spatial_std)
    
    result = np.hstack(spatial_features)
    
    if config.VERBOSE:
        print(f"Spatial features shape: {result.shape}")
    
    return result


# ============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# ============================================================================

def extract_advanced_features(X, data_3d=None):
    """
    Extract comprehensive advanced features.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix (n_samples, n_bands)
    data_3d : ndarray, optional
        3D hyperspectral data (bands, height, width) for spatial features
        
    Returns:
    --------
    X_augmented : ndarray
        Augmented feature matrix with all derived features
    feature_names : list
        Names of all features
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("ADVANCED FEATURE ENGINEERING")
        print(f"{'='*60}")
        print(f"Original spectral bands: {X.shape[1]}")
    
    features = [X]
    feature_names = [f'band_{i}' for i in range(X.shape[1])]
    
    # Determine band indices based on number of bands
    n_bands = X.shape[1]
    
    # Approximate indices for different spectral regions
    blue_idx = int(n_bands * 0.15)
    green_idx = int(n_bands * 0.25)
    red_idx = int(n_bands * 0.35)
    nir_idx = int(n_bands * 0.45) if n_bands > 50 else min(n_bands-1, 70)
    swir_idx = int(n_bands * 0.70) if n_bands > 100 else min(n_bands-1, 150)
    
    # Spectral indices
    if config.USE_SPECTRAL_INDICES:
        if config.VERBOSE:
            print("\nComputing spectral indices...")
        
        # Vegetation indices
        if config.USE_VEGETATION_INDICES:
            features.append(compute_ndvi(X, red_idx, nir_idx))
            feature_names.append('NDVI')
            
            features.append(compute_evi(X, red_idx, nir_idx, blue_idx))
            feature_names.append('EVI')
            
            features.append(compute_greenness(X, green_idx, red_idx))
            feature_names.append('Greenness')
        
        # Urban indices
        if config.USE_URBAN_INDICES:
            features.append(compute_ndbi(X, nir_idx, swir_idx))
            feature_names.append('NDBI')
            
            features.append(compute_ui(X, nir_idx, swir_idx))
            feature_names.append('UI')
        
        # Brightness indices
        if config.USE_BRIGHTNESS_INDICES:
            features.append(compute_brightness(X))
            feature_names.append('Brightness')
            
            features.append(compute_spectral_slope(X))
            feature_names.append('Spectral_Slope')
            
            features.append(compute_absorption_depth(X, red_idx))
            feature_names.append('Absorption_Depth')
        
        # Sidewalk-specific spectral signature features
        if getattr(config, 'USE_PAVEMENT_SPECTRAL_SIGNATURE', True):
            if config.VERBOSE:
                print("\nComputing sidewalk-specific spectral signature...")
            pavement_features = compute_pavement_spectral_signature(X, red_idx, nir_idx, swir_idx)
            features.append(pavement_features)
            # Add feature names for pavement signature
            pavement_feature_names = ['NDVI_pavement', 'Brightness_pavement', 'Spectral_Std_pavement', 
                                     'Concrete_Index', 'Asphalt_Index', 'Visible_NIR_Ratio', 
                                     'Spectral_Slope_pavement', 'CV_pavement']
            # Adjust if SWIR not available
            if swir_idx >= n_bands:
                pavement_feature_names = pavement_feature_names[:4] + pavement_feature_names[5:]
            feature_names.extend(pavement_feature_names[:pavement_features.shape[1]])
            if config.VERBOSE:
                print(f"Added {pavement_features.shape[1]} sidewalk-specific spectral features")
        
        if config.VERBOSE:
            print(f"Added {len(feature_names) - X.shape[1]} spectral indices")
    
    # Statistical features
    if config.USE_STATISTICAL_FEATURES:
        if config.VERBOSE:
            print("\nComputing statistical features...")
        
        stat_features = compute_statistical_features(X)
        features.append(stat_features)
        feature_names.extend(['Mean', 'Std', 'Skewness', 'Kurtosis', 
                             'Min', 'Max', 'Range', 'CoefVar'])
        
        if config.VERBOSE:
            print(f"Added {stat_features.shape[1]} statistical features")
    
    # Spatial features
    if config.USE_SPATIAL_FEATURES and data_3d is not None:
        if config.VERBOSE:
            print("\nComputing spatial features...")
        
        spatial_features = compute_spatial_features(data_3d, config.SPATIAL_WINDOW_SIZE)
        features.append(spatial_features)
        
        n_spatial = spatial_features.shape[1]
        feature_names.extend([f'Spatial_{i}' for i in range(n_spatial)])
        
        if config.VERBOSE:
            print(f"Added {n_spatial} spatial features")
    
    # Combine all features
    X_augmented = np.hstack(features)
    
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print(f"Total features: {X.shape[1]} -> {X_augmented.shape[1]}")
        print(f"Feature augmentation: +{X_augmented.shape[1] - X.shape[1]} features "
              f"({X_augmented.shape[1]/X.shape[1]:.2f}x)")
        print(f"{'='*60}")
    
    return X_augmented, feature_names

