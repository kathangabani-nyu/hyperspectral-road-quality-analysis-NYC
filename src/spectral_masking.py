"""
Spectral masking for pavement classification.
Uses spectral characteristics to filter out obvious non-pavement areas.
"""

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, generic_filter
import advanced_config as config


def get_filtering_thresholds(strategy='moderate'):
    """
    Get thresholds based on filtering strategy.
    
    Parameters:
    -----------
    strategy : str
        Filtering strategy: 'conservative', 'moderate', or 'aggressive'
        
    Returns:
    --------
    thresholds : dict
        Dictionary with threshold values
    """
    strategies = {
        'conservative': {
            'ndvi_threshold': 0.4,  # Less aggressive
            'brightness_low_percentile': 2,
            'brightness_high_percentile': 98,
            'variance_percentile': 95
        },
        'moderate': {
            'ndvi_threshold': 0.3,
            'brightness_low_percentile': 5,
            'brightness_high_percentile': 99,
            'variance_percentile': 90
        },
        'aggressive': {
            'ndvi_threshold': 0.25,
            'brightness_low_percentile': 10,
            'brightness_high_percentile': 97,
            'variance_percentile': 85
        }
    }
    return strategies.get(strategy, strategies['moderate'])


def create_spectral_mask(X, data_3d=None):
    """
    Create mask based on spectral characteristics.
    
    Filters out:
    - High NDVI (vegetation)
    - Very low brightness (shadows/water)
    - Very high brightness (bright roofs)
    - High spectral variance (non-uniform surfaces)
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix (n_samples, n_bands)
    data_3d : ndarray, optional
        3D data for computing indices
        
    Returns:
    --------
    mask : ndarray
        Boolean mask (True = likely pavement, False = filter out)
    filter_stats : dict
        Dictionary with detailed filtering statistics
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("CREATING SPECTRAL MASK")
        print(f"{'='*60}")
    
    # Get filtering strategy and thresholds
    strategy = getattr(config, 'SPECTRAL_FILTERING_STRATEGY', 'moderate')
    strategy_thresholds = get_filtering_thresholds(strategy)
    
    # Use config values with strategy defaults as fallback
    ndvi_threshold = getattr(config, 'NDVI_THRESHOLD', strategy_thresholds['ndvi_threshold'])
    brightness_low_percentile = getattr(config, 'BRIGHTNESS_LOW_PERCENTILE', strategy_thresholds['brightness_low_percentile'])
    brightness_high_percentile = getattr(config, 'BRIGHTNESS_HIGH_PERCENTILE', strategy_thresholds['brightness_high_percentile'])
    variance_percentile = getattr(config, 'SPECTRAL_VARIANCE_PERCENTILE', strategy_thresholds['variance_percentile'])
    
    # Filter toggles
    use_ndvi_filter = getattr(config, 'USE_NDVI_FILTER', True)
    use_brightness_filter = getattr(config, 'USE_BRIGHTNESS_FILTER', True)
    use_variance_filter = getattr(config, 'USE_VARIANCE_FILTER', True)
    use_percentile_brightness = getattr(config, 'USE_PERCENTILE_BRIGHTNESS', True)
    use_percentile_variance = getattr(config, 'USE_PERCENTILE_VARIANCE', True)
    
    # Absolute thresholds (if not using percentiles)
    brightness_low_absolute = getattr(config, 'BRIGHTNESS_LOW_ABSOLUTE', None)
    brightness_high_absolute = getattr(config, 'BRIGHTNESS_HIGH_ABSOLUTE', None)
    variance_absolute = getattr(config, 'VARIANCE_ABSOLUTE_THRESHOLD', None)
    
    if config.VERBOSE:
        print(f"Filtering strategy: {strategy}")
        print(f"  NDVI threshold: {ndvi_threshold}")
        print(f"  Brightness low percentile: {brightness_low_percentile}")
        print(f"  Brightness high percentile: {brightness_high_percentile}")
        print(f"  Variance percentile: {variance_percentile}")
    
    n_bands = X.shape[1]
    original_count = X.shape[0]
    
    # Approximate band indices
    blue_idx = int(n_bands * 0.15)
    green_idx = int(n_bands * 0.25)
    red_idx = int(n_bands * 0.35)
    nir_idx = int(n_bands * 0.45) if n_bands > 50 else min(n_bands-1, 70)
    
    # Initialize mask (all True = keep all initially)
    mask = np.ones(X.shape[0], dtype=bool)
    filter_stats = {
        'original_count': original_count,
        'after_ndvi': original_count,
        'after_brightness_low': original_count,
        'after_brightness_high': original_count,
        'after_variance': original_count,
        'final_count': original_count,
        'ndvi_filtered': 0,
        'brightness_low_filtered': 0,
        'brightness_high_filtered': 0,
        'variance_filtered': 0,
        'ndvi_stats': {},
        'brightness_stats': {},
        'variance_stats': {}
    }
    
    initial_count = mask.sum()
    
    # 1. Filter high NDVI (vegetation)
    if use_ndvi_filter and nir_idx < n_bands and red_idx < n_bands:
        red = X[:, red_idx]
        nir = X[:, nir_idx]
        denominator = nir + red + 1e-10
        ndvi = (nir - red) / denominator
        
        mask = mask & (ndvi < ndvi_threshold)
        n_filtered = initial_count - mask.sum()
        filter_stats['ndvi_filtered'] = n_filtered
        filter_stats['after_ndvi'] = mask.sum()
        filter_stats['ndvi_stats'] = {
            'mean_filtered': float(ndvi[~mask].mean()) if (~mask).sum() > 0 else 0.0,
            'mean_kept': float(ndvi[mask].mean()) if mask.sum() > 0 else 0.0,
            'threshold': ndvi_threshold
        }
        
        if config.VERBOSE:
            print(f"  Filtered {n_filtered} pixels with NDVI >= {ndvi_threshold} (vegetation)")
            if filter_stats['ndvi_stats']['mean_filtered'] > 0:
                print(f"    Mean NDVI of filtered: {filter_stats['ndvi_stats']['mean_filtered']:.3f}")
                print(f"    Mean NDVI of kept: {filter_stats['ndvi_stats']['mean_kept']:.3f}")
        initial_count = mask.sum()
    
    # 2. Filter very low brightness (shadows, water)
    if use_brightness_filter:
        brightness = X.mean(axis=1)
        
        if use_percentile_brightness and brightness_low_absolute is None:
            if mask.sum() > 0:
                brightness_low = np.percentile(brightness[mask], brightness_low_percentile)
            else:
                brightness_low = np.percentile(brightness, brightness_low_percentile)
        else:
            brightness_low = brightness_low_absolute if brightness_low_absolute is not None else np.percentile(brightness, brightness_low_percentile)
        
        mask = mask & (brightness > brightness_low)
        n_filtered = initial_count - mask.sum()
        filter_stats['brightness_low_filtered'] = n_filtered
        filter_stats['after_brightness_low'] = mask.sum()
        filter_stats['brightness_stats']['low_threshold'] = float(brightness_low)
        filter_stats['brightness_stats']['mean_filtered_low'] = float(brightness[~mask].mean()) if (~mask).sum() > 0 else 0.0
        
        if config.VERBOSE:
            print(f"  Filtered {n_filtered} pixels with very low brightness (shadows/water)")
        initial_count = mask.sum()
        
        # 3. Filter very high brightness (bright roofs, concrete)
        if use_percentile_brightness and brightness_high_absolute is None:
            if mask.sum() > 0:
                brightness_extreme = np.percentile(brightness[mask], brightness_high_percentile)
            else:
                brightness_extreme = np.percentile(brightness, brightness_high_percentile)
        else:
            brightness_extreme = brightness_high_absolute if brightness_high_absolute is not None else np.percentile(brightness, brightness_high_percentile)
        
        mask = mask & (brightness < brightness_extreme)
        n_filtered = initial_count - mask.sum()
        filter_stats['brightness_high_filtered'] = n_filtered
        filter_stats['after_brightness_high'] = mask.sum()
        filter_stats['brightness_stats']['high_threshold'] = float(brightness_extreme)
        filter_stats['brightness_stats']['mean_filtered_high'] = float(brightness[~mask].mean()) if (~mask).sum() > 0 else 0.0
        
        if config.VERBOSE:
            print(f"  Filtered {n_filtered} pixels with extreme brightness (bright roofs)")
        initial_count = mask.sum()
    
    # 4. Filter based on spectral variance (pavement is relatively uniform)
    if use_variance_filter:
        spectral_std = X.std(axis=1)
        if mask.sum() > 0:
            if use_percentile_variance and variance_absolute is None:
                std_threshold = np.percentile(spectral_std[mask], variance_percentile)
            else:
                std_threshold = variance_absolute if variance_absolute is not None else np.percentile(spectral_std[mask], variance_percentile)
            
            mask = mask & (spectral_std < std_threshold)
            n_filtered = initial_count - mask.sum()
            filter_stats['variance_filtered'] = n_filtered
            filter_stats['after_variance'] = mask.sum()
            filter_stats['variance_stats'] = {
                'threshold': float(std_threshold),
                'mean_filtered': float(spectral_std[~mask].mean()) if (~mask).sum() > 0 else 0.0,
                'mean_kept': float(spectral_std[mask].mean()) if mask.sum() > 0 else 0.0
            }
            
            if config.VERBOSE:
                print(f"  Filtered {n_filtered} pixels with high spectral variance")
    
    filter_stats['final_count'] = mask.sum()
    
    # Enhanced logging with recommendations
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("SPECTRAL MASKING SUMMARY")
        print(f"{'='*60}")
        print(f"Original pixels: {original_count:,}")
        print(f"After NDVI filter: {filter_stats['after_ndvi']:,} ({100*filter_stats['after_ndvi']/original_count:.1f}%)")
        print(f"After brightness low filter: {filter_stats['after_brightness_low']:,} ({100*filter_stats['after_brightness_low']/original_count:.1f}%)")
        print(f"After brightness high filter: {filter_stats['after_brightness_high']:,} ({100*filter_stats['after_brightness_high']/original_count:.1f}%)")
        print(f"After variance filter: {filter_stats['after_variance']:,} ({100*filter_stats['after_variance']/original_count:.1f}%)")
        print(f"\nFilter Statistics:")
        print(f"  NDVI filter: Removed {filter_stats['ndvi_filtered']:,} pixels ({100*filter_stats['ndvi_filtered']/original_count:.1f}%)")
        print(f"  Brightness filters: Removed {filter_stats['brightness_low_filtered'] + filter_stats['brightness_high_filtered']:,} pixels ({100*(filter_stats['brightness_low_filtered'] + filter_stats['brightness_high_filtered'])/original_count:.1f}%)")
        print(f"  Variance filter: Removed {filter_stats['variance_filtered']:,} pixels ({100*filter_stats['variance_filtered']/original_count:.1f}%)")
        print(f"\nFinal result:")
        print(f"  Pixels kept: {mask.sum():,} ({100*mask.sum()/len(mask):.1f}%)")
        print(f"  Pixels filtered: {(~mask).sum():,} ({100*(~mask).sum()/len(mask):.1f}%)")
        
        # Recommendations
        pixels_kept_pct = 100 * mask.sum() / original_count
        print(f"\nRecommendations:")
        if pixels_kept_pct < 5:
            print(f"  - Current filtering keeps {pixels_kept_pct:.1f}% of pixels (very aggressive)")
            print(f"  - Consider using 'conservative' filtering strategy")
            print(f"  - Or increase NDVI threshold to 0.35-0.4")
        elif pixels_kept_pct < 10:
            print(f"  - Current filtering keeps {pixels_kept_pct:.1f}% of pixels (moderate)")
            print(f"  - May be appropriate for dense urban areas with lots of vegetation")
        else:
            print(f"  - Current filtering keeps {pixels_kept_pct:.1f}% of pixels (good)")
    
    return mask, filter_stats


def apply_morphological_filtering(classification_map, kernel_size=3):
    """
    Apply morphological operations to clean up classification.
    
    - Opening: Removes small noise (isolated pavement pixels)
    - Closing: Fills small holes (missing pavement pixels)
    
    Parameters:
    -----------
    classification_map : ndarray
        2D binary classification map (0=non-pavement, 1=pavement)
    kernel_size : int
        Size of morphological kernel
        
    Returns:
    --------
    filtered_map : ndarray
        Morphologically filtered classification map
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("APPLYING MORPHOLOGICAL FILTERING")
        print(f"{'='*60}")
        n_before = classification_map.sum()
    
    # Convert to boolean
    binary = classification_map.astype(bool)
    
    # Opening: erosion then dilation (removes small noise)
    opened = binary_opening(binary, structure=np.ones((kernel_size, kernel_size)))
    
    # Closing: dilation then erosion (fills small holes)
    filtered = binary_closing(opened, structure=np.ones((kernel_size, kernel_size)))
    
    filtered_map = filtered.astype(int)
    
    if config.VERBOSE:
        n_after = filtered_map.sum()
        n_changed = np.abs(classification_map - filtered_map).sum()
        print(f"  Pixels before: {n_before}")
        print(f"  Pixels after: {n_after}")
        print(f"  Pixels changed: {n_changed} ({100*n_changed/max(n_before, 1):.2f}%)")
    
    return filtered_map


def apply_majority_filter(classification_map, window_size=5):
    """
    Apply majority filter to smooth classification.
    
    Each pixel is replaced by the majority class in its neighborhood.
    
    Parameters:
    -----------
    classification_map : ndarray
        2D binary classification map
    window_size : int
        Size of neighborhood window (must be odd)
        
    Returns:
    --------
    filtered_map : ndarray
        Majority-filtered classification map
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("APPLYING MAJORITY FILTER")
        print(f"{'='*60}")
        n_before = classification_map.sum()
    
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    def majority_filter(values):
        """Return majority class in neighborhood."""
        values_flat = values.flatten()
        # Handle edge cases
        if len(values_flat) == 0:
            return 0
        counts = np.bincount(values_flat.astype(int))
        if len(counts) == 0:
            return 0
        return int(counts.argmax())
    
    filtered_map = generic_filter(
        classification_map.astype(float),
        majority_filter,
        size=window_size,
        mode='reflect'
    ).astype(int)
    
    if config.VERBOSE:
        n_after = filtered_map.sum()
        n_changed = np.abs(classification_map - filtered_map).sum()
        print(f"  Pixels before: {n_before}")
        print(f"  Pixels after: {n_after}")
        print(f"  Pixels changed: {n_changed} ({100*n_changed/classification_map.size:.2f}%)")
    
    return filtered_map
