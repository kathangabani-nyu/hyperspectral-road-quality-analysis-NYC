"""
Spectral masking for pavement classification.
Uses spectral characteristics to filter out obvious non-pavement areas.
"""

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, generic_filter
import advanced_config as config


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
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("CREATING SPECTRAL MASK")
        print(f"{'='*60}")
    
    n_bands = X.shape[1]
    
    # Approximate band indices
    blue_idx = int(n_bands * 0.15)
    green_idx = int(n_bands * 0.25)
    red_idx = int(n_bands * 0.35)
    nir_idx = int(n_bands * 0.45) if n_bands > 50 else min(n_bands-1, 70)
    
    # Initialize mask (all True = keep all initially)
    mask = np.ones(X.shape[0], dtype=bool)
    initial_count = mask.sum()
    
    # 1. Filter high NDVI (vegetation)
    if nir_idx < n_bands and red_idx < n_bands:
        red = X[:, red_idx]
        nir = X[:, nir_idx]
        denominator = nir + red + 1e-10
        ndvi = (nir - red) / denominator
        
        # Pavement should have NDVI < 0.3 (adjust threshold)
        ndvi_threshold = 0.3
        mask = mask & (ndvi < ndvi_threshold)
        
        if config.VERBOSE:
            n_filtered = initial_count - mask.sum()
            print(f"  Filtered {n_filtered} pixels with NDVI >= {ndvi_threshold} (vegetation)")
            initial_count = mask.sum()
    
    # 2. Filter very low brightness (shadows, water)
    brightness = X.mean(axis=1)
    brightness_low = np.percentile(brightness[mask], 5)  # Bottom 5% of remaining pixels
    mask = mask & (brightness > brightness_low)
    
    if config.VERBOSE:
        n_filtered = initial_count - mask.sum()
        print(f"  Filtered {n_filtered} pixels with very low brightness (shadows/water)")
        initial_count = mask.sum()
    
    # 3. Filter very high brightness (bright roofs, concrete)
    # Keep some bright areas (could be pavement), but filter extreme outliers
    brightness_extreme = np.percentile(brightness[mask], 99)  # Top 1% of remaining pixels
    mask = mask & (brightness < brightness_extreme)
    
    if config.VERBOSE:
        n_filtered = initial_count - mask.sum()
        print(f"  Filtered {n_filtered} pixels with extreme brightness (bright roofs)")
        initial_count = mask.sum()
    
    # 4. Filter based on spectral variance (pavement is relatively uniform)
    spectral_std = X.std(axis=1)
    if mask.sum() > 0:
        std_threshold = np.percentile(spectral_std[mask], 90)  # Top 10% variance of remaining
        # Keep pixels with moderate variance (pavement is uniform but not perfectly flat)
        mask = mask & (spectral_std < std_threshold)
        
        if config.VERBOSE:
            n_filtered = initial_count - mask.sum()
            print(f"  Filtered {n_filtered} pixels with high spectral variance")
    
    if config.VERBOSE:
        print(f"\nSpectral mask summary:")
        print(f"  Pixels kept: {mask.sum()} ({100*mask.sum()/len(mask):.1f}%)")
        print(f"  Pixels filtered: {(~mask).sum()} ({100*(~mask).sum()/len(mask):.1f}%)")
    
    return mask


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
