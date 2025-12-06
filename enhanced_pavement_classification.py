"""
Enhanced Multi-Stage Pavement Classification

Professional approach combining:
1. Spectral-based filtering (remove obvious non-pavement)
2. Geometric filtering (remove rooftops by shape)
3. Spectral validation (verify remaining regions)
4. Context-aware refinement (road adjacency, connectivity)
"""

import numpy as np
from scipy.ndimage import label, binary_dilation, binary_erosion, distance_transform_edt
from scipy.ndimage import maximum_filter, gaussian_filter
import advanced_config as config


def spectral_filter_rooftops(hyperspectral_data, classification_map, X_valid, valid_mask):
    """
    Use spectral characteristics to identify and remove rooftops.
    
    Rooftops typically have:
    - High brightness (reflective materials)
    - Low NDVI (not vegetation)
    - High spectral variance (mixed materials)
    - Specific spectral signatures
    
    Parameters:
    -----------
    hyperspectral_data : ndarray
        3D hyperspectral data (height, width, bands)
    classification_map : ndarray
        Binary classification map
    X_valid : ndarray
        Feature matrix for valid pixels
    valid_mask : ndarray
        Boolean mask of valid pixels
        
    Returns:
    --------
    filtered_map : ndarray
        Map with spectrally-identified rooftops removed
    """
    if config.VERBOSE:
        print(f"\nSpectral filtering of rooftops...")
    
    height, width = classification_map.shape
    n_bands = X_valid.shape[1]
    
    # Get band indices
    red_idx = int(n_bands * 0.35)
    nir_idx = int(n_bands * 0.45) if n_bands > 50 else min(n_bands-1, 70)
    
    # Create spectral features for all pixels in classification
    filtered_map = classification_map.copy()
    
    # Get indices of pavement pixels in full image
    classification_flat = classification_map.flatten()
    valid_mask_flat = valid_mask.flatten()
    pavement_mask_flat = classification_flat & valid_mask_flat
    pavement_indices_full = np.where(pavement_mask_flat)[0]
    
    if len(pavement_indices_full) == 0:
        return filtered_map
    
    # Map full indices to X_valid indices
    valid_indices_full = np.where(valid_mask_flat)[0]
    full_to_valid_map = {full_idx: valid_idx for valid_idx, full_idx in enumerate(valid_indices_full)}
    pavement_indices_valid = np.array([full_to_valid_map[idx] for idx in pavement_indices_full if idx in full_to_valid_map])
    
    if len(pavement_indices_valid) == 0:
        return filtered_map
    
    # Compute spectral characteristics for pavement pixels
    pavement_spectra = X_valid[pavement_indices_valid]
    
    # Compute NDVI
    if nir_idx < n_bands and red_idx < n_bands:
        red = pavement_spectra[:, red_idx]
        nir = pavement_spectra[:, nir_idx]
        ndvi = (nir - red) / (nir + red + 1e-10)
    else:
        ndvi = np.zeros(len(pavement_indices))
    
    # Compute brightness
    brightness = pavement_spectra.mean(axis=1)
    
    # Compute spectral variance
    spectral_variance = pavement_spectra.std(axis=1)
    
    # Rooftop criteria (spectral) - More conservative:
    # - Very high brightness (above 90th percentile) - only very bright roofs
    # - Low NDVI (< 0.15, not vegetation)
    # - High spectral variance (above 80th percentile) - mixed materials
    brightness_threshold = np.percentile(brightness, 90)  # More conservative
    variance_threshold = np.percentile(spectral_variance, 80)  # More conservative
    
    is_rooftop_spectral = (
        (brightness > brightness_threshold) &
        (ndvi < 0.15) &  # Stricter
        (spectral_variance > variance_threshold)
    )
    
    # Remove spectrally-identified rooftops
    rooftop_indices_valid = pavement_indices_valid[is_rooftop_spectral]
    rooftop_indices_full = pavement_indices_full[is_rooftop_spectral]
    
    if len(rooftop_indices_full) > 0:
        # Convert flat indices to 2D coordinates
        y_coords, x_coords = np.unravel_index(rooftop_indices_full, (height, width))
        filtered_map[y_coords, x_coords] = 0
        
        if config.VERBOSE:
            print(f"  Removed {len(rooftop_indices_full)} pixels based on spectral characteristics")
    
    return filtered_map


def multi_scale_geometric_filtering(classification_map):
    """
    Multi-scale geometric filtering to remove rooftops at different sizes.
    More conservative to avoid removing all pavement.
    
    Parameters:
    -----------
    classification_map : ndarray
        Binary classification map
        
    Returns:
    --------
    filtered_map : ndarray
        Map with rooftops removed at multiple scales
    """
    if config.VERBOSE:
        print(f"\nMulti-scale geometric filtering...")
    
    filtered_map = classification_map.copy()
    
    # More conservative filtering - only remove obvious rooftops
    scales = [
        {'max_aspect': 1.5, 'min_area': 200, 'max_area': 1500, 'compactness': 0.75},  # Very square, compact
        {'max_aspect': 1.8, 'min_area': 300, 'max_area': 3000, 'compactness': 0.70},  # Square, compact
    ]
    
    total_removed = 0
    
    for scale_idx, scale_params in enumerate(scales):
        labeled, n_regions = label(filtered_map)
        
        if n_regions == 0:
            break
        
        n_removed_scale = 0
        
        for region_id in range(1, n_regions + 1):
            region_mask = (labeled == region_id)
            region_area = region_mask.sum()
            
            if region_area < scale_params['min_area'] or region_area > scale_params['max_area']:
                continue
            
            coords = np.argwhere(region_mask)
            if len(coords) == 0:
                continue
            
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            bbox_width = x_max - x_min + 1
            bbox_height = y_max - y_min + 1
            aspect_ratio = max(bbox_width, bbox_height) / (min(bbox_width, bbox_height) + 1e-10)
            bbox_area = bbox_width * bbox_height
            compactness = region_area / bbox_area if bbox_area > 0 else 0
            
            is_rooftop = (
                aspect_ratio <= scale_params['max_aspect'] and
                compactness > scale_params['compactness'] and
                scale_params['min_area'] <= region_area <= scale_params['max_area']
            )
            
            if is_rooftop:
                filtered_map[region_mask] = 0
                n_removed_scale += 1
                total_removed += 1
        
        if config.VERBOSE:
            print(f"  Scale {scale_idx + 1}: Removed {n_removed_scale} regions")
    
    if config.VERBOSE:
        print(f"  Total removed: {total_removed} rooftop regions")
        print(f"  Remaining: {filtered_map.sum()} pixels")
    
    return filtered_map


def detect_and_preserve_linear_features(classification_map, min_width=2, max_width=12):
    """
    Detect and preserve linear features (roads and sidewalks).
    
    Parameters:
    -----------
    classification_map : ndarray
        Binary classification map
    min_width : int
        Minimum feature width
    max_width : int
        Maximum feature width
        
    Returns:
    --------
    linear_map : ndarray
        Map with linear features preserved
    """
    if config.VERBOSE:
        print(f"\nDetecting and preserving linear features...")
    
    # Distance transform
    dist_transform = distance_transform_edt(classification_map)
    
    # Find linear features (within width range)
    linear_mask = (dist_transform >= min_width) & (dist_transform <= max_width)
    
    # Find centerlines
    local_maxima = (dist_transform == maximum_filter(dist_transform, size=7))
    local_maxima = local_maxima & linear_mask
    
    # Label centerlines
    labeled_lines, n_lines = label(local_maxima)
    
    linear_map = np.zeros_like(classification_map, dtype=bool)
    
    for line_id in range(1, n_lines + 1):
        line_mask = (labeled_lines == line_id)
        line_length = line_mask.sum()
        
        if line_length >= 15:  # Minimum length
            # Expand to full width
            line_width = int(dist_transform[line_mask].mean() * 2)
            line_width = min(line_width, max_width)
            
            expanded = binary_dilation(line_mask, iterations=line_width)
            linear_map = linear_map | expanded
    
    # Combine with original, prioritizing linear features
    combined_map = classification_map.copy()
    combined_map[linear_map] = 1  # Ensure linear features are kept
    
    if config.VERBOSE:
        print(f"  Detected {n_lines} linear segments")
        print(f"  Linear feature pixels: {linear_map.sum()}")
    
    return combined_map


def apply_connectivity_network_filtering(classification_map, min_network_size=200):
    """
    Keep only regions that form connected networks.
    
    Parameters:
    -----------
    classification_map : ndarray
        Binary classification map
    min_network_size : int
        Minimum size for isolated networks
        
    Returns:
    --------
    network_map : ndarray
        Map with network connectivity applied
    """
    if config.VERBOSE:
        print(f"\nApplying network connectivity filtering...")
    
    labeled, n_regions = label(classification_map)
    
    if n_regions == 0:
        return classification_map
    
    # Find largest component (main network)
    region_sizes = np.array([np.sum(labeled == i) for i in range(1, n_regions + 1)])
    main_region_id = np.argmax(region_sizes) + 1
    
    network_map = np.zeros_like(classification_map)
    
    # Keep main network
    main_mask = (labeled == main_region_id)
    network_map[main_mask] = 1
    
    # Keep other large networks (sidewalks/roads that might be disconnected)
    for region_id in range(1, n_regions + 1):
        if region_id == main_region_id:
            continue
        
        region_mask = (labeled == region_id)
        if region_mask.sum() >= min_network_size:
            network_map[region_mask] = 1
    
    if config.VERBOSE:
        print(f"  Main network: {main_mask.sum()} pixels")
        print(f"  Total network: {network_map.sum()} pixels")
    
    return network_map


def enhanced_pavement_classification(
    classification_map,
    hyperspectral_data,
    X_valid,
    valid_mask
):
    """
    Enhanced multi-stage pavement classification.
    
    Pipeline:
    1. Spectral filtering (remove spectrally-identified rooftops)
    2. Multi-scale geometric filtering (remove rooftops by shape)
    3. Linear feature detection (preserve roads/sidewalks)
    4. Network connectivity (keep connected pavement networks)
    
    Parameters:
    -----------
    classification_map : ndarray
        Initial binary classification map
    hyperspectral_data : ndarray
        3D hyperspectral data
    X_valid : ndarray
        Feature matrix for valid pixels
    valid_mask : ndarray
        Boolean mask of valid pixels
        
    Returns:
    --------
    enhanced_map : ndarray
        Enhanced classification map
    """
    if config.VERBOSE:
        print(f"\n{'='*70}")
        print("ENHANCED MULTI-STAGE PAVEMENT CLASSIFICATION")
        print(f"{'='*70}")
        print(f"Initial pavement pixels: {classification_map.sum()}")
    
    # Stage 1: Spectral filtering
    stage1 = spectral_filter_rooftops(
        hyperspectral_data,
        classification_map,
        X_valid,
        valid_mask
    )
    
    # Stage 2: Multi-scale geometric filtering
    stage2 = multi_scale_geometric_filtering(stage1)
    
    # Stage 3: Detect and preserve linear features
    stage3 = detect_and_preserve_linear_features(stage2, min_width=2, max_width=12)
    
    # Stage 4: Network connectivity (more lenient)
    final = apply_connectivity_network_filtering(stage3, min_network_size=100)  # Lower threshold
    
    # Safety check: if we removed too much, use a less aggressive version
    if final.sum() < classification_map.sum() * 0.1:  # Less than 10% remaining
        if config.VERBOSE:
            print(f"\nWARNING: Too much removed ({final.sum()} pixels remaining). Using less aggressive filtering.")
        
        # Fallback: just do geometric filtering without spectral
        final = multi_scale_geometric_filtering(classification_map)
        final = detect_and_preserve_linear_features(final, min_width=2, max_width=15)
    
    if config.VERBOSE:
        print(f"\nFinal pavement pixels: {final.sum()} "
              f"({final.sum() / classification_map.size * 100:.2f}%)")
        if classification_map.sum() > 0:
            reduction = (1 - final.sum() / (classification_map.sum() + 1e-10)) * 100
            print(f"Reduction: {reduction:.1f}%")
    
    return final

