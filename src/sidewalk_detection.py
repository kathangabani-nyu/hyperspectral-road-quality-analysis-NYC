"""
Sidewalk-Specific Pavement Detection

Enhanced detection focused on sidewalks (narrow linear features) while
excluding building rooftops using geometric and contextual properties.

Key Features:
- Narrow width detection (sidewalks are typically 1-5 meters wide)
- Linear connectivity (sidewalks form paths)
- Adjacency to roads (sidewalks are usually next to roads)
- Exclude large rectangular regions (rooftops)
"""

import numpy as np
from scipy.ndimage import label, binary_dilation, distance_transform_edt
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.ndimage import gaussian_filter
import advanced_config as config


def detect_narrow_linear_features(classification_map, min_width=1, max_width=8, min_length=10):
    """
    Detect narrow linear features (sidewalks) using distance transform.
    
    Sidewalks are typically:
    - Narrow (1-5 pixels wide at typical resolution)
    - Elongated (high length/width ratio)
    - Form connected networks
    
    Parameters:
    -----------
    classification_map : ndarray
        Binary classification map
    min_width : int
        Minimum feature width in pixels
    max_width : int
        Maximum feature width in pixels (sidewalks are narrow)
    min_length : int
        Minimum feature length in pixels
        
    Returns:
    --------
    sidewalk_map : ndarray
        Binary map of detected sidewalks
    """
    if config.VERBOSE:
        print(f"\nDetecting narrow linear features (sidewalks)...")
        print(f"  Width range: {min_width}-{max_width} pixels")
        print(f"  Min length: {min_length} pixels")
    
    # Distance transform to find centerlines
    dist_transform = distance_transform_edt(classification_map)
    
    # Find pixels within width range
    width_mask = (dist_transform >= min_width) & (dist_transform <= max_width)
    
    # Find centerlines (local maxima)
    local_maxima = (dist_transform == maximum_filter(dist_transform, size=5))
    local_maxima = local_maxima & width_mask
    
    # Label connected centerlines
    labeled_lines, n_lines = label(local_maxima)
    
    sidewalk_map = np.zeros_like(classification_map, dtype=bool)
    
    for line_id in range(1, n_lines + 1):
        line_mask = (labeled_lines == line_id)
        line_length = line_mask.sum()
        
        if line_length >= min_length:
            # Expand to full width
            line_width = int(dist_transform[line_mask].mean() * 2)
            line_width = min(line_width, max_width)
            
            expanded = binary_dilation(line_mask, iterations=line_width)
            sidewalk_map = sidewalk_map | expanded
    
    if config.VERBOSE:
        print(f"  Detected {n_lines} sidewalk segments")
        print(f"  Sidewalk pixels: {sidewalk_map.sum()}")
    
    return sidewalk_map


def detect_road_adjacent_pavement(classification_map, sidewalk_candidates, road_dilation=15):
    """
    Identify pavement that is adjacent to roads (likely sidewalks).
    
    Sidewalks are typically adjacent to roads. This function:
    1. Identifies road-like features (wider linear features)
    2. Finds pavement adjacent to roads
    3. Prioritizes these areas as sidewalks
    
    Parameters:
    -----------
    classification_map : ndarray
        Binary classification map
    sidewalk_candidates : ndarray
        Binary map of sidewalk candidates
    road_dilation : int
        Dilation radius to find areas adjacent to roads
        
    Returns:
    --------
    adjacent_map : ndarray
        Binary map of road-adjacent pavement
    """
    if config.VERBOSE:
        print(f"\nDetecting road-adjacent pavement...")
    
    # Detect roads (wider linear features)
    dist_transform = distance_transform_edt(classification_map)
    road_mask = (dist_transform >= 8) & (dist_transform <= 30)  # Roads are wider
    
    # Dilate roads to find adjacent areas
    road_dilated = binary_dilation(road_mask, iterations=road_dilation)
    
    # Find pavement adjacent to roads
    adjacent_map = sidewalk_candidates & road_dilated & (~road_mask)
    
    if config.VERBOSE:
        print(f"  Road-adjacent pavement pixels: {adjacent_map.sum()}")
    
    return adjacent_map


def filter_rooftops_aggressive(classification_map, max_aspect_ratio=2.0, min_area=150, max_area=4000):
    """
    Aggressively filter building rooftops using stricter criteria.
    
    Building roofs are:
    - Square/rectangular (low aspect ratio)
    - Isolated (not connected to road network)
    - Medium to large size
    - High compactness
    
    Parameters:
    -----------
    classification_map : ndarray
        Binary classification map
    max_aspect_ratio : float
        Maximum aspect ratio for rooftops
    min_area : int
        Minimum area to consider
    max_area : int
        Maximum area for rooftops
        
    Returns:
    --------
    filtered_map : ndarray
        Map with rooftops removed
    """
    if config.VERBOSE:
        print(f"\nAggressively filtering building rooftops...")
        print(f"  Criteria: aspect_ratio <= {max_aspect_ratio}, "
              f"area {min_area}-{max_area}, compactness > 0.65")
    
    # Label connected components
    labeled, n_regions = label(classification_map)
    
    if n_regions == 0:
        return classification_map
    
    filtered_map = np.zeros_like(classification_map)
    n_removed = 0
    
    for region_id in range(1, n_regions + 1):
        region_mask = (labeled == region_id)
        region_area = region_mask.sum()
        
        # Keep very small or very large regions
        if region_area < min_area or region_area > max_area:
            filtered_map[region_mask] = 1
            continue
        
        # Compute bounding box
        coords = np.argwhere(region_mask)
        if len(coords) == 0:
            continue
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        bbox_width = x_max - x_min + 1
        bbox_height = y_max - y_min + 1
        
        # Aspect ratio
        aspect_ratio = max(bbox_width, bbox_height) / (min(bbox_width, bbox_height) + 1e-10)
        
        # Compactness
        bbox_area = bbox_width * bbox_height
        compactness = region_area / bbox_area if bbox_area > 0 else 0
        
        # Stricter rooftop criteria
        is_rooftop = (
            aspect_ratio <= max_aspect_ratio and
            compactness > 0.65 and  # Higher compactness threshold
            min_area <= region_area <= max_area
        )
        
        if is_rooftop:
            n_removed += 1
            if config.VERBOSE and n_removed <= 10:
                print(f"  Removed rooftop {region_id}: area={region_area}, "
                      f"aspect={aspect_ratio:.2f}, compactness={compactness:.2f}")
        else:
            filtered_map[region_mask] = 1
    
    if config.VERBOSE:
        print(f"  Removed {n_removed} rooftop regions")
        print(f"  Remaining: {filtered_map.sum()} pixels")
    
    return filtered_map


def apply_sidewalk_geometric_constraints(sidewalk_map, min_length=15, max_width=6):
    """
    Apply geometric constraints specific to sidewalks.
    
    Sidewalks should be:
    - Elongated (high length/width ratio)
    - Narrow (consistent width)
    - Form connected paths
    
    Parameters:
    -----------
    sidewalk_map : ndarray
        Binary sidewalk map
    min_length : int
        Minimum sidewalk length
    max_width : int
        Maximum sidewalk width
        
    Returns:
    --------
    constrained_map : ndarray
        Map with geometric constraints applied
    """
    if config.VERBOSE:
        print(f"\nApplying sidewalk geometric constraints...")
    
    # Label connected components
    labeled, n_regions = label(sidewalk_map)
    
    if n_regions == 0:
        return sidewalk_map
    
    constrained_map = np.zeros_like(sidewalk_map)
    
    for region_id in range(1, n_regions + 1):
        region_mask = (labeled == region_id)
        
        # Compute dimensions
        coords = np.argwhere(region_mask)
        if len(coords) == 0:
            continue
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        length = max(y_max - y_min, x_max - x_min) + 1
        width = min(y_max - y_min, x_max - x_min) + 1
        
        # Check constraints
        is_valid_sidewalk = (
            length >= min_length and
            width <= max_width and
            length / (width + 1e-10) >= 2.0  # Elongated
        )
        
        if is_valid_sidewalk:
            constrained_map[region_mask] = 1
    
    if config.VERBOSE:
        print(f"  Valid sidewalk regions: {constrained_map.sum()} pixels")
    
    return constrained_map


def sidewalk_specific_filtering(classification_map, use_road_adjacency=True):
    """
    Apply sidewalk-specific filtering to classification map.
    
    This function:
    1. Detects narrow linear features (sidewalks)
    2. Filters rooftops aggressively
    3. Applies geometric constraints
    4. Optionally uses road adjacency
    
    Parameters:
    -----------
    classification_map : ndarray
        Binary classification map
    use_road_adjacency : bool
        Whether to use road adjacency information
        
    Returns:
    --------
    sidewalk_map : ndarray
        Filtered map focused on sidewalks
    """
    if config.VERBOSE:
        print(f"\n{'='*70}")
        print("SIDEWALK-SPECIFIC FILTERING")
        print("Focusing on narrow linear features (sidewalks)")
        print(f"{'='*70}")
    
    # Step 1: Aggressively filter rooftops
    filtered = filter_rooftops_aggressive(
        classification_map,
        max_aspect_ratio=2.0,
        min_area=150,
        max_area=4000
    )
    
    # Step 2: Detect narrow linear features (sidewalks)
    sidewalk_candidates = detect_narrow_linear_features(
        filtered,
        min_width=1,
        max_width=8,
        min_length=10
    )
    
    # Step 3: Optionally use road adjacency
    if use_road_adjacency:
        road_adjacent = detect_road_adjacent_pavement(
            filtered,
            sidewalk_candidates,
            road_dilation=15
        )
        # Combine: prioritize road-adjacent, but also keep other narrow features
        sidewalk_map = sidewalk_candidates | road_adjacent
    else:
        sidewalk_map = sidewalk_candidates
    
    # Step 4: Apply geometric constraints
    sidewalk_map = apply_sidewalk_geometric_constraints(
        sidewalk_map,
        min_length=15,
        max_width=6
    )
    
    if config.VERBOSE:
        print(f"\nFinal sidewalk pixels: {sidewalk_map.sum()} "
              f"({sidewalk_map.sum() / classification_map.size * 100:.2f}%)")
    
    return sidewalk_map


def integrate_manual_mask(classification_map, manual_mask_path):
    """
    Integrate manual mask into classification.
    
    Parameters:
    -----------
    classification_map : ndarray
        Automated classification map
    manual_mask_path : str
        Path to manual mask file
        
    Returns:
    --------
    combined_map : ndarray
        Combined classification map
    """
    import os
    
    if not os.path.exists(manual_mask_path):
        if config.VERBOSE:
            print(f"\nManual mask not found: {manual_mask_path}")
            print("  Using automated classification only")
        return classification_map
    
    manual_mask = np.load(manual_mask_path).astype(bool)
    
    if manual_mask.shape != classification_map.shape:
        if config.VERBOSE:
            print(f"\nManual mask shape mismatch: {manual_mask.shape} vs {classification_map.shape}")
            print("  Using automated classification only")
        return classification_map
    
    # Combine: manual mask takes priority (union)
    combined_map = classification_map | manual_mask
    
    if config.VERBOSE:
        print(f"\nIntegrated manual mask:")
        print(f"  Manual pavement pixels: {manual_mask.sum()}")
        print(f"  Automated pavement pixels: {classification_map.sum()}")
        print(f"  Combined pavement pixels: {combined_map.sum()}")
    
    return combined_map

