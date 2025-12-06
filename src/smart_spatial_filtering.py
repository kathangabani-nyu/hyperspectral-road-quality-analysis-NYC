"""
Smart Spatial Filtering for Pavement Classification

NOVEL APPROACH: Use geometric properties to distinguish pavement from building roofs.

Key Insight:
- Roads: Elongated, form networks, have consistent width
- Building roofs: Square/rectangular, isolated, variable size
- Parking lots: Large, relatively uniform, connected to roads

Strategy:
1. Identify elongated features (roads) - high length/width ratio
2. Identify large uniform regions (parking lots) - large area, low aspect ratio
3. Remove isolated square/rectangular regions (building roofs)
4. Apply connectivity constraints (pavement forms networks)
"""

import numpy as np
from scipy.ndimage import label, binary_dilation, distance_transform_edt
from scipy.ndimage import maximum_filter
import advanced_config as config


def filter_building_roofs(classification_map, max_aspect_ratio=2.5, min_area=200, max_area=8000):
    """
    Filter out building roofs based on geometric properties.
    
    Building roofs are typically:
    - Square or rectangular (low aspect ratio)
    - Isolated (not connected to road network)
    - Medium size (not too small, not too large)
    - High compactness (fill their bounding box)
    
    Parameters:
    -----------
    classification_map : ndarray
        Binary classification map
    max_aspect_ratio : float
        Maximum aspect ratio for building roofs (relaxed to 2.5)
    min_area : int
        Minimum area to consider
    max_area : int
        Maximum area for building roofs
        
    Returns:
    --------
    filtered_map : ndarray
        Map with building roofs removed
    """
    if config.VERBOSE:
        print(f"\nFiltering building roofs using geometric properties...")
        print(f"  Criteria: aspect_ratio <= {max_aspect_ratio}, "
              f"area {min_area}-{max_area}, compactness > 0.6")
    
    # Label connected components
    labeled, n_regions = label(classification_map)
    
    if n_regions == 0:
        return classification_map
    
    filtered_map = np.zeros_like(classification_map)
    n_removed = 0
    removed_areas = []
    
    for region_id in range(1, n_regions + 1):
        region_mask = (labeled == region_id)
        region_area = region_mask.sum()
        
        # Always keep very small or very large regions
        if region_area < min_area:
            filtered_map[region_mask] = 1
            continue
        
        if region_area > max_area:
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
        
        # Compactness (area / bounding_box_area)
        bbox_area = bbox_width * bbox_height
        compactness = region_area / bbox_area if bbox_area > 0 else 0
        
        # Building roof criteria (relaxed):
        # - Low to moderate aspect ratio (square/rectangular, not elongated)
        # - Moderate to high compactness (fills bounding box reasonably well)
        # - Medium size
        is_building_roof = (
            aspect_ratio <= max_aspect_ratio and
            compactness > 0.6 and  # Fills at least 60% of bounding box
            min_area <= region_area <= max_area
        )
        
        if is_building_roof:
            n_removed += 1
            removed_areas.append(region_area)
            if config.VERBOSE and n_removed <= 15:  # Log first 15
                print(f"  Removed region {region_id}: area={region_area}, "
                      f"aspect={aspect_ratio:.2f}, compactness={compactness:.2f}")
        else:
            filtered_map[region_mask] = 1
    
    if config.VERBOSE:
        print(f"  Removed {n_removed} building roof regions")
        if removed_areas:
            print(f"  Removed areas: min={min(removed_areas)}, max={max(removed_areas)}, "
                  f"mean={np.mean(removed_areas):.0f}")
        print(f"  Remaining: {filtered_map.sum()} pixels")
    
    return filtered_map


def enhance_road_network(classification_map, min_length=20, max_width=40):
    """
    Enhance road network by identifying elongated features.
    
    Parameters:
    -----------
    classification_map : ndarray
        Binary classification map
    min_length : int
        Minimum road length
    max_width : int
        Maximum road width
        
    Returns:
    --------
    enhanced_map : ndarray
        Map with road network enhanced
    """
    if config.VERBOSE:
        print(f"\nEnhancing road network...")
    
    # Distance transform
    dist_transform = distance_transform_edt(classification_map)
    
    # Find centerlines (local maxima)
    local_maxima = (dist_transform == maximum_filter(dist_transform, size=5))
    local_maxima = local_maxima & (dist_transform > 1)
    
    # Label centerlines
    labeled_lines, n_lines = label(local_maxima)
    
    enhanced_map = classification_map.copy()
    
    for line_id in range(1, n_lines + 1):
        line_mask = (labeled_lines == line_id)
        line_length = line_mask.sum()
        
        if line_length >= min_length:
            # Expand to full road width
            line_width = int(dist_transform[line_mask].mean() * 2)
            line_width = min(line_width, max_width)
            
            expanded = binary_dilation(line_mask, iterations=line_width)
            enhanced_map = enhanced_map | expanded
    
    if config.VERBOSE:
        print(f"  Enhanced with {n_lines} road segments")
        print(f"  Enhanced map: {enhanced_map.sum()} pixels")
    
    return enhanced_map


def apply_connectivity_filtering(classification_map, min_region_size=50):
    """
    Keep only connected regions that form networks.
    
    Remove isolated small regions.
    
    Parameters:
    -----------
    classification_map : ndarray
        Binary classification map
    min_region_size : int
        Minimum size for isolated regions
        
    Returns:
    --------
    connected_map : ndarray
        Map with connectivity filtering applied
    """
    if config.VERBOSE:
        print(f"\nApplying connectivity filtering...")
    
    # Label connected components
    labeled, n_regions = label(classification_map)
    
    if n_regions == 0:
        return classification_map
    
    # Find largest component (main network)
    region_sizes = np.array([np.sum(labeled == i) for i in range(1, n_regions + 1)])
    main_region_id = np.argmax(region_sizes) + 1
    
    connected_map = np.zeros_like(classification_map)
    
    # Keep main network
    main_mask = (labeled == main_region_id)
    connected_map[main_mask] = 1
    
    # Keep large isolated regions (parking lots)
    for region_id in range(1, n_regions + 1):
        if region_id == main_region_id:
            continue
        
        region_mask = (labeled == region_id)
        if region_mask.sum() >= min_region_size:
            connected_map[region_mask] = 1
    
    if config.VERBOSE:
        print(f"  Main network: {main_mask.sum()} pixels")
        print(f"  Total connected: {connected_map.sum()} pixels")
    
    return connected_map


def smart_spatial_filtering(classification_map):
    """
    Apply smart spatial filtering to remove building roofs and enhance roads.
    
    Parameters:
    -----------
    classification_map : ndarray
        Binary classification map
        
    Returns:
    --------
    filtered_map : ndarray
        Spatially filtered classification map
    """
    if config.VERBOSE:
        print(f"\n{'='*70}")
        print("SMART SPATIAL FILTERING")
        print("Removing building roofs, enhancing road network")
        print(f"{'='*70}")
    
    # Step 1: Filter building roofs
    filtered = filter_building_roofs(
        classification_map,
        max_aspect_ratio=2.0,
        min_area=100,
        max_area=5000
    )
    
    # Step 2: Enhance road network
    enhanced = enhance_road_network(
        filtered,
        min_length=20,
        max_width=40
    )
    
    # Step 3: Apply connectivity
    final = apply_connectivity_filtering(
        enhanced,
        min_region_size=50
    )
    
    return final

