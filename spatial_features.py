"""
Advanced spatial features for hyperspectral classification.
Implements superpixel segmentation and spatial-spectral feature fusion.
"""

import numpy as np
from scipy.ndimage import uniform_filter, generic_filter
import advanced_config as config


def compute_superpixels(rgb_image, n_segments=200, compactness=10.0):
    """
    Compute superpixels using SLIC algorithm.
    
    Parameters:
    -----------
    rgb_image : ndarray
        RGB image (height, width, 3) for superpixel computation
    n_segments : int
        Approximate number of superpixels
    compactness : float
        Balance between color proximity and space proximity
        
    Returns:
    --------
    segments : ndarray
        Superpixel labels (height, width)
    """
    try:
        from skimage.segmentation import slic
        # Ensure RGB is in 0-255 range
        if rgb_image.max() <= 1.0:
            rgb_uint8 = (rgb_image * 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb_image.astype(np.uint8)
        
        # Compute superpixels
        segments = slic(rgb_uint8, n_segments=n_segments, compactness=compactness, 
                       start_label=0, channel_axis=2)
        
        if config.VERBOSE:
            n_actual = len(np.unique(segments))
            print(f"  Computed {n_actual} superpixels (target: {n_segments})")
        
        return segments
    except ImportError:
        if config.VERBOSE:
            print(f"  Warning: scikit-image not available, using grid-based segmentation")
    except Exception as e:
        if config.VERBOSE:
            print(f"  Warning: Could not compute superpixels: {e}")
            print(f"  Falling back to grid-based segmentation")
    
    # Fallback: grid-based segmentation
    h, w = rgb_image.shape[:2]
    grid_size = int(np.sqrt(n_segments))
    segments = np.zeros((h, w), dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            y_start = i * h // grid_size
            y_end = (i + 1) * h // grid_size
            x_start = j * w // grid_size
            x_end = (j + 1) * w // grid_size
            segments[y_start:y_end, x_start:x_end] = i * grid_size + j
    return segments


def extract_superpixel_features(hyperspectral_data, segments):
    """
    Extract features for each superpixel.
    
    Parameters:
    -----------
    hyperspectral_data : ndarray
        3D hyperspectral data (bands, height, width)
    segments : ndarray
        Superpixel labels (height, width)
        
    Returns:
    --------
    superpixel_features : ndarray
        Features per superpixel (n_superpixels, n_features)
    superpixel_map : ndarray
        Map assigning each pixel to its superpixel features
    """
    n_bands = hyperspectral_data.shape[0]
    height, width = hyperspectral_data.shape[1:]
    n_superpixels = len(np.unique(segments))
    
    if config.VERBOSE:
        print(f"  Extracting features for {n_superpixels} superpixels...")
    
    # Compute mean spectral signature per superpixel
    superpixel_features = []
    
    for seg_id in range(n_superpixels):
        mask = (segments == seg_id)
        if mask.sum() == 0:
            continue
        
        # Mean spectral signature
        mean_spectrum = hyperspectral_data[:, mask].mean(axis=1)
        
        # Standard deviation
        std_spectrum = hyperspectral_data[:, mask].std(axis=1)
        
        # Combine
        features = np.concatenate([mean_spectrum, std_spectrum])
        superpixel_features.append(features)
    
    superpixel_features = np.array(superpixel_features)
    
    # Create map: for each pixel, assign its superpixel features
    superpixel_map = np.zeros((n_superpixels, 2 * n_bands))
    for seg_id in range(n_superpixels):
        mask = (segments == seg_id)
        if mask.sum() > 0:
            idx = np.where(np.unique(segments) == seg_id)[0]
            if len(idx) > 0:
                superpixel_map[seg_id] = superpixel_features[idx[0]]
    
    return superpixel_features, superpixel_map


def assign_superpixel_features_to_pixels(superpixel_map, segments):
    """
    Assign superpixel features to each pixel.
    
    Parameters:
    -----------
    superpixel_map : ndarray
        Features per superpixel
    segments : ndarray
        Superpixel labels
        
    Returns:
    --------
    pixel_features : ndarray
        Features for each pixel (height, width, n_features)
    """
    height, width = segments.shape
    n_features = superpixel_map.shape[1]
    
    pixel_features = np.zeros((height, width, n_features))
    
    for seg_id in range(len(superpixel_map)):
        mask = (segments == seg_id)
        pixel_features[mask] = superpixel_map[seg_id]
    
    return pixel_features


def compute_spatial_context_features(classification_map, window_size=5):
    """
    Compute spatial context features based on neighborhood classification.
    
    Parameters:
    -----------
    classification_map : ndarray
        2D classification map
    window_size : int
        Size of neighborhood window
        
    Returns:
    --------
    context_features : ndarray
        Spatial context features (height, width, n_features)
    """
    height, width = classification_map.shape
    
    # Local class distribution
    def compute_class_dist(values):
        """Compute distribution of classes in neighborhood."""
        unique, counts = np.unique(values, return_counts=True)
        dist = np.zeros(2)  # Binary classification
        for u, c in zip(unique, counts):
            if u >= 0 and u < 2:
                dist[int(u)] = c
        return dist / (values.size + 1e-10)
    
    class_dist = generic_filter(
        classification_map.astype(float),
        compute_class_dist,
        size=window_size,
        mode='reflect'
    )
    
    # Reshape to (height, width, n_features)
    if class_dist.ndim == 2:
        context_features = np.stack([class_dist, 1 - class_dist], axis=2)
    else:
        context_features = class_dist
    
    return context_features


def validate_clusters_spatially(cluster_map, rgb_image, min_compactness=0.3):
    """
    Validate clusters based on spatial compactness.
    Pavement should form relatively compact, contiguous regions.
    
    Parameters:
    -----------
    cluster_map : ndarray
        2D cluster map
    rgb_image : ndarray
        RGB image for reference
    min_compactness : float
        Minimum compactness threshold
        
    Returns:
    --------
    cluster_scores : dict
        Spatial validation scores for each cluster
    """
    from scipy.ndimage import label, find_objects
    
    cluster_scores = {}
    unique_clusters = np.unique(cluster_map)
    unique_clusters = unique_clusters[unique_clusters >= 0]
    
    for cluster_id in unique_clusters:
        cluster_mask = (cluster_map == cluster_id)
        
        # Label connected components
        labeled, n_components = label(cluster_mask)
        
        if n_components == 0:
            cluster_scores[cluster_id] = 0.0
            continue
        
        # Compute compactness for each component
        compactness_scores = []
        for i in range(1, n_components + 1):
            component_mask = (labeled == i)
            area = component_mask.sum()
            
            if area == 0:
                continue
            
            # Bounding box
            bbox = find_objects(labeled == i)[0]
            if bbox is None:
                continue
            
            bbox_area = (bbox[0].stop - bbox[0].start) * (bbox[1].stop - bbox[1].start)
            
            # Compactness: area / bounding_box_area
            compactness = area / bbox_area if bbox_area > 0 else 0.0
            compactness_scores.append(compactness)
        
        # Average compactness
        avg_compactness = np.mean(compactness_scores) if compactness_scores else 0.0
        
        # Penalize if too many small fragments
        fragment_penalty = min(1.0, n_components / 10.0)  # Penalize if >10 fragments
        
        cluster_scores[cluster_id] = avg_compactness * (1.0 - fragment_penalty * 0.5)
    
    return cluster_scores

