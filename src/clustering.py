"""
Unsupervised clustering for hyperspectral data.
Used to identify potential pavement areas.
"""

import numpy as np
from sklearn.cluster import KMeans
try:
    import advanced_config as config
except ImportError:
    import config


def perform_kmeans_clustering(X, n_clusters=None, sample_size=None):
    """
    Perform K-means clustering on hyperspectral data.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix (n_samples, n_features)
    n_clusters : int
        Number of clusters (default from config)
    sample_size : int
        Number of samples to use for clustering (for speed)
        
    Returns:
    --------
    labels : ndarray
        Cluster labels for all samples
    cluster_centers : ndarray
        Cluster centers
    kmeans : KMeans object
        Fitted KMeans model
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("K-MEANS CLUSTERING")
        print(f"{'='*60}")
    
    if n_clusters is None:
        n_clusters = config.N_CLUSTERS
    
    if sample_size is None:
        sample_size = config.MAX_CLUSTERING_SAMPLES
    
    # Sample data for faster clustering if needed
    if len(X) > sample_size:
        if config.VERBOSE:
            print(f"Sampling {sample_size} pixels from {len(X)} for clustering...")
        sample_indices = np.random.RandomState(config.RANDOM_STATE).choice(
            len(X), sample_size, replace=False
        )
        X_sample = X[sample_indices]
    else:
        X_sample = X
        sample_indices = np.arange(len(X))
    
    if config.VERBOSE:
        print(f"Clustering {len(X_sample)} samples into {n_clusters} clusters...")
    
    # Perform K-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=config.RANDOM_STATE,
        n_init=10,
        max_iter=300,
        verbose=0
    )
    
    sample_labels = kmeans.fit_predict(X_sample)
    
    if config.VERBOSE:
        print(f"Clustering complete!")
        print(f"\nCluster distribution (on sample):")
        for i in range(n_clusters):
            count = np.sum(sample_labels == i)
            pct = 100 * count / len(sample_labels)
            print(f"  Cluster {i}: {count} pixels ({pct:.1f}%)")
    
    # Predict labels for all data
    if len(X) > sample_size:
        if config.VERBOSE:
            print(f"\nPredicting cluster labels for all {len(X)} pixels...")
        labels = kmeans.predict(X)
    else:
        labels = sample_labels
    
    if config.VERBOSE:
        print(f"\nFull cluster distribution:")
        for i in range(n_clusters):
            count = np.sum(labels == i)
            pct = 100 * count / len(labels)
            print(f"  Cluster {i}: {count} pixels ({pct:.1f}%)")
    
    return labels, kmeans.cluster_centers_, kmeans


def reshape_labels_to_map(labels, valid_mask, height, width):
    """
    Reshape 1D cluster labels to 2D map.
    
    Parameters:
    -----------
    labels : ndarray
        1D cluster labels
    valid_mask : ndarray
        Boolean mask indicating valid pixels
    height : int
        Image height
    width : int
        Image width
        
    Returns:
    --------
    cluster_map : ndarray
        2D array (height, width) with cluster labels
        Invalid pixels are set to -1
    """
    n_pixels = height * width
    cluster_map_flat = np.full(n_pixels, -1, dtype=int)
    cluster_map_flat[valid_mask] = labels
    cluster_map = cluster_map_flat.reshape(height, width)
    
    return cluster_map


def identify_pavement_clusters(cluster_centers, cluster_map, X_full=None, valid_mask=None, rgb_image=None):
    """
    Identify which clusters represent pavement using spectral and spatial characteristics.
    
    This function uses:
    - Spectral characteristics: Low NDVI, moderate brightness, low variance
    - Spatial characteristics: Compactness, contiguity
    
    Parameters:
    -----------
    cluster_centers : ndarray
        Cluster centers (n_clusters, n_features)
    cluster_map : ndarray
        2D cluster map
    X_full : ndarray, optional
        Full feature matrix for computing spectral indices
    valid_mask : ndarray, optional
        Boolean mask for valid pixels
    rgb_image : ndarray, optional
        RGB image for spatial validation
        
    Returns:
    --------
    pavement_clusters : list
        List of cluster IDs identified as pavement
    """
    n_clusters = cluster_centers.shape[0]
    
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("IDENTIFYING PAVEMENT CLUSTERS")
        print(f"{'='*60}")
        print("\nCluster characteristics (based on mean reflectance):")
        
        # Compute mean reflectance for each cluster
        mean_reflectances = cluster_centers.mean(axis=1)
        
        # Sort by reflectance
        sorted_indices = np.argsort(mean_reflectances)
        
        for idx in sorted_indices:
            mean_refl = mean_reflectances[idx]
            n_pixels = np.sum(cluster_map == idx)
            pct = 100 * n_pixels / (cluster_map >= 0).sum() if (cluster_map >= 0).sum() > 0 else 0
            print(f"  Cluster {idx}: mean_reflectance={mean_refl:.1f}, "
                  f"pixels={n_pixels} ({pct:.1f}%)")
    
    # Improved method: Use spectral characteristics if X_full is provided
    if X_full is not None and valid_mask is not None:
        if config.VERBOSE:
            print("\nUsing spectral characteristics for cluster identification...")
        
        n_bands = X_full.shape[1]
        red_idx = int(n_bands * 0.35)
        nir_idx = int(n_bands * 0.45) if n_bands > 50 else min(n_bands-1, 70)
        
        pavement_clusters = []
        
        for cluster_id in range(n_clusters):
            cluster_mask_2d = (cluster_map == cluster_id)
            cluster_mask_flat = cluster_mask_2d.flatten() & valid_mask
            
            if cluster_mask_flat.sum() == 0:
                continue
            
            cluster_pixels = X_full[cluster_mask_flat]
            
            # Calculate spectral characteristics
            if nir_idx < n_bands and red_idx < n_bands:
                red = cluster_pixels[:, red_idx].mean()
                nir = cluster_pixels[:, nir_idx].mean()
                denominator = nir + red + 1e-10
                ndvi = (nir - red) / denominator
            else:
                ndvi = 0
            
            # Brightness
            brightness = cluster_pixels.mean()
            
            # Spectral uniformity (pavement is relatively uniform)
            spectral_std = cluster_pixels.std(axis=1).mean()
            
            # Mean reflectance
            mean_refl = cluster_centers[cluster_id].mean()
            
            # Pavement criteria:
            # - Low NDVI (< 0.3) - not vegetation
            # - Moderate-high brightness (not shadow/water, not extreme)
            # - Low spectral variance (uniform surface)
            brightness_percentiles = np.percentile(X_full[valid_mask].mean(axis=1), [10, 90])
            
            is_pavement = (
                ndvi < 0.3 and  # Not vegetation
                brightness > brightness_percentiles[0] and  # Not too dark
                brightness < brightness_percentiles[1] and  # Not too bright
                spectral_std < np.percentile(X_full[valid_mask].std(axis=1), 75)  # Relatively uniform
            )
            
            if is_pavement:
                pavement_clusters.append(cluster_id)
                if config.VERBOSE:
                    print(f"  Cluster {cluster_id}: NDVI={ndvi:.3f}, brightness={brightness:.1f}, "
                          f"std={spectral_std:.1f} -> PAVEMENT")
            elif config.VERBOSE:
                print(f"  Cluster {cluster_id}: NDVI={ndvi:.3f}, brightness={brightness:.1f}, "
                      f"std={spectral_std:.1f} -> non-pavement")
        
        # Add spatial validation if RGB image provided
        if rgb_image is not None:
            try:
                from spatial_features import validate_clusters_spatially
                spatial_scores = validate_clusters_spatially(cluster_map, rgb_image)
                
                if config.VERBOSE:
                    print("\nSpatial validation scores:")
                    for cluster_id, score in spatial_scores.items():
                        print(f"  Cluster {cluster_id}: compactness={score:.3f}")
                
                # Filter pavement clusters by spatial compactness
                min_compactness = 0.2
                pavement_clusters = [c for c in pavement_clusters 
                                   if spatial_scores.get(c, 0) >= min_compactness]
                
                if config.VERBOSE:
                    print(f"\nAfter spatial validation: {pavement_clusters}")
            except Exception as e:
                if config.VERBOSE:
                    print(f"  Warning: Spatial validation failed: {e}")
        
        if config.VERBOSE:
            print(f"\nSpectral-based identification: {pavement_clusters}")
    
    else:
        # Fall back to original heuristic method
        if config.VERBOSE:
            print("\nUsing reflectance-based heuristic (no spectral data provided)...")
        
        mean_reflectances = cluster_centers.mean(axis=1)
        
        # Clusters in the middle 40-70 percentile of reflectance are likely pavement
        reflectance_percentiles = np.percentile(mean_reflectances, [30, 70])
        
        pavement_clusters = []
        for i in range(n_clusters):
            if reflectance_percentiles[0] <= mean_reflectances[i] <= reflectance_percentiles[1]:
                pavement_clusters.append(i)
        
        if config.VERBOSE:
            print(f"\nReflectance-based identification: {pavement_clusters}")
    
    if config.VERBOSE:
        print("\nNote: You can manually adjust by editing the 'pavement_clusters' variable")
        print("      in the main script or by modifying the selection logic here.")
    
    return pavement_clusters


def create_binary_mask(cluster_map, pavement_clusters):
    """
    Create binary pavement mask from cluster map.
    
    Parameters:
    -----------
    cluster_map : ndarray
        2D cluster map
    pavement_clusters : list
        List of cluster IDs to mark as pavement
        
    Returns:
    --------
    pavement_mask : ndarray
        Binary mask (1=pavement, 0=non-pavement)
    """
    pavement_mask = np.isin(cluster_map, pavement_clusters).astype(int)
    
    # Set invalid pixels to 0
    pavement_mask[cluster_map < 0] = 0
    
    if config.VERBOSE:
        total_valid = (cluster_map >= 0).sum()
        n_pavement = pavement_mask.sum()
        n_non_pavement = total_valid - n_pavement
        
        print(f"\n{'='*60}")
        print("BINARY MASK CREATED")
        print(f"{'='*60}")
        print(f"Pavement clusters: {pavement_clusters}")
        print(f"Pavement pixels: {n_pavement} ({100*n_pavement/total_valid:.1f}%)")
        print(f"Non-pavement pixels: {n_non_pavement} ({100*n_non_pavement/total_valid:.1f}%)")
    
    return pavement_mask


def analyze_cluster_spectral_signatures(X, labels, cluster_centers):
    """
    Analyze spectral signatures of each cluster.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix (n_samples, n_features)
    labels : ndarray
        Cluster labels
    cluster_centers : ndarray
        Cluster centers
        
    Returns:
    --------
    stats : dict
        Dictionary with cluster statistics
    """
    n_clusters = len(np.unique(labels[labels >= 0]))
    
    stats = {}
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_samples = X[cluster_mask]
        
        stats[i] = {
            'n_samples': len(cluster_samples),
            'mean': cluster_samples.mean(axis=0),
            'std': cluster_samples.std(axis=0),
            'min': cluster_samples.min(axis=0),
            'max': cluster_samples.max(axis=0)
        }
    
    return stats

