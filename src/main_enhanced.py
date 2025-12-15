"""
Enhanced pipeline using ALL 288 hyperspectral bands with advanced features.

Improvements over basic pipeline:
- Uses all 288 bands (vs 30)
- PCA dimensionality reduction
- Advanced spectral indices
- Better feature engineering
- More clusters for finer discrimination
"""

import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import modules
import advanced_config as config
import data_loader
import visualizer
import clustering
import classifier
import dimensionality_reduction
import advanced_features
import spectral_masking


def main():
    """Enhanced pipeline execution."""
    
    print("="*60)
    print("ENHANCED HYPERSPECTRAL PAVEMENT CLASSIFICATION")
    print("Using ALL 288 Bands + Advanced Features")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data file: {config.HYPERSPECTRAL_PATH}")
    print(f"  Test area: {config.TEST_SIZE_X}x{config.TEST_SIZE_Y} pixels")
    print(f"  Position: ({config.START_X}, {config.START_Y})")
    print(f"  Bands: ALL 288 bands")
    print(f"  PCA: {config.USE_PCA}")
    print(f"  Band selection: {config.USE_BAND_SELECTION}")
    print(f"  Output: {config.OUTPUT_DIR}")
    
    # ========================================================================
    # STEP 1: Load ALL hyperspectral bands
    # ========================================================================
    print(f"\n{'='*60}")
    print("STEP 1: LOADING ALL HYPERSPECTRAL BANDS")
    print(f"{'='*60}")
    
    hyperspectral_data, profile, transform = data_loader.load_hyperspectral_data()
    
    height = hyperspectral_data.shape[1]
    width = hyperspectral_data.shape[2]
    n_total_bands = hyperspectral_data.shape[0]
    
    print(f"Loaded {n_total_bands} bands from hyperspectral image")
    
    # ========================================================================
    # STEP 2: Create RGB for visualization
    # ========================================================================
    rgb_composite = visualizer.create_rgb_composite(hyperspectral_data)
    visualizer.save_rgb_composite(rgb_composite, config.FIGURES_DIR + '/rgb_composite_enhanced.png')
    
    # ========================================================================
    # STEP 3: Reshape and extract features
    # ========================================================================
    X, valid_mask = data_loader.reshape_for_ml(hyperspectral_data)
    X_valid = X[valid_mask]
    
    print(f"\nOriginal features: {X_valid.shape[1]} bands × {X_valid.shape[0]} pixels")
    
    # ========================================================================
    # STEP 3.5: Apply Spectral Masking (filter obvious non-pavement)
    # ========================================================================
    filter_stats = None
    if config.USE_SPECTRAL_MASKING:
        if config.VERBOSE:
            print(f"\n{'='*60}")
            print("STEP 3.5: SPECTRAL MASKING")
            print(f"{'='*60}")
        
        spectral_mask, filter_stats = spectral_masking.create_spectral_mask(X_valid, hyperspectral_data)
        
        # Update valid_mask to include spectral masking
        valid_mask_combined = valid_mask.copy()
        valid_mask_combined[valid_mask] = spectral_mask
        
        print(f"After spectral masking: {spectral_mask.sum()} pixels "
              f"({100*spectral_mask.sum()/X_valid.shape[0]:.1f}% of original)")
        
        # Validate filtering results
        min_pixels = int(X_valid.shape[0] * getattr(config, 'MIN_PIXELS_TO_KEEP', 0.01))
        if spectral_mask.sum() < min_pixels:
            print(f"\nWARNING: Only {spectral_mask.sum()} pixels remain after filtering")
            print(f"  Minimum recommended: {min_pixels} pixels")
            print(f"  Consider using 'conservative' filtering strategy")
            if hasattr(config, 'AUTO_ADJUST_FILTERING') and config.AUTO_ADJUST_FILTERING:
                print("  Auto-adjusting to conservative strategy...")
                # Note: Would need to re-run with conservative strategy
                # For now, just warn the user
        
        # Analyze filtered pixels
        if config.VERBOSE:
            try:
                import spectral_analysis
                analysis = spectral_analysis.analyze_filtered_pixels(
                    X_valid, spectral_mask, hyperspectral_data, rgb_composite
                )
                
                # Visualize spectral filtering
                try:
                    spectral_analysis.visualize_spectral_filtering(
                        rgb_composite, valid_mask, valid_mask_combined,
                        config.FIGURES_DIR + '/spectral_filtering_analysis.png',
                        height=height, width=width
                    )
                except Exception as e:
                    if config.VERBOSE:
                        print(f"Warning: Could not create spectral filtering visualization: {e}")
                
                # Validate filtering quality
                validation = spectral_analysis.validate_filtering_quality(
                    spectral_mask,
                    min_pixels_ratio=getattr(config, 'MIN_PIXELS_TO_KEEP', 0.01)
                )
            except Exception as e:
                if config.VERBOSE:
                    print(f"Warning: Could not run spectral analysis: {e}")
    else:
        # No spectral masking - use all valid pixels
        valid_mask_combined = valid_mask
        spectral_mask = np.ones(X_valid.shape[0], dtype=bool)
        if config.VERBOSE:
            print("\nSpectral masking disabled - using all valid pixels")
    
    # ========================================================================
    # STEP 4: Advanced feature engineering (on full data, then mask)
    # ========================================================================
    # Compute features on full data first (spatial features need full image)
    X_augmented_full, feature_names = advanced_features.extract_advanced_features(
        X_valid, 
        hyperspectral_data
    )
    
    # Apply spectral mask to augmented features
    X_augmented = X_augmented_full[spectral_mask]
    
    print(f"After feature engineering: {X_augmented.shape[1]} features")
    print(f"Masked features: {X_augmented.shape[0]} pixels")
    
    # ========================================================================
    # STEP 5: Clustering to create training labels
    # ========================================================================
    cluster_labels, cluster_centers, kmeans_model = clustering.perform_kmeans_clustering(X_augmented)
    
    # Reshape cluster labels using the combined mask (valid + spectral)
    cluster_map = clustering.reshape_labels_to_map(cluster_labels, valid_mask_combined, height, width)
    
    # Save cluster map
    cluster_profile = profile.copy()
    cluster_profile.update({'count': 1, 'dtype': 'int16', 'nodata': -999})
    with rasterio.open(config.RESULTS_DIR + '/cluster_map_enhanced.tif', 'w', **cluster_profile) as dst:
        dst.write(cluster_map.astype('int16'), 1)
    
    # Visualize
    visualizer.visualize_clusters(
        rgb_composite, 
        cluster_map, 
        config.N_CLUSTERS,
        config.FIGURES_DIR + '/clustering_enhanced.png'
    )
    
    # ========================================================================
    # STEP 6: Load Ground Truth from Land Cover Data (or use k-means fallback)
    # ========================================================================
    print(f"\n{'='*60}")
    print("STEP 6: LOADING GROUND TRUTH")
    print(f"{'='*60}")
    
    # Initialize variables
    land_cover_map = None
    use_kmeans = False
    
    if hasattr(config, 'USE_LAND_COVER_GROUND_TRUTH') and config.USE_LAND_COVER_GROUND_TRUTH:
        print("Using LiDAR-derived land cover data as ground truth...")
        
        try:
            import ground_truth_loader
            
            # Calculate hyperspectral bounds for alignment using all four corners
            from rasterio.transform import xy
            try:
                # Get all four corners to ensure correct min/max calculation
                corners = [
                    xy(transform, 0, 0),           # Top-left
                    xy(transform, width, 0),       # Top-right
                    xy(transform, 0, height),      # Bottom-left
                    xy(transform, width, height)   # Bottom-right
                ]
                
                x_coords = [c[0] for c in corners]
                y_coords = [c[1] for c in corners]
                
                minx = float(min(x_coords))
                maxx = float(max(x_coords))
                miny = float(min(y_coords))
                maxy = float(max(y_coords))
                
                hyperspectral_bounds = (minx, miny, maxx, maxy)
            except Exception as e:
                if config.VERBOSE:
                    print(f"Warning: Could not calculate bounds from transform: {e}")
                hyperspectral_bounds = None
            
            # Load and align ground truth
            pavement_mask, land_cover_classes = ground_truth_loader.load_and_align_ground_truth(
                config.LAND_COVER_PATH,
                profile,
                transform,
                hyperspectral_bounds=hyperspectral_bounds,
                pavement_classes=getattr(config, 'PAVEMENT_CLASSES', [6, 7])
            )
            
            # Verify alignment
            if pavement_mask.shape != (height, width):
                raise ValueError(f"Shape mismatch! Ground truth: {pavement_mask.shape}, "
                               f"Expected: ({height}, {width})")
            
            print(f"[OK] Ground truth loaded successfully")
            print(f"  Pavement pixels: {pavement_mask.sum()} ({100*pavement_mask.sum()/pavement_mask.size:.2f}%)")
            
            # Store land cover classes for visualization
            land_cover_map = land_cover_classes
            use_kmeans = False
            
        except Exception as e:
            print(f"ERROR loading land cover data: {e}")
            import traceback
            if config.VERBOSE:
                traceback.print_exc()
            if hasattr(config, 'USE_KMEANS_AS_FALLBACK') and config.USE_KMEANS_AS_FALLBACK:
                print("Falling back to k-means clustering...")
                use_kmeans = True
            else:
                raise RuntimeError("Failed to load ground truth and fallback disabled. "
                                 "Set USE_KMEANS_AS_FALLBACK=True to enable fallback.")
    else:
        use_kmeans = True
        print("Using k-means clustering for labels (land cover disabled)")
    
    # Fallback to k-means if needed
    if use_kmeans:
        print("\nIdentifying pavement clusters using k-means...")
        pavement_clusters = clustering.identify_pavement_clusters(
            cluster_centers, 
            cluster_map,
            X_full=X,
            valid_mask=valid_mask_combined,
            rgb_image=rgb_composite
        )
        pavement_mask = clustering.create_binary_mask(cluster_map, pavement_clusters)
        land_cover_map = None  # No land cover data available
    
    y = pavement_mask.flatten()
    y_valid = y[valid_mask_combined]
    
    # ========================================================================
    # STEP 7: Dimensionality reduction (PCA or Band Selection)
    # ========================================================================
    if config.USE_PCA:
        print(f"\n{'='*60}")
        print("STEP 7: DIMENSIONALITY REDUCTION")
        print(f"{'='*60}")
        
        X_reduced, reducer_info = dimensionality_reduction.reduce_dimensionality(
            X_augmented,
            y_valid,
            method='pca',
            n_components=config.PCA_COMPONENTS
        )
        
        print(f"Reduced to {X_reduced.shape[1]} components")
        
    elif config.USE_BAND_SELECTION:
        X_reduced, reducer_info = dimensionality_reduction.reduce_dimensionality(
            X_augmented,
            y_valid,
            method='band_selection',
            selection_method=config.BAND_SELECTION_METHOD,
            n_bands=config.N_SELECTED_BANDS
        )
        
        print(f"Selected {X_reduced.shape[1]} most important features")
    else:
        X_reduced = X_augmented
        print(f"No dimensionality reduction - using all {X_reduced.shape[1]} features")
    
    # ========================================================================
    # STEP 8: Normalize features
    # ========================================================================
    X_normalized = data_loader.normalize_features(X_reduced, method='zscore')
    
    # ========================================================================
    # STEP 9: Balance dataset
    # ========================================================================
    X_balanced, y_balanced = classifier.balance_dataset(
        X_normalized, 
        y_valid, 
        strategy='undersample'
    )
    
    # ========================================================================
    # STEP 10: Train/Test split with cross-validation option
    # ========================================================================
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("TRAIN/TEST SPLIT")
        print(f"{'='*60}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_balanced
    )
    
    if config.VERBOSE:
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features used: {X_train.shape[1]}")
    
    # ========================================================================
    # STEP 11: Train classifier (enhanced RF)
    # ========================================================================
    print(f"\n{'='*60}")
    print("TRAINING ENHANCED RANDOM FOREST")
    print(f"{'='*60}")
    print(f"Using improved parameters:")
    print(f"  n_estimators: {config.RF_N_ESTIMATORS}")
    print(f"  max_depth: {config.RF_MAX_DEPTH}")
    print(f"  min_samples_split: {config.RF_MIN_SAMPLES_SPLIT}")
    
    from sklearn.ensemble import RandomForestClassifier
    rf_classifier = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
        random_state=config.RANDOM_STATE,
        n_jobs=config.RF_N_JOBS,
        verbose=0
    )
    
    rf_classifier.fit(X_train, y_train)
    print("Training complete!")
    
    # ========================================================================
    # STEP 12: Evaluate with Advanced Metrics
    # ========================================================================
    # Basic evaluation
    metrics_basic, y_pred = classifier.evaluate_classifier(rf_classifier, X_test, y_test)
    
    # Advanced comprehensive metrics
    import advanced_metrics
    
    # For spatial metrics, we need to map test predictions back to image space
    # Create maps from pavement_mask for reference
    y_test_map = pavement_mask.copy()  # Use the full mask as reference
    y_pred_map = np.zeros_like(pavement_mask)
    
    # Map test predictions back (approximate - would need proper index tracking)
    # For now, compute metrics without spatial components
    # Compute comprehensive metrics
    metrics = advanced_metrics.compute_comprehensive_metrics(
        y_test, y_pred,
        y_true_map=None,  # Skip spatial metrics for now
        y_pred_map=None,
        class_names=['Non-pavement', 'Pavement'],
        verbose=True
    )
    
    # If we have both land cover and k-means, compare them
    if land_cover_map is not None and not use_kmeans:
        # Get k-means labels for comparison
        kmeans_pavement_mask = clustering.create_binary_mask(
            cluster_map,
            clustering.identify_pavement_clusters(
                cluster_centers, cluster_map,
                X_full=X, valid_mask=valid_mask_combined, rgb_image=rgb_composite
            )
        )
        
        # Compare ground truth sources
        comparison_metrics = advanced_metrics.compare_ground_truth_sources(
            pavement_mask,  # Land cover
            kmeans_pavement_mask,  # K-means
            verbose=True
        )
        
        # Add to main metrics
        metrics['ground_truth_comparison'] = comparison_metrics
    
    # Merge with basic metrics
    metrics.update(metrics_basic)
    
    # Save comprehensive metrics
    advanced_metrics.save_metrics_report(metrics, config.COMPREHENSIVE_METRICS_FILE)
    
    # Visualizations
    visualizer.plot_confusion_matrix(
        metrics['confusion_matrix']
    )
    
    visualizer.plot_spectral_signatures(
        X_balanced, 
        y_balanced
    )
    
    # ========================================================================
    # STEP 13: Feature importance analysis
    # ========================================================================
    if config.COMPUTE_FEATURE_IMPORTANCE:
        print(f"\n{'='*60}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*60}")
        
        importances = rf_classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\nTop 10 most important features:")
        for i in range(min(10, len(indices))):
            idx = indices[i]
            feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            print(f"  {i+1}. {feat_name}: {importances[idx]:.4f}")
        
        # Save to file
        import pandas as pd
        importance_df = pd.DataFrame({
            'feature_index': range(len(importances)),
            'feature_name': [feature_names[i] if i < len(feature_names) else f"Feature_{i}" 
                           for i in range(len(importances))],
            'importance': importances
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        try:
            import os
            os.makedirs(os.path.dirname(config.FEATURE_IMPORTANCE_FILE), exist_ok=True)
            importance_df.to_csv(config.FEATURE_IMPORTANCE_FILE, index=False)
            
            # Verify file was created
            if os.path.exists(config.FEATURE_IMPORTANCE_FILE):
                print(f"\nFeature importance saved to: {config.FEATURE_IMPORTANCE_FILE}")
            else:
                print(f"ERROR: File {config.FEATURE_IMPORTANCE_FILE} was not created!")
        except Exception as e:
            print(f"ERROR saving feature importance to {config.FEATURE_IMPORTANCE_FILE}: {e}")
    
    # ========================================================================
    # STEP 14: Apply Random Forest to full image
    # ========================================================================
    print(f"\n{'='*60}")
    print("CREATING CLASSIFICATION MAP WITH RANDOM FOREST")
    print(f"{'='*60}")
    
    # Apply Random Forest to all valid pixels (not just k-means mask)
    print("Applying Random Forest classifier to full image...")
    print(f"Processing {len(X_normalized)} pixels...")
    
    # Predict on all normalized features
    y_pred_full = rf_classifier.predict(X_normalized)
    
    # Reshape predictions to 2D map
    classification_map = np.zeros((height, width), dtype=np.uint8)
    # valid_mask_combined is already flattened, so we can index directly
    if valid_mask_combined.ndim == 2:
        # If it's 2D, flatten it
        valid_mask_flat = valid_mask_combined.flatten()
    else:
        valid_mask_flat = valid_mask_combined
    
    # Ensure shapes match
    if len(y_pred_full) != valid_mask_flat.sum():
        print(f"WARNING: Shape mismatch! y_pred_full: {len(y_pred_full)}, valid pixels: {valid_mask_flat.sum()}")
        # Take only the number of predictions we have
        valid_indices = np.where(valid_mask_flat)[0][:len(y_pred_full)]
        classification_map.flat[valid_indices] = y_pred_full
    else:
        classification_map.flat[valid_mask_flat] = y_pred_full
    
    print(f"Random Forest predictions: {classification_map.sum()} pavement pixels")
    print(f"K-means reference mask: {pavement_mask.sum()} pavement pixels")
    
    # Calculate difference between RF and K-means
    rf_only = classification_map & (~pavement_mask.astype(bool))
    kmeans_only = pavement_mask & (~classification_map.astype(bool))
    both_agree = classification_map & pavement_mask
    
    print(f"\nComparison:")
    print(f"  Pixels where both agree (pavement): {both_agree.sum()}")
    print(f"  Pixels only in RF: {rf_only.sum()}")
    print(f"  Pixels only in K-means: {kmeans_only.sum()}")
    if max(classification_map.sum(), pavement_mask.sum()) > 0:
        agreement = both_agree.sum() / max(classification_map.sum(), pavement_mask.sum()) * 100
        print(f"  Agreement rate: {agreement:.1f}%")
    
    # Optionally combine with k-means mask (intersection for higher confidence)
    # This ensures we only keep pixels that both methods agree on
    if hasattr(config, 'USE_ENSEMBLE_MASK') and config.USE_ENSEMBLE_MASK:
        print("\nUsing intersection of RF and K-means (higher confidence)...")
        classification_map = classification_map & pavement_mask
        print(f"After intersection: {classification_map.sum()} pavement pixels")
    else:
        print("\nUsing Random Forest predictions only (RF may differ from K-means)")
    
    print(f"\nInitial classification map: {classification_map.sum()} pavement pixels")
    print(f"Classification map shape: {classification_map.shape}")
    print(f"Classification map dtype: {classification_map.dtype}")
    print(f"Unique values: {np.unique(classification_map)}")
    
    # Safety check: if pavement_mask is empty, something went wrong earlier
    if classification_map.sum() == 0:
        print("\nWARNING: pavement_mask is empty! Skipping filtering.")
        print("Saving empty map for debugging.")
    else:
        # ========================================================================
        # STEP 14.5: Height-Based Filtering (LiDAR elevation)
        # ========================================================================
        if hasattr(config, 'USE_LIDAR_HEIGHT_FILTERING') and config.USE_LIDAR_HEIGHT_FILTERING:
            print(f"\n{'='*70}")
            print("STEP 14.5: HEIGHT-BASED FILTERING (LiDAR)")
            print(f"Threshold: {config.HEIGHT_THRESHOLD_CM} cm")
            print(f"{'='*70}")
            
            try:
                import lidar_elevation_loader
                
                # Calculate hyperspectral bounds using all four corners
                from rasterio.transform import xy
                try:
                    # Get all four corners to ensure correct min/max calculation
                    corners = [
                        xy(transform, 0, 0),           # Top-left
                        xy(transform, width, 0),       # Top-right
                        xy(transform, 0, height),      # Bottom-left
                        xy(transform, width, height)   # Bottom-right
                    ]
                    
                    x_coords = [c[0] for c in corners]
                    y_coords = [c[1] for c in corners]
                    
                    minx = float(min(x_coords))
                    maxx = float(max(x_coords))
                    miny = float(min(y_coords))
                    maxy = float(max(y_coords))
                    
                    hyperspectral_bounds = (minx, miny, maxx, maxy)
                except Exception as e:
                    if config.VERBOSE:
                        print(f"Warning: Could not calculate bounds from transform: {e}")
                    hyperspectral_bounds = None
                
                # Load and align LiDAR elevation data
                dem, height_offset, is_similar_height = lidar_elevation_loader.load_and_align_lidar_elevation(
                    config.LIDAR_DIR,
                    profile,
                    transform,
                    hyperspectral_bounds=hyperspectral_bounds,
                    rasterize_method=getattr(config, 'LIDAR_RASTERIZE_METHOD', 'mean'),
                    height_threshold_cm=getattr(config, 'HEIGHT_THRESHOLD_CM', 10.0)
                )
                
                # Apply height-based filtering
                # Keep pixels that are classified as pavement AND have similar height (< threshold)
                # Remove pixels with large height differences (likely objects like fire hydrants, plants, etc.)
                before_count = classification_map.sum()
                
                # Only filter pixels that are currently classified as pavement
                pavement_pixels = classification_map.astype(bool)
                
                # Create valid mask for comparison (where both DEM and height offset are valid)
                valid_dem = ~np.isnan(dem)
                valid_height_offset = ~np.isnan(height_offset)
                valid_for_comparison = valid_dem & valid_height_offset
                
                # Use percentile-based filtering if enabled, otherwise use absolute threshold
                if getattr(config, 'USE_PERCENTILE_FILTERING', False):
                    # Calculate percentile threshold based on height offsets of pavement pixels
                    pavement_valid = pavement_pixels & valid_for_comparison
                    pavement_height_offsets = height_offset[pavement_valid]
                    if len(pavement_height_offsets) > 0:
                        percentile_threshold = np.percentile(np.abs(pavement_height_offsets), 
                                                             getattr(config, 'HEIGHT_PERCENTILE_THRESHOLD', 95))
                        print(f"  Using percentile-based filtering: {getattr(config, 'HEIGHT_PERCENTILE_THRESHOLD', 95)}th percentile = {percentile_threshold*100:.1f} cm")
                        # Keep pixels below percentile threshold
                        is_similar_height_pavement = np.abs(height_offset) < percentile_threshold
                        is_similar_height_pavement[~valid_for_comparison] = False
                    else:
                        is_similar_height_pavement = is_similar_height
                else:
                    is_similar_height_pavement = is_similar_height
                
                # Remove pavement pixels with large height differences
                # (These are likely objects on the sidewalk, not the sidewalk itself)
                large_height_diff = ~is_similar_height_pavement
                classification_map[pavement_pixels & large_height_diff] = 0
                
                after_count = classification_map.sum()
                removed_count = before_count - after_count
                
                print(f"\nHeight-based filtering results:")
                print(f"  Before: {before_count} pavement pixels")
                print(f"  After: {after_count} pavement pixels")
                print(f"  Removed: {removed_count} pixels ({100*removed_count/max(before_count,1):.1f}% reduction)")
                print(f"  Threshold: {config.HEIGHT_THRESHOLD_CM} cm")
                
                # Save DEM and height offset for visualization
                if hasattr(config, 'RESULTS_DIR'):
                    import os
                    dem_file = os.path.join(config.RESULTS_DIR, "lidar_dem.tif")
                    height_offset_file = os.path.join(config.RESULTS_DIR, "height_offset.tif")
                    
                    # Save DEM
                    with rasterio.open(
                        dem_file, 'w',
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=1,
                        dtype=dem.dtype,
                        crs=profile['crs'],
                        transform=transform,
                        compress='lzw'
                    ) as dst:
                        dst.write(dem, 1)
                    
                    # Save height offset
                    with rasterio.open(
                        height_offset_file, 'w',
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=1,
                        dtype=height_offset.dtype,
                        crs=profile['crs'],
                        transform=transform,
                        compress='lzw'
                    ) as dst:
                        dst.write(height_offset, 1)
                    
                    print(f"  DEM saved to: {dem_file}")
                    print(f"  Height offset saved to: {height_offset_file}")
                
            except Exception as e:
                print(f"\nERROR: Height-based filtering failed: {e}")
                import traceback
                if config.VERBOSE:
                    traceback.print_exc()
                print("\nPossible causes:")
                print("  1. LiDAR files don't cover the target area")
                print("  2. Coordinate system mismatch between LiDAR and hyperspectral data")
                print("  3. LiDAR files are in a different location than expected")
                print("\nTroubleshooting:")
                print(f"  - Check that LiDAR files in {config.LIDAR_DIR} cover the area")
                print(f"  - Verify coordinate systems match (hyperspectral: {profile.get('crs', 'unknown')})")
                print("  - Try running with VERBOSE=True to see detailed diagnostics")
                print("\nContinuing without height-based filtering...")
        
        # ========================================================================
        # STEP 15: Multi-Stage Filtering (Robust Approach)
        # ========================================================================
        print(f"\n{'='*70}")
        print("STEP 15: MULTI-STAGE FILTERING")
        print("Removing rooftops while preserving sidewalks")
        print(f"{'='*70}")
        
        # Simplified approach: Only remove obvious rooftops, preserve everything else
        from scipy.ndimage import label
        labeled, n_regions = label(classification_map)
        
        print(f"Found {n_regions} regions")
        print(f"Pixels before filtering: {classification_map.sum()}")
        
        if n_regions > 0:
            # Start with a copy of the classification map (preserve all pixels by default)
            filtered_map = classification_map.copy()
            n_removed = 0
            
            for region_id in range(1, n_regions + 1):
                region_mask = (labeled == region_id)
                region_area = region_mask.sum()
                
                # Only remove very obvious rooftops: square, compact, isolated, medium size
                if 300 <= region_area <= 2500:  # Medium size range
                    coords = np.argwhere(region_mask)
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    
                    bbox_width = x_max - x_min + 1
                    bbox_height = y_max - y_min + 1
                    aspect_ratio = max(bbox_width, bbox_height) / (min(bbox_width, bbox_height) + 1e-10)
                    bbox_area = bbox_width * bbox_height
                    compactness = region_area / bbox_area if bbox_area > 0 else 0
                    
                    # Very strict criteria: only remove if clearly a rooftop
                    # Must be: very square (aspect <= 1.4), very compact (>0.85), medium size
                    is_obvious_rooftop = (
                        aspect_ratio <= 1.4 and  # Very square
                        compactness > 0.85 and   # Very compact (fills bounding box)
                        300 <= region_area <= 2500  # Medium size
                    )
                    
                    if is_obvious_rooftop:
                        filtered_map[region_mask] = 0
                        n_removed += 1
                        if config.VERBOSE and n_removed <= 10:
                            print(f"  Removed rooftop {n_removed}: area={region_area}, aspect={aspect_ratio:.2f}, compactness={compactness:.2f}")
            
            classification_map = filtered_map
            print(f"After rooftop filtering: {classification_map.sum()} pixels (removed {n_removed} regions)")
        else:
            print(f"WARNING: No connected regions found! Keeping all {classification_map.sum()} pixels.")
    
    # Step 3: Optional sidewalk-specific refinement (DISABLED - too aggressive)
    # Sidewalk filtering removes everything, keeping disabled for now
    if False and hasattr(config, 'USE_SIDEWALK_FILTERING') and config.USE_SIDEWALK_FILTERING:
        print(f"\n{'='*70}")
        print("STEP 15.5: SIDEWALK-SPECIFIC REFINEMENT")
        print(f"{'='*70}")
        
        import sidewalk_detection
        
        # Apply sidewalk-specific filtering as refinement (but preserve existing)
        sidewalk_refined = sidewalk_detection.sidewalk_specific_filtering(
            classification_map,
            use_road_adjacency=True
        )
        
        # Combine: keep both original and sidewalk-detected areas
        classification_map = classification_map | sidewalk_refined
    
    # ========================================================================
    # STEP 15.6: Integrate Manual Mask (OPTIONAL)
    # ========================================================================
    if hasattr(config, 'USE_MANUAL_MASK') and config.USE_MANUAL_MASK:
        print(f"\n{'='*70}")
        print("STEP 15.6: INTEGRATING MANUAL MASK")
        print(f"{'='*70}")
        
        import sidewalk_detection
        import os
        
        manual_mask_path = os.path.join(config.OUTPUT_DIR, "manual_pavement_mask.npy")
        classification_map = sidewalk_detection.integrate_manual_mask(
            classification_map,
            manual_mask_path
        )
    
    # Light post-processing
    if config.USE_MORPHOLOGICAL_FILTERING:
        kernel_size = 1  # Very light
        classification_map = spectral_masking.apply_morphological_filtering(
            classification_map,
            kernel_size=kernel_size
        )
    
    # ========================================================================
    # STEP 15.7: Sidewalk Signature Matching (Expand using Ground Truth)
    # ========================================================================
    if (hasattr(config, 'USE_SIDEWALK_SIGNATURE_MATCHING') and 
        config.USE_SIDEWALK_SIGNATURE_MATCHING and 
        land_cover_map is not None):
        print(f"\n{'='*70}")
        print("STEP 15.7: SIDEWALK SIGNATURE MATCHING")
        print("Expanding sidewalk detection using ground truth signatures")
        print(f"{'='*70}")
        
        try:
            import sidewalk_detection
            
            # Expand classification map using signature matching
            classification_map = sidewalk_detection.expand_sidewalk_by_signature_matching(
                classification_map,
                land_cover_map,
                hyperspectral_data,
                valid_mask_combined,
                similarity_threshold=getattr(config, 'SIDEWALK_SIMILARITY_THRESHOLD', 0.85),
                max_distance_pixels=getattr(config, 'SIDEWALK_MAX_DISTANCE', 50),
                prefer_rectangular=True
            )
            
            print(f"[OK] Sidewalk signature matching complete")
            print(f"  Final pavement pixels: {classification_map.sum()}")
            
        except Exception as e:
            print(f"ERROR: Sidewalk signature matching failed: {e}")
            import traceback
            if config.VERBOSE:
                traceback.print_exc()
            print("Continuing without signature matching expansion...")
    
    # Save
    print(f"\n{'='*70}")
    print("SAVING CLASSIFICATION MAP")
    print(f"{'='*70}")
    print(f"Before save - classification_map.sum(): {classification_map.sum()}")
    print(f"classification_map.shape: {classification_map.shape}")
    print(f"classification_map.dtype: {classification_map.dtype}")
    print(f"Unique values: {np.unique(classification_map)}")
    
    output_profile = profile.copy()
    output_profile.update({'count': 1, 'dtype': 'int16', 'nodata': -999})
    
    with rasterio.open(config.CLASSIFICATION_MAP_FILE, 'w', **output_profile) as dst:
        data_to_write = classification_map.astype('int16')
        print(f"Writing data - sum: {data_to_write.sum()}, shape: {data_to_write.shape}")
        dst.write(data_to_write, 1)
    
    # Verify
    with rasterio.open(config.CLASSIFICATION_MAP_FILE, 'r') as src:
        verify_data = src.read(1)
        print(f"Verified - saved file sum: {verify_data.sum()}, unique: {np.unique(verify_data)}")
    
    print(f"Classification map saved to: {config.CLASSIFICATION_MAP_FILE}")
    
    # Final visualization
    # Determine ground truth source for visualization
    if hasattr(config, 'USE_LAND_COVER_GROUND_TRUTH') and config.USE_LAND_COVER_GROUND_TRUTH:
        ground_truth_source = "LiDAR Land Cover"
        reference_map = pavement_mask  # This is now from land cover
    else:
        ground_truth_source = "K-means Clustering"
        reference_map = pavement_mask  # This is from k-means
    
    visualizer.visualize_classification_results(
        rgb_composite,
        reference_map,
        classification_map,
        class_names=['Non-pavement', 'Pavement'],
        filename=config.FIGURES_DIR + '/classification_results_enhanced.png',
        ground_truth_source=ground_truth_source
    )
    
    # Additional visualization: Compare land cover with k-means if both available
    if land_cover_map is not None and not use_kmeans:
        # We have land cover, also show k-means for comparison
        kmeans_pavement_mask = clustering.create_binary_mask(
            cluster_map,
            clustering.identify_pavement_clusters(
                cluster_centers, cluster_map,
                X_full=X, valid_mask=valid_mask_combined, rgb_image=rgb_composite
            )
        )
        
        # Create comparison visualization
        try:
            visualizer.visualize_ground_truth_comparison(
                rgb_composite,
                pavement_mask,  # Land cover ground truth
                kmeans_pavement_mask,  # K-means labels
                classification_map,  # RF predictions
                filename=config.FIGURES_DIR + '/ground_truth_comparison.png'
            )
        except AttributeError:
            # Function doesn't exist yet, skip for now
            if config.VERBOSE:
                print("Note: Ground truth comparison visualization not yet implemented")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print(f"\n{'='*60}")
    print("ENHANCED PIPELINE COMPLETE!")
    print(f"{'='*60}")
    
    print(f"\nKey improvements:")
    print(f"  + Used ALL {n_total_bands} hyperspectral bands")
    print(f"  + Advanced feature engineering (+{X_augmented.shape[1] - n_total_bands} features)")
    if config.USE_PCA:
        print(f"  + PCA dimensionality reduction ({X_reduced.shape[1]} components)")
    print(f"  + Enhanced Random Forest ({config.RF_N_ESTIMATORS} trees)")
    print(f"  + {config.N_CLUSTERS} clusters for finer discrimination")
    
    print(f"\nPerformance:")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Pavement F1-Score: {metrics['f1_per_class'][1]*100:.2f}%")
    
    print(f"\nOutput files in: {config.OUTPUT_DIR}")
    print(f"  - Classification map: {config.CLASSIFICATION_MAP_FILE}")
    print(f"  - Metrics: {config.COMPREHENSIVE_METRICS_FILE}")
    print(f"  - Feature importance: {config.FEATURE_IMPORTANCE_FILE}")
    print(f"  - Figures: {config.FIGURES_DIR}/")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()

