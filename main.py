"""
Main pipeline for hyperspectral pavement classification.

This script orchestrates the complete workflow:
1. Load hyperspectral data
2. Create RGB visualization
3. Perform clustering to identify surface types
4. Create training labels from clustering
5. Train Random Forest classifier
6. Evaluate and apply classifier
7. Generate visualizations and save results

Usage:
    python main.py
"""

import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import config
import data_loader
import visualizer
import feature_engineering
import clustering
import classifier


def main():
    """Main pipeline execution."""
    
    print("="*60)
    print("HYPERSPECTRAL PAVEMENT CLASSIFICATION PIPELINE")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data file: {config.HYPERSPECTRAL_PATH}")
    print(f"  Test area: {config.TEST_SIZE_X}x{config.TEST_SIZE_Y} pixels")
    print(f"  Starting position: ({config.START_X}, {config.START_Y})")
    print(f"  Number of bands: {config.NUM_BANDS}")
    print(f"  Output directory: {config.OUTPUT_DIR}")
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    hyperspectral_data, profile, transform = data_loader.load_hyperspectral_data()
    
    height = hyperspectral_data.shape[1]
    width = hyperspectral_data.shape[2]
    
    # ========================================================================
    # STEP 2: Create and Save RGB Composite
    # ========================================================================
    rgb_composite = visualizer.create_rgb_composite(hyperspectral_data)
    visualizer.save_rgb_composite(rgb_composite)
    
    # ========================================================================
    # STEP 3: Reshape Data for ML
    # ========================================================================
    X, valid_mask = data_loader.reshape_for_ml(hyperspectral_data)
    
    # Use only valid pixels for clustering
    X_valid = X[valid_mask]
    
    # ========================================================================
    # STEP 4: Perform K-means Clustering
    # ========================================================================
    cluster_labels, cluster_centers, kmeans_model = clustering.perform_kmeans_clustering(X_valid)
    
    # Reshape cluster labels to 2D map
    cluster_map = clustering.reshape_labels_to_map(cluster_labels, valid_mask, height, width)
    
    # Save cluster map as GeoTIFF
    cluster_profile = profile.copy()
    cluster_profile.update({
        'count': 1,
        'dtype': 'int16',
        'nodata': -999
    })
    
    with rasterio.open(config.CLUSTER_MAP_FILE, 'w', **cluster_profile) as dst:
        dst.write(cluster_map.astype('int16'), 1)
    
    if config.VERBOSE:
        print(f"\nCluster map saved to: {config.CLUSTER_MAP_FILE}")
    
    # Visualize clustering results
    visualizer.visualize_clusters(rgb_composite, cluster_map, config.N_CLUSTERS)
    
    # ========================================================================
    # STEP 5: Identify Pavement Clusters
    # ========================================================================
    pavement_clusters = clustering.identify_pavement_clusters(cluster_centers, cluster_map)
    
    # Create binary pavement mask
    pavement_mask = clustering.create_binary_mask(cluster_map, pavement_clusters)
    
    # Flatten mask for training
    y = pavement_mask.flatten()
    y_valid = y[valid_mask]
    
    # ========================================================================
    # STEP 6: Feature Engineering
    # ========================================================================
    if config.USE_SPECTRAL_INDICES:
        X_augmented = feature_engineering.augment_features(X_valid)
    else:
        X_augmented = X_valid
    
    # Normalize features
    X_normalized = data_loader.normalize_features(X_augmented, method='zscore')
    
    # ========================================================================
    # STEP 7: Balance Dataset
    # ========================================================================
    X_balanced, y_balanced = classifier.balance_dataset(X_normalized, y_valid, strategy='undersample')
    
    # ========================================================================
    # STEP 8: Train/Test Split
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
    
    # ========================================================================
    # STEP 9: Train Classifier
    # ========================================================================
    rf_classifier = classifier.train_random_forest(X_train, y_train)
    
    # ========================================================================
    # STEP 10: Evaluate Classifier
    # ========================================================================
    metrics, y_pred = classifier.evaluate_classifier(rf_classifier, X_test, y_test)
    
    # Save metrics to file
    classifier.save_metrics_to_file(metrics)
    
    # Plot confusion matrix
    visualizer.plot_confusion_matrix(metrics['confusion_matrix'])
    
    # Plot spectral signatures
    visualizer.plot_spectral_signatures(X_balanced, y_balanced)
    
    # ========================================================================
    # STEP 11: Apply Classifier to Full Image
    # ========================================================================
    
    # First, we need to normalize ALL data (not just valid pixels)
    # We'll use the same normalization parameters from training
    if config.USE_SPECTRAL_INDICES:
        X_full = feature_engineering.augment_features(X)
    else:
        X_full = X
    
    X_full_normalized = data_loader.normalize_features(X_full, method='zscore')
    
    # Apply classifier
    predictions = classifier.apply_classifier(rf_classifier, X_full_normalized)
    
    # Reshape to 2D map
    classification_map = predictions.reshape(height, width)
    
    # ========================================================================
    # STEP 12: Save Classification Map
    # ========================================================================
    output_profile = profile.copy()
    output_profile.update({
        'count': 1,
        'dtype': 'int16',
        'nodata': -999
    })
    
    with rasterio.open(config.CLASSIFICATION_MAP_FILE, 'w', **output_profile) as dst:
        dst.write(classification_map.astype('int16'), 1)
    
    if config.VERBOSE:
        print(f"\nClassification map saved to: {config.CLASSIFICATION_MAP_FILE}")
    
    # ========================================================================
    # STEP 13: Visualize Final Results
    # ========================================================================
    visualizer.visualize_classification_results(
        rgb_composite,
        pavement_mask,
        classification_map,
        class_names=['Non-pavement', 'Pavement']
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print("\nOutput files:")
    print(f"  - Classification map: {config.CLASSIFICATION_MAP_FILE}")
    print(f"  - Cluster map: {config.CLUSTER_MAP_FILE}")
    print(f"  - RGB composite: {config.RGB_COMPOSITE_FILE}")
    print(f"  - Clustering results: {config.CLUSTERING_RESULTS_FILE}")
    print(f"  - Classification results: {config.CLASSIFICATION_RESULTS_FILE}")
    print(f"  - Metrics: {config.METRICS_FILE}")
    print(f"\nAll outputs saved to: {config.OUTPUT_DIR}")
    
    print("\n" + "="*60)
    print("NEXT STEPS & RECOMMENDATIONS")
    print("="*60)
    print("\n1. REVIEW CLUSTERING RESULTS:")
    print(f"   Open: {config.CLUSTERING_RESULTS_FILE}")
    print("   - Check if pavement clusters were correctly identified")
    print("   - If not, manually adjust 'pavement_clusters' in the code")
    
    print("\n2. ADJUST PARAMETERS:")
    print("   Edit config.py to modify:")
    print("   - RGB band indices for better visualization")
    print("   - Number of clusters")
    print("   - Random Forest parameters")
    print("   - Test area size and location")
    
    print("\n3. IMPROVE CLASSIFICATION:")
    print("   - Try different test areas to get diverse training data")
    print("   - Use more/fewer spectral bands")
    print("   - Enable/disable spectral indices")
    print("   - Adjust cluster selection criteria")
    
    print("\n4. PROCESS FULL IMAGE:")
    print("   - Increase TEST_SIZE_X and TEST_SIZE_Y in config.py")
    print("   - Or process multiple tiles and mosaic results")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

