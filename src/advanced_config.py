"""
Advanced configuration for research-grade hyperspectral pavement classification.

This configuration supports:
- Full hyperspectral bands (288)
- Multiple classifiers and ensemble methods
- Advanced feature engineering
- Dimensionality reduction techniques
- Statistical validation
"""

import os

# ============================================================================
# FILE PATHS
# ============================================================================
HYPERSPECTRAL_PATH = "NYU_TuckMapping_20190511_Mission1_NE-SW_288Bands.pix"
OUTPUT_DIR = "outputs_research"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")

# Create output directories
for dir_path in [OUTPUT_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR, ANALYSIS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# DATA EXTRACTION PARAMETERS
# ============================================================================
# Test area size - optimized for best results
TEST_SIZE_X = 800  # Larger area for maximum statistical validity
TEST_SIZE_Y = 800
START_X = 2500  # Different block: try area with more sidewalks and roads
START_Y = 3000

# Hyperspectral bands
USE_ALL_BANDS = True  # Use all 288 bands
NUM_BANDS = None  # None = use all when USE_ALL_BANDS is True

# Alternative: Sample specific bands if USE_ALL_BANDS is False
BAND_INDICES = list(range(0, 288, 3))  # Every 3rd band as alternative

# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================
USE_PCA = True
PCA_COMPONENTS = 75  # More components for richer feature representation
PCA_VARIANCE_THRESHOLD = 0.995  # Retain 99.5% variance for maximum information

# Band selection methods
USE_BAND_SELECTION = True
BAND_SELECTION_METHOD = 'importance'  # 'importance', 'correlation', 'mutual_info'
N_SELECTED_BANDS = 100  # Number of bands to select

# ============================================================================
# RGB VISUALIZATION PARAMETERS
# ============================================================================
# For 288 bands, try different combinations for better visualization
# Option 1: NIR-Red-Green (false color, good for urban areas)
RED_BAND_IDX = 70     # NIR band for false color
GREEN_BAND_IDX = 50   # Red band for false color
BLUE_BAND_IDX = 20    # Green band for false color

# Option 2: True visible spectrum (if bands correspond)
# RED_BAND_IDX = 60
# GREEN_BAND_IDX = 40
# BLUE_BAND_IDX = 15

# Option 3: Sample across full range
# RED_BAND_IDX = 200
# GREEN_BAND_IDX = 100
# BLUE_BAND_IDX = 30

# Increased contrast for better visualization
RGB_PERCENTILE_LOW = 1
RGB_PERCENTILE_HIGH = 99

# ============================================================================
# CLUSTERING PARAMETERS
# ============================================================================
N_CLUSTERS = 7  # More clusters for finer surface type discrimination
MAX_CLUSTERING_SAMPLES = 50000
RANDOM_STATE = 42

# ============================================================================
# SPECTRAL MASKING
# ============================================================================
USE_SPECTRAL_MASKING = True  # Filter obvious non-pavement before classification

# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================
# Spectral indices
USE_SPECTRAL_INDICES = True
USE_VEGETATION_INDICES = True  # NDVI, EVI, SAVI
USE_URBAN_INDICES = True       # NDBI, UI, IBI
USE_BRIGHTNESS_INDICES = True   # Brightness, greenness, wetness

# Spatial features
USE_SPATIAL_FEATURES = True
SPATIAL_WINDOW_SIZE = 3

# Sidewalk-specific spectral features
USE_PAVEMENT_SPECTRAL_SIGNATURE = True  # Use specialized features for sidewalk detection  # 3x3 window for spatial statistics
USE_TEXTURE_FEATURES = False  # GLCM textures (computationally expensive)

# Statistical features
USE_STATISTICAL_FEATURES = True  # Mean, std, skewness, kurtosis per pixel

# ============================================================================
# CLASSIFICATION PARAMETERS
# ============================================================================
# Multiple classifiers for ensemble
USE_RANDOM_FOREST = True
USE_SVM = True
USE_XGBOOST = True
USE_ENSEMBLE = True  # Combine multiple classifiers

# Random Forest
RF_N_ESTIMATORS = 300  # Maximum for best accuracy
RF_MAX_DEPTH = 30
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2
RF_N_JOBS = -1

# SVM
SVM_KERNEL = 'rbf'  # 'linear', 'rbf', 'poly'
SVM_C = 10.0
SVM_GAMMA = 'scale'

# XGBoost
XGB_N_ESTIMATORS = 300  # Maximum for best accuracy
XGB_MAX_DEPTH = 10
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8

# Ensemble
ENSEMBLE_METHOD = 'voting'  # 'voting', 'stacking'
ENSEMBLE_VOTING = 'soft'  # 'hard', 'soft'

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
# Cross-validation
USE_CROSS_VALIDATION = True
CV_FOLDS = 5  # 5-fold cross-validation
CV_STRATIFIED = True

# Train/test split
TEST_SIZE = 0.3
MAX_SAMPLES_PER_CLASS = 50000

# Class balancing
BALANCE_METHOD = 'undersample'  # 'undersample', 'oversample', 'smote'

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================
# Metrics to compute
COMPUTE_CONFUSION_MATRIX = True
COMPUTE_ROC_CURVE = True
COMPUTE_PR_CURVE = True
COMPUTE_FEATURE_IMPORTANCE = True
COMPUTE_CLASS_SEPARABILITY = True

# Statistical testing
USE_MCNEMAR_TEST = True  # Test if classifiers significantly different
USE_KAPPA_STATISTICS = True  # Cohen's kappa
CONFIDENCE_LEVEL = 0.95

# ============================================================================
# SPECTRAL MASKING PARAMETERS (Enhanced)
# ============================================================================
USE_SPECTRAL_MASKING = True  # Filter obvious non-pavement before training
SPECTRAL_FILTERING_STRATEGY = 'moderate'  # Options: 'conservative', 'moderate', 'aggressive'

# NDVI filtering
NDVI_THRESHOLD = 0.3  # Filter pixels with NDVI >= this (vegetation)
USE_NDVI_FILTER = True

# Brightness filtering
BRIGHTNESS_LOW_PERCENTILE = 5  # Filter bottom X% (shadows/water)
BRIGHTNESS_HIGH_PERCENTILE = 99  # Filter top X% (bright roofs)
USE_BRIGHTNESS_FILTER = True
USE_PERCENTILE_BRIGHTNESS = True  # If False, use absolute thresholds
BRIGHTNESS_LOW_ABSOLUTE = None  # Absolute low threshold (if not using percentile)
BRIGHTNESS_HIGH_ABSOLUTE = None  # Absolute high threshold

# Spectral variance filtering
SPECTRAL_VARIANCE_PERCENTILE = 90  # Filter top X% variance
USE_VARIANCE_FILTER = True
USE_PERCENTILE_VARIANCE = True
VARIANCE_ABSOLUTE_THRESHOLD = None

# Minimum pixels to keep
MIN_PIXELS_TO_KEEP = 0.01  # Keep at least 1% of pixels (safety check)
AUTO_ADJUST_FILTERING = False  # Auto-adjust to conservative if too aggressive

# Preset configurations for different scenarios
SPECTRAL_FILTERING_PRESETS = {
    'urban_dense': {
        'strategy': 'aggressive',
        'ndvi_threshold': 0.25,
        'brightness_low_percentile': 10
    },
    'urban_sparse': {
        'strategy': 'moderate',
        'ndvi_threshold': 0.3,
        'brightness_low_percentile': 5
    },
    'suburban': {
        'strategy': 'conservative',
        'ndvi_threshold': 0.35,
        'brightness_low_percentile': 2
    }
}

# ============================================================================
# POST-PROCESSING PARAMETERS
# ============================================================================
USE_MORPHOLOGICAL_FILTERING = True
MORPHOLOGICAL_KERNEL_SIZE = 3
USE_MAJORITY_FILTER = True
MAJORITY_FILTER_SIZE = 5

# ============================================================================
# SIDEWALK-SPECIFIC FILTERING
# ============================================================================
USE_SIDEWALK_FILTERING = True  # Set to True to focus on sidewalks (narrow linear features)
SIDEWALK_MIN_WIDTH = 1  # Minimum sidewalk width in pixels
SIDEWALK_MAX_WIDTH = 8  # Maximum sidewalk width in pixels
SIDEWALK_MIN_LENGTH = 10  # Minimum sidewalk length in pixels

# SIDEWALK SIGNATURE MATCHING (Expand using Ground Truth)
# ============================================================================
USE_SIDEWALK_SIGNATURE_MATCHING = True  # Expand sidewalk detection using ground truth signatures
SIDEWALK_SIMILARITY_THRESHOLD = 0.80  # Minimum cosine similarity to match (0-1) - lowered for more matches
SIDEWALK_MAX_DISTANCE = 100  # Maximum distance from existing sidewalk to expand (pixels) - increased
USE_ROAD_ADJACENCY = True  # Use road adjacency to identify sidewalks

# ============================================================================
# MANUAL MASKING
# ============================================================================
USE_MANUAL_MASK = False  # Set to True to integrate manual mask (disabled for now)
MANUAL_MASK_PATH = os.path.join(OUTPUT_DIR, "manual_pavement_mask.npy")

# ============================================================================
# OUTPUT PARAMETERS
# ============================================================================
# Output files
CLASSIFICATION_MAP_FILE = os.path.join(RESULTS_DIR, "pavement_classification_advanced.tif")
PCA_COMPONENTS_FILE = os.path.join(MODELS_DIR, "pca_model.pkl")
CLASSIFIER_FILE = os.path.join(MODELS_DIR, "classifier_ensemble.pkl")
FEATURE_IMPORTANCE_FILE = os.path.join(ANALYSIS_DIR, "feature_importance.csv")
CV_RESULTS_FILE = os.path.join(ANALYSIS_DIR, "cv_results.csv")
STATISTICAL_TESTS_FILE = os.path.join(ANALYSIS_DIR, "statistical_tests.txt")
COMPREHENSIVE_METRICS_FILE = os.path.join(ANALYSIS_DIR, "comprehensive_metrics.txt")

# Figures
ROC_CURVE_FILE = os.path.join(FIGURES_DIR, "roc_curves.png")
PR_CURVE_FILE = os.path.join(FIGURES_DIR, "precision_recall_curves.png")
FEATURE_IMPORTANCE_PLOT = os.path.join(FIGURES_DIR, "feature_importance.png")
CLASSIFIER_COMPARISON_FILE = os.path.join(FIGURES_DIR, "classifier_comparison.png")
PCA_VARIANCE_PLOT = os.path.join(FIGURES_DIR, "pca_variance_explained.png")
CLASS_SEPARABILITY_PLOT = os.path.join(FIGURES_DIR, "class_separability.png")

FIGURE_DPI = 400  # Higher DPI for better visualization quality

# ============================================================================
# COMPUTATIONAL PARAMETERS
# ============================================================================
# Memory management
USE_MEMORY_MAPPING = False  # For very large datasets
PROCESSING_BATCH_SIZE = 10000

# Parallel processing
N_JOBS = -1  # Use all available cores
VERBOSE = True

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
RANDOM_STATE = 42
import numpy as np
import random
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ============================================================================
# GROUND TRUTH DATA
# ============================================================================
# Use LiDAR-derived land cover data as ground truth (replaces k-means labels)
USE_LAND_COVER_GROUND_TRUTH = True
# Land cover path - handle special characters in directory name
LAND_COVER_BASE_DIR = "Land Cover Raster Data (2017) – 6in Resolution"
LAND_COVER_PATH = os.path.join(LAND_COVER_BASE_DIR, "Land_Cover", "NYC_2017_LiDAR_LandCover.img")
PAVEMENT_CLASSES = [6, 7]  # Class 6: Roads, Class 7: Other Impervious
USE_KMEANS_AS_FALLBACK = False  # Set to True if land cover unavailable

# ============================================================================
# LIDAR ELEVATION DATA FOR HEIGHT-BASED FILTERING
# ============================================================================
# Use LiDAR elevation data for height-based filtering
USE_LIDAR_HEIGHT_FILTERING = True
LIDAR_DIR = r"C:\Users\katha\Downloads\lidar"  # Path to directory containing .laz files
HEIGHT_THRESHOLD_CM = 50.0  # Height difference threshold in centimeters (increased to 50cm for less aggressive filtering)
# Pixels with height difference < threshold are considered similar (likely sidewalk)
# Pixels with height difference > threshold are considered objects (fire hydrant, plant, etc.)
# Note: Increased to 50cm to account for natural sidewalk variation and avoid over-filtering
# Only remove obvious elevated objects (>50cm), not normal sidewalk texture variation
USE_GROUND_LEVEL_FILTERING = True  # If True, compare to estimated ground level (DTM) to distinguish rooftops from sidewalks
# Ground level mode: Rooftops are typically >5m above ground, sidewalks are at ground level
# This is more effective than local neighborhood comparison for rooftop removal
ROOFTOP_EXCLUSION_THRESHOLD_M = 5.0  # Exclude pixels more than this many meters above ground (rooftops) - increased to be less aggressive
USE_LOCAL_VARIATION_FILTER = False  # Disable local variation check - too aggressive, removes legitimate sidewalks
LIDAR_RASTERIZE_METHOD = 'mean'  # Method for rasterizing points: 'mean', 'min', 'max'
USE_PERCENTILE_FILTERING = False  # If True, use percentile-based filtering instead of absolute threshold
HEIGHT_PERCENTILE_THRESHOLD = 95  # Keep pixels below this percentile of height differences
SAVE_PRE_FILTER_RESULTS = True  # Save classification map before LiDAR filtering for comparison

# ============================================================================
# RESEARCH METADATA
# ============================================================================
EXPERIMENT_NAME = "Hyperspectral_Pavement_Classification"
EXPERIMENT_VERSION = "2.0_Research_Grade"
DATASET_NAME = "NYU_TuckMapping_Brooklyn_2019"
AUTHOR = "Research Team"
DATE = "2025-10-28"

print(f"Advanced configuration loaded: {EXPERIMENT_NAME} v{EXPERIMENT_VERSION}")

