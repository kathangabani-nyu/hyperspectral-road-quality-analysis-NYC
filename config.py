"""
Configuration file for hyperspectral pavement classification.
All parameters and settings in one place for easy modification.
"""

import os

# ============================================================================
# FILE PATHS
# ============================================================================
HYPERSPECTRAL_PATH = "NYU_TuckMapping_20190511_Mission1_NE-SW_288Bands.pix"
OUTPUT_DIR = "outputs"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================================
# DATA EXTRACTION PARAMETERS
# ============================================================================
# Test area size (in pixels)
TEST_SIZE_X = 200  # Reduced even more for faster testing with 29GB file
TEST_SIZE_Y = 200  # Reduced even more for faster testing with 29GB file

# Starting coordinates for test area
START_X = 2000  # Moving to middle of image where data exists
START_Y = 3000  # Moving to middle of image where data exists

# Number of hyperspectral bands to use (reduce for speed, use more for accuracy)
# Set to None to use all bands, or specify a range
NUM_BANDS = 30  # Reduced to 30 for faster testing with large file
# BAND_INDICES = list(range(0, 288, 3))  # Alternative: sample every 3rd band

# ============================================================================
# RGB VISUALIZATION PARAMETERS
# ============================================================================
# Band indices for RGB composite (0-indexed in the selected bands)
# Adjust based on which bands you're using and your sensor
RED_BAND_IDX = 25  # Adjusted for 30 bands
GREEN_BAND_IDX = 15  # Adjusted for 30 bands
BLUE_BAND_IDX = 5

# Percentile-based contrast stretching (helps avoid black images)
RGB_PERCENTILE_LOW = 2   # Lower percentile for contrast stretch
RGB_PERCENTILE_HIGH = 98  # Upper percentile for contrast stretch

# ============================================================================
# CLUSTERING PARAMETERS
# ============================================================================
# Number of clusters for K-means
N_CLUSTERS = 5

# Maximum number of pixels to sample for clustering (for speed)
MAX_CLUSTERING_SAMPLES = 20000  # Reduced for faster testing

# Random seed for reproducibility
RANDOM_STATE = 42

# ============================================================================
# CLASSIFICATION PARAMETERS
# ============================================================================
# Random Forest parameters
RF_N_ESTIMATORS = 50  # Reduced from 100 for faster testing
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_SPLIT = 10
RF_N_JOBS = -1  # Use all CPU cores

# Train/test split
TEST_SIZE = 0.3

# Maximum training samples per class (for balanced training)
MAX_SAMPLES_PER_CLASS = 20000  # Reduced for faster testing

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================
# Whether to use additional spectral indices
USE_SPECTRAL_INDICES = True

# Whether to apply spatial smoothing
USE_SPATIAL_SMOOTHING = False
SMOOTHING_SIGMA = 1.0

# ============================================================================
# OUTPUT PARAMETERS
# ============================================================================
# Output file names
CLASSIFICATION_MAP_FILE = os.path.join(RESULTS_DIR, "pavement_classification_map.tif")
CLUSTER_MAP_FILE = os.path.join(RESULTS_DIR, "cluster_map.tif")
RGB_COMPOSITE_FILE = os.path.join(FIGURES_DIR, "rgb_composite.png")
CLUSTERING_RESULTS_FILE = os.path.join(FIGURES_DIR, "clustering_results.png")
CLASSIFICATION_RESULTS_FILE = os.path.join(FIGURES_DIR, "classification_results.png")
METRICS_FILE = os.path.join(RESULTS_DIR, "classification_metrics.txt")

# Figure DPI
FIGURE_DPI = 300

# ============================================================================
# LOGGING PARAMETERS
# ============================================================================
VERBOSE = True  # Print progress messages
LOG_FILE = os.path.join(OUTPUT_DIR, "processing_log.txt")

