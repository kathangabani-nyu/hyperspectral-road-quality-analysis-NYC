"""
Visualization functions for hyperspectral data.
Includes proper normalization to avoid black/unclear images.
"""

import numpy as np
import matplotlib
# Force non-interactive backend to avoid display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import config

# Try to use advanced_config if available (for enhanced pipeline)
try:
    import advanced_config
    # Use advanced_config if it has USE_ALL_BANDS (indicates it's being used)
    if hasattr(advanced_config, 'USE_ALL_BANDS'):
        config = advanced_config
except ImportError:
    pass  # Use basic config


def percentile_stretch(band, low_percentile=2, high_percentile=98):
    """
    Apply percentile-based contrast stretching to avoid black images.
    
    Parameters:
    -----------
    band : ndarray
        2D array representing an image band
    low_percentile : float
        Lower percentile for stretch (default 2)
    high_percentile : float
        Upper percentile for stretch (default 98)
        
    Returns:
    --------
    stretched : ndarray
        Contrast-stretched band normalized to [0, 1]
    """
    # Remove NaN and Inf values for percentile calculation
    valid_data = band[np.isfinite(band)]
    
    if len(valid_data) == 0:
        return np.zeros_like(band)
    
    # Calculate percentiles
    low_val = np.percentile(valid_data, low_percentile)
    high_val = np.percentile(valid_data, high_percentile)
    
    # Avoid division by zero
    if high_val == low_val:
        return np.zeros_like(band)
    
    # Stretch and clip
    stretched = (band - low_val) / (high_val - low_val)
    stretched = np.clip(stretched, 0, 1)
    
    return stretched


def create_rgb_composite(hyperspectral_data, 
                         red_idx=None, 
                         green_idx=None, 
                         blue_idx=None,
                         stretch=True):
    """
    Create RGB composite from hyperspectral data with proper normalization.
    
    Parameters:
    -----------
    hyperspectral_data : ndarray
        Shape (bands, height, width)
    red_idx, green_idx, blue_idx : int
        Band indices for RGB (0-indexed). If None, uses config values.
    stretch : bool
        Whether to apply percentile stretching
        
    Returns:
    --------
    rgb_composite : ndarray
        RGB image with shape (height, width, 3), values in [0, 1]
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("CREATING RGB COMPOSITE")
        print(f"{'='*60}")
    
    # Use config values if not specified
    if red_idx is None:
        red_idx = min(config.RED_BAND_IDX, hyperspectral_data.shape[0] - 1)
    if green_idx is None:
        green_idx = min(config.GREEN_BAND_IDX, hyperspectral_data.shape[0] - 1)
    if blue_idx is None:
        blue_idx = min(config.BLUE_BAND_IDX, hyperspectral_data.shape[0] - 1)
    
    if config.VERBOSE:
        print(f"Using bands: R={red_idx}, G={green_idx}, B={blue_idx}")
    
    # Extract bands
    height = hyperspectral_data.shape[1]
    width = hyperspectral_data.shape[2]
    rgb_composite = np.zeros((height, width, 3))
    
    rgb_composite[:, :, 0] = hyperspectral_data[red_idx, :, :]
    rgb_composite[:, :, 1] = hyperspectral_data[green_idx, :, :]
    rgb_composite[:, :, 2] = hyperspectral_data[blue_idx, :, :]
    
    # Apply contrast stretching to each band
    if stretch:
        if config.VERBOSE:
            print(f"Applying percentile stretch ({config.RGB_PERCENTILE_LOW}%-{config.RGB_PERCENTILE_HIGH}%)")
        
        for i in range(3):
            rgb_composite[:, :, i] = percentile_stretch(
                rgb_composite[:, :, i],
                config.RGB_PERCENTILE_LOW,
                config.RGB_PERCENTILE_HIGH
            )
    else:
        # Simple min-max normalization
        for i in range(3):
            band = rgb_composite[:, :, i]
            band_min = band.min()
            band_max = band.max()
            if band_max > band_min:
                rgb_composite[:, :, i] = (band - band_min) / (band_max - band_min)
    
    if config.VERBOSE:
        print(f"RGB composite created: shape {rgb_composite.shape}")
        print(f"Value range: [{rgb_composite.min():.3f}, {rgb_composite.max():.3f}]")
    
    return rgb_composite


def save_rgb_composite(rgb_composite, filename=None):
    """
    Save RGB composite as PNG file.
    
    Parameters:
    -----------
    rgb_composite : ndarray
        RGB image (height, width, 3)
    filename : str
        Output filename
    """
    if filename is None:
        filename = config.RGB_COMPOSITE_FILE
    
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_composite)
    plt.title('RGB Composite', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    try:
        import os
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Use absolute path to avoid OneDrive sync issues
        abs_filename = os.path.abspath(filename)
        
        plt.savefig(abs_filename, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        # Force file system sync
        import sys
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.CreateFileW(abs_filename, 0x40000000, 0, None, 3, 0x02000000, None)
            if handle != -1:
                kernel32.FlushFileBuffers(handle)
                kernel32.CloseHandle(handle)
        
        # Verify file was created and has content
        if os.path.exists(abs_filename) and os.path.getsize(abs_filename) > 0:
            print(f"RGB composite saved to: {abs_filename} ({os.path.getsize(abs_filename)} bytes)")
        else:
            print(f"ERROR: File {abs_filename} was not created or is empty!")
    except Exception as e:
        print(f"ERROR saving RGB composite to {filename}: {e}")
        plt.close()


def visualize_clusters(rgb_composite, cluster_map, n_clusters, filename=None):
    """
    Visualize K-means clustering results alongside RGB composite.
    
    Parameters:
    -----------
    rgb_composite : ndarray
        RGB image (height, width, 3)
    cluster_map : ndarray
        2D array with cluster labels
    n_clusters : int
        Number of clusters
    filename : str
        Output filename
    """
    if filename is None:
        filename = config.CLUSTERING_RESULTS_FILE
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # RGB composite
    axes[0].imshow(rgb_composite)
    axes[0].set_title('RGB Composite', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Cluster map
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    im = axes[1].imshow(cluster_map, cmap=cmap, vmin=0, vmax=n_clusters)
    axes[1].set_title(f'K-means Clusters (n={n_clusters})', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Cluster ID', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        # Verify file was created
        import os
        if os.path.exists(filename):
            if config.VERBOSE:
                print(f"Clustering visualization saved to: {filename}")
        else:
            print(f"ERROR: File {filename} was not created!")
    except Exception as e:
        print(f"ERROR saving clustering visualization to {filename}: {e}")
        plt.close()


def visualize_classification_results(rgb_composite, ground_truth, prediction, 
                                     class_names=None, filename=None, ground_truth_source=None):
    """
    Visualize classification results: RGB, ground truth, and prediction.
    
    Parameters:
    -----------
    rgb_composite : ndarray
        RGB image (height, width, 3)
    ground_truth : ndarray
        Ground truth labels (height, width)
    prediction : ndarray
        Predicted labels (height, width)
    class_names : list
        Names of classes
    filename : str
        Output filename
    """
    if filename is None:
        filename = config.CLASSIFICATION_RESULTS_FILE
    
    if class_names is None:
        class_names = ['Non-pavement', 'Pavement']
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    # RGB composite
    axes[0].imshow(rgb_composite)
    axes[0].set_title('RGB Composite', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth
    cmap = ListedColormap(['#d62728', '#2ca02c'])  # Red for non-pavement, green for pavement
    im1 = axes[1].imshow(ground_truth, cmap=cmap, vmin=0, vmax=1)
    if ground_truth_source:
        gt_title = f'Ground Truth\n({ground_truth_source})'
    else:
        gt_title = 'Reference Labels\n(from K-means clustering)'
    axes[1].set_title(gt_title, 
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, ticks=[0, 1])
    cbar1.set_ticklabels(class_names)
    
    # Prediction
    im2 = axes[2].imshow(prediction, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title('Classification Result\n(Random Forest)', 
                      fontsize=14, fontweight='bold')
    axes[2].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, ticks=[0, 1])
    cbar2.set_ticklabels(class_names)
    
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        # Verify file was created
        import os
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            print(f"Classification results visualization saved to: {filename} ({os.path.getsize(filename)} bytes)")
        else:
            print(f"ERROR: File {filename} was not created or is empty!")
    except Exception as e:
        print(f"ERROR saving classification results to {filename}: {e}")
        plt.close()


def plot_spectral_signatures(X, y, class_names=None, n_samples_per_class=100):
    """
    Plot average spectral signatures for each class.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix (n_samples, n_bands)
    y : ndarray
        Class labels (n_samples,)
    class_names : list
        Names of classes
    n_samples_per_class : int
        Number of samples to use per class
    """
    if class_names is None:
        class_names = ['Non-pavement', 'Pavement']
    
    unique_classes = np.unique(y)
    n_bands = X.shape[1]
    
    plt.figure(figsize=(12, 6))
    
    for class_id in unique_classes:
        class_mask = y == class_id
        class_samples = X[class_mask]
        
        # Sample if too many
        if len(class_samples) > n_samples_per_class:
            indices = np.random.choice(len(class_samples), n_samples_per_class, replace=False)
            class_samples = class_samples[indices]
        
        # Calculate mean and std
        mean_signature = class_samples.mean(axis=0)
        std_signature = class_samples.std(axis=0)
        
        # Plot
        band_numbers = np.arange(1, n_bands + 1)
        label = class_names[int(class_id)] if int(class_id) < len(class_names) else f'Class {int(class_id)}'
        
        plt.plot(band_numbers, mean_signature, label=label, linewidth=2)
        plt.fill_between(band_numbers, 
                         mean_signature - std_signature,
                         mean_signature + std_signature,
                         alpha=0.2)
    
    plt.xlabel('Band Number', fontsize=12)
    plt.ylabel('Reflectance', fontsize=12)
    plt.title('Average Spectral Signatures by Class', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = config.FIGURES_DIR + '/spectral_signatures.png'
    try:
        plt.savefig(filename, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        # Verify file was created
        import os
        if os.path.exists(filename):
            if config.VERBOSE:
                print(f"Spectral signatures plot saved to: {filename}")
        else:
            print(f"ERROR: File {filename} was not created!")
    except Exception as e:
        print(f"ERROR saving spectral signatures to {filename}: {e}")
        plt.close()


def plot_confusion_matrix(cm, class_names=None, normalize=False):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    cm : ndarray
        Confusion matrix (n_classes, n_classes)
    class_names : list
        Names of classes
    normalize : bool
        Whether to normalize the confusion matrix
    """
    if class_names is None:
        class_names = ['Non-pavement', 'Pavement']
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14)
    
    fig.tight_layout()
    
    filename = config.FIGURES_DIR + '/confusion_matrix.png'
    try:
        plt.savefig(filename, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        # Verify file was created
        import os
        if os.path.exists(filename):
            if config.VERBOSE:
                print(f"Confusion matrix plot saved to: {filename}")
        else:
            print(f"ERROR: File {filename} was not created!")
    except Exception as e:
        print(f"ERROR saving confusion matrix to {filename}: {e}")
        plt.close()

