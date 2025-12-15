"""
Spectral filtering analysis and visualization tools.

Provides functions to analyze, visualize, and validate spectral masking results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import advanced_config as config


def analyze_filtered_pixels(X, mask, hyperspectral_data=None, rgb_image=None):
    """
    Analyze what types of pixels are being filtered.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix (n_samples, n_bands)
    mask : ndarray
        Boolean mask (True = kept, False = filtered)
    hyperspectral_data : ndarray, optional
        3D hyperspectral data
    rgb_image : ndarray, optional
        RGB composite for visualization
        
    Returns:
    --------
    analysis : dict
        Dictionary with analysis results
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("ANALYZING FILTERED PIXELS")
        print(f"{'='*60}")
    
    n_bands = X.shape[1]
    red_idx = int(n_bands * 0.35)
    nir_idx = int(n_bands * 0.45) if n_bands > 50 else min(n_bands-1, 70)
    
    # Compute spectral indices for all pixels
    if nir_idx < n_bands and red_idx < n_bands:
        red = X[:, red_idx]
        nir = X[:, nir_idx]
        denominator = nir + red + 1e-10
        ndvi = (nir - red) / denominator
    else:
        ndvi = np.zeros(X.shape[0])
    
    brightness = X.mean(axis=1)
    spectral_std = X.std(axis=1)
    
    # Separate kept and filtered pixels
    kept_pixels = mask
    filtered_pixels = ~mask
    
    analysis = {
        'total_pixels': len(mask),
        'kept_count': kept_pixels.sum(),
        'filtered_count': filtered_pixels.sum(),
        'kept_percentage': 100 * kept_pixels.sum() / len(mask),
        'filtered_percentage': 100 * filtered_pixels.sum() / len(mask),
        'ndvi': {
            'kept_mean': float(ndvi[kept_pixels].mean()) if kept_pixels.sum() > 0 else 0.0,
            'kept_std': float(ndvi[kept_pixels].std()) if kept_pixels.sum() > 0 else 0.0,
            'filtered_mean': float(ndvi[filtered_pixels].mean()) if filtered_pixels.sum() > 0 else 0.0,
            'filtered_std': float(ndvi[filtered_pixels].std()) if filtered_pixels.sum() > 0 else 0.0,
        },
        'brightness': {
            'kept_mean': float(brightness[kept_pixels].mean()) if kept_pixels.sum() > 0 else 0.0,
            'kept_std': float(brightness[kept_pixels].std()) if kept_pixels.sum() > 0 else 0.0,
            'filtered_mean': float(brightness[filtered_pixels].mean()) if filtered_pixels.sum() > 0 else 0.0,
            'filtered_std': float(brightness[filtered_pixels].std()) if filtered_pixels.sum() > 0 else 0.0,
        },
        'variance': {
            'kept_mean': float(spectral_std[kept_pixels].mean()) if kept_pixels.sum() > 0 else 0.0,
            'kept_std': float(spectral_std[kept_pixels].std()) if kept_pixels.sum() > 0 else 0.0,
            'filtered_mean': float(spectral_std[filtered_pixels].mean()) if filtered_pixels.sum() > 0 else 0.0,
            'filtered_std': float(spectral_std[filtered_pixels].std()) if filtered_pixels.sum() > 0 else 0.0,
        }
    }
    
    if config.VERBOSE:
        print(f"\nPixel Statistics:")
        print(f"  Total pixels: {analysis['total_pixels']:,}")
        print(f"  Kept: {analysis['kept_count']:,} ({analysis['kept_percentage']:.1f}%)")
        print(f"  Filtered: {analysis['filtered_count']:,} ({analysis['filtered_percentage']:.1f}%)")
        
        print(f"\nNDVI Statistics:")
        print(f"  Kept pixels - Mean: {analysis['ndvi']['kept_mean']:.3f}, Std: {analysis['ndvi']['kept_std']:.3f}")
        print(f"  Filtered pixels - Mean: {analysis['ndvi']['filtered_mean']:.3f}, Std: {analysis['ndvi']['filtered_std']:.3f}")
        
        print(f"\nBrightness Statistics:")
        print(f"  Kept pixels - Mean: {analysis['brightness']['kept_mean']:.3f}, Std: {analysis['brightness']['kept_std']:.3f}")
        print(f"  Filtered pixels - Mean: {analysis['brightness']['filtered_mean']:.3f}, Std: {analysis['brightness']['filtered_std']:.3f}")
        
        print(f"\nSpectral Variance Statistics:")
        print(f"  Kept pixels - Mean: {analysis['variance']['kept_mean']:.3f}, Std: {analysis['variance']['kept_std']:.3f}")
        print(f"  Filtered pixels - Mean: {analysis['variance']['filtered_mean']:.3f}, Std: {analysis['variance']['filtered_std']:.3f}")
    
    return analysis


def visualize_spectral_filtering(rgb_image, original_mask, filtered_mask, output_path=None, 
                                  height=None, width=None):
    """
    Create visualization showing filtered pixels.
    
    Parameters:
    -----------
    rgb_image : ndarray
        RGB composite image (H, W, 3)
    original_mask : ndarray
        Original valid mask (1D or 2D)
    filtered_mask : ndarray
        Filtered mask after spectral masking (1D or 2D)
    output_path : str, optional
        Path to save visualization
    height : int, optional
        Image height (if masks are 1D)
    width : int, optional
        Image width (if masks are 1D)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if rgb_image is None:
        if config.VERBOSE:
            print("Warning: No RGB image provided, skipping visualization")
        return None
    
    # Reshape masks if needed
    if original_mask.ndim == 1:
        if height is None or width is None:
            if config.VERBOSE:
                print("Warning: Cannot reshape 1D mask without height/width")
            return None
        original_mask_2d = original_mask.reshape(height, width)
        filtered_mask_2d = filtered_mask.reshape(height, width)
    else:
        original_mask_2d = original_mask
        filtered_mask_2d = filtered_mask
    
    # Ensure RGB image is in correct format
    if rgb_image.ndim == 3 and rgb_image.shape[2] == 3:
        rgb_display = rgb_image.copy()
        # Normalize to 0-1 range if needed
        if rgb_display.max() > 1.0:
            rgb_display = rgb_display / rgb_display.max()
    else:
        if config.VERBOSE:
            print("Warning: RGB image format not recognized")
        return None
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original RGB
    axes[0].imshow(rgb_display)
    axes[0].set_title('Original RGB Composite', fontsize=12)
    axes[0].axis('off')
    
    # Show filtered pixels overlay
    filtered_only = original_mask_2d & (~filtered_mask_2d)
    
    # Create overlay: red for filtered pixels
    overlay = np.zeros_like(rgb_display)
    overlay[filtered_only] = [1.0, 0.0, 0.0]  # Red for filtered
    
    axes[1].imshow(rgb_display)
    axes[1].imshow(overlay, alpha=0.3)
    axes[1].set_title('Filtered Pixels (Red Overlay)', fontsize=12)
    axes[1].axis('off')
    
    # Show kept pixels overlay
    kept_only = filtered_mask_2d
    
    # Create overlay: green for kept pixels
    overlay2 = np.zeros_like(rgb_display)
    overlay2[kept_only] = [0.0, 1.0, 0.0]  # Green for kept
    
    axes[2].imshow(rgb_display)
    axes[2].imshow(overlay2, alpha=0.3)
    axes[2].set_title('Kept Pixels (Green Overlay)', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=getattr(config, 'FIGURE_DPI', 300), bbox_inches='tight')
        if config.VERBOSE:
            print(f"Spectral filtering visualization saved to: {output_path}")
    
    return fig


def validate_filtering_quality(mask, ground_truth=None, min_pixels_ratio=0.01):
    """
    Check if filtering is too aggressive and suggest adjustments.
    
    Parameters:
    -----------
    mask : ndarray
        Boolean mask after filtering
    ground_truth : ndarray, optional
        Ground truth labels (1D or 2D)
    min_pixels_ratio : float
        Minimum ratio of pixels that should remain
        
    Returns:
    --------
    validation : dict
        Dictionary with validation results and recommendations
    """
    total_pixels = len(mask) if mask.ndim == 1 else mask.size
    kept_pixels = mask.sum()
    kept_ratio = kept_pixels / total_pixels
    
    validation = {
        'total_pixels': total_pixels,
        'kept_pixels': kept_pixels,
        'kept_ratio': kept_ratio,
        'kept_percentage': 100 * kept_ratio,
        'is_too_aggressive': kept_ratio < min_pixels_ratio,
        'recommendations': []
    }
    
    # Check if too aggressive
    if validation['is_too_aggressive']:
        validation['recommendations'].append(
            f"Filtering is too aggressive: only {validation['kept_percentage']:.1f}% of pixels remain"
        )
        validation['recommendations'].append(
            f"Consider using 'conservative' filtering strategy"
        )
        validation['recommendations'].append(
            f"Or increase NDVI threshold to 0.35-0.4"
        )
    elif kept_ratio < 0.05:
        validation['recommendations'].append(
            f"Filtering is quite aggressive: {validation['kept_percentage']:.1f}% of pixels remain"
        )
        validation['recommendations'].append(
            f"May be appropriate for dense urban areas, but consider 'moderate' strategy"
        )
    elif kept_ratio < 0.10:
        validation['recommendations'].append(
            f"Filtering is moderate: {validation['kept_percentage']:.1f}% of pixels remain"
        )
    else:
        validation['recommendations'].append(
            f"Filtering is conservative: {validation['kept_percentage']:.1f}% of pixels remain"
        )
    
    # Compare with ground truth if available
    if ground_truth is not None:
        gt_flat = ground_truth.flatten() if ground_truth.ndim == 2 else ground_truth
        mask_flat = mask.flatten() if mask.ndim == 2 else mask
        
        if len(gt_flat) == len(mask_flat):
            # Check how many ground truth pavement pixels are kept
            pavement_pixels = (gt_flat > 0) if gt_flat.dtype == bool else (gt_flat == 1)
            kept_pavement = pavement_pixels & mask_flat
            
            if pavement_pixels.sum() > 0:
                pavement_recall = kept_pavement.sum() / pavement_pixels.sum()
                validation['pavement_recall'] = pavement_recall
                validation['pavement_pixels_total'] = pavement_pixels.sum()
                validation['pavement_pixels_kept'] = kept_pavement.sum()
                
                if pavement_recall < 0.8:
                    validation['recommendations'].append(
                        f"Only {100*pavement_recall:.1f}% of ground truth pavement pixels are kept"
                    )
                    validation['recommendations'].append(
                        "Consider less aggressive filtering to preserve more pavement"
                    )
    
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("FILTERING VALIDATION")
        print(f"{'='*60}")
        print(f"Total pixels: {validation['total_pixels']:,}")
        print(f"Kept pixels: {validation['kept_pixels']:,} ({validation['kept_percentage']:.1f}%)")
        if validation['is_too_aggressive']:
            print(f"WARNING: Filtering is too aggressive!")
        if validation['recommendations']:
            print(f"\nRecommendations:")
            for rec in validation['recommendations']:
                print(f"  - {rec}")
    
    return validation

