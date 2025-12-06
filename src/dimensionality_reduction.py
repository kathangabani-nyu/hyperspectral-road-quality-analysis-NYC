"""
Dimensionality reduction for hyperspectral data.

Implements:
- PCA (Principal Component Analysis)
- Band selection based on importance, correlation, or mutual information
- Variance-based component selection
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import advanced_config as config


def apply_pca(X, n_components=None, variance_threshold=None):
    """
    Apply PCA to hyperspectral data.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix (n_samples, n_bands)
    n_components : int, optional
        Number of components to retain
    variance_threshold : float, optional
        Retain components explaining this much variance (e.g., 0.99 for 99%)
        
    Returns:
    --------
    X_pca : ndarray
        Transformed data
    pca_model : PCA
        Fitted PCA model
    variance_explained : ndarray
        Explained variance ratio for each component
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("APPLYING PCA DIMENSIONALITY REDUCTION")
        print(f"{'='*60}")
        print(f"Original features: {X.shape[1]}")
    
    # Determine number of components
    if variance_threshold is not None:
        # First fit with all components to see variance
        pca_temp = PCA(n_components=min(X.shape[0], X.shape[1]))
        pca_temp.fit(X)
        cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        
        if config.VERBOSE:
            print(f"Variance threshold: {variance_threshold*100}%")
            print(f"Components needed: {n_components}")
    
    elif n_components is None:
        n_components = config.PCA_COMPONENTS
    
    # Apply PCA
    pca_model = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    X_pca = pca_model.fit_transform(X)
    
    variance_explained = pca_model.explained_variance_ratio_
    cumsum_variance = np.cumsum(variance_explained)
    
    if config.VERBOSE:
        print(f"PCA components retained: {n_components}")
        print(f"Total variance explained: {cumsum_variance[-1]*100:.2f}%")
        print(f"Top 5 components explain: {cumsum_variance[min(4, len(cumsum_variance)-1)]*100:.2f}%")
        print(f"Reduced features: {X_pca.shape[1]}")
        print(f"Dimensionality reduction: {X.shape[1]} -> {X_pca.shape[1]} ({X_pca.shape[1]/X.shape[1]*100:.1f}%)")
    
    return X_pca, pca_model, variance_explained


def plot_pca_variance(variance_explained, filename=None):
    """
    Plot explained variance by PCA components.
    
    Parameters:
    -----------
    variance_explained : ndarray
        Explained variance ratio for each component
    filename : str, optional
        Output filename
    """
    if filename is None:
        filename = config.PCA_VARIANCE_PLOT
    
    cumsum_variance = np.cumsum(variance_explained)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance
    ax1.bar(range(1, len(variance_explained) + 1), variance_explained, alpha=0.7)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Variance Explained by Each Component', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance
    ax2.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'b-', linewidth=2)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    ax2.axhline(y=0.99, color='g', linestyle='--', label='99% Variance')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    if config.VERBOSE:
        print(f"PCA variance plot saved to: {filename}")


def select_bands_by_importance(X, y, n_bands=100):
    """
    Select most important bands using Random Forest feature importance.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix (n_samples, n_bands)
    y : ndarray
        Labels
    n_bands : int
        Number of bands to select
        
    Returns:
    --------
    selected_indices : ndarray
        Indices of selected bands
    importance_scores : ndarray
        Importance scores for all bands
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("BAND SELECTION BY IMPORTANCE")
        print(f"{'='*60}")
        print(f"Selecting {n_bands} most important bands from {X.shape[1]}")
    
    # Train a quick Random Forest to get feature importance
    rf_temp = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    # Sample data if too large
    if len(X) > 10000:
        sample_idx = np.random.choice(len(X), 10000, replace=False)
        rf_temp.fit(X[sample_idx], y[sample_idx])
    else:
        rf_temp.fit(X, y)
    
    importance_scores = rf_temp.feature_importances_
    selected_indices = np.argsort(importance_scores)[-n_bands:][::-1]
    
    if config.VERBOSE:
        print(f"Top 5 most important bands: {selected_indices[:5]}")
        print(f"Top 5 importance scores: {importance_scores[selected_indices[:5]]}")
        print(f"Selected bands range: {selected_indices.min()} to {selected_indices.max()}")
    
    return selected_indices, importance_scores


def select_bands_by_correlation(X, y, n_bands=100, method='pearson'):
    """
    Select bands with highest correlation to labels.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix (n_samples, n_bands)
    y : ndarray
        Labels
    n_bands : int
        Number of bands to select
    method : str
        'pearson' or 'spearman'
        
    Returns:
    --------
    selected_indices : ndarray
        Indices of selected bands
    correlation_scores : ndarray
        Correlation scores for all bands
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("BAND SELECTION BY CORRELATION")
        print(f"{'='*60}")
        print(f"Selecting {n_bands} bands with highest correlation")
    
    # Compute correlation of each band with labels
    correlation_scores = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
    
    # Handle NaN correlations
    correlation_scores = np.nan_to_num(correlation_scores, nan=0.0)
    
    selected_indices = np.argsort(correlation_scores)[-n_bands:][::-1]
    
    if config.VERBOSE:
        print(f"Top 5 most correlated bands: {selected_indices[:5]}")
        print(f"Top 5 correlation scores: {correlation_scores[selected_indices[:5]]}")
    
    return selected_indices, correlation_scores


def select_bands_by_mutual_info(X, y, n_bands=100):
    """
    Select bands with highest mutual information with labels.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix (n_samples, n_bands)
    y : ndarray
        Labels
    n_bands : int
        Number of bands to select
        
    Returns:
    --------
    selected_indices : ndarray
        Indices of selected bands
    mi_scores : ndarray
        Mutual information scores for all bands
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("BAND SELECTION BY MUTUAL INFORMATION")
        print(f"{'='*60}")
        print(f"Selecting {n_bands} bands with highest mutual information")
    
    # Sample if too large (MI computation can be slow)
    if len(X) > 10000:
        sample_idx = np.random.choice(len(X), 10000, replace=False)
        X_sample = X[sample_idx]
        y_sample = y[sample_idx]
    else:
        X_sample = X
        y_sample = y
    
    mi_scores = mutual_info_classif(X_sample, y_sample, random_state=config.RANDOM_STATE)
    selected_indices = np.argsort(mi_scores)[-n_bands:][::-1]
    
    if config.VERBOSE:
        print(f"Top 5 bands by MI: {selected_indices[:5]}")
        print(f"Top 5 MI scores: {mi_scores[selected_indices[:5]]}")
    
    return selected_indices, mi_scores


def plot_band_importance(scores, band_indices, method_name, filename=None):
    """
    Plot band importance scores.
    
    Parameters:
    -----------
    scores : ndarray
        Importance/correlation/MI scores
    band_indices : ndarray
        Indices of selected bands
    method_name : str
        Name of selection method
    filename : str, optional
        Output filename
    """
    if filename is None:
        filename = config.FEATURE_IMPORTANCE_PLOT
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # All bands
    ax1.plot(scores, alpha=0.7, linewidth=1)
    ax1.scatter(band_indices, scores[band_indices], c='r', s=20, zorder=5, label='Selected')
    ax1.set_xlabel('Band Index', fontsize=12)
    ax1.set_ylabel(f'{method_name} Score', fontsize=12)
    ax1.set_title(f'Band Importance ({method_name})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Selected bands only
    ax2.bar(range(len(band_indices)), scores[band_indices])
    ax2.set_xlabel('Selected Band Rank', fontsize=12)
    ax2.set_ylabel(f'{method_name} Score', fontsize=12)
    ax2.set_title(f'Top {len(band_indices)} Selected Bands', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    if config.VERBOSE:
        print(f"Band importance plot saved to: {filename}")


def reduce_dimensionality(X, y=None, method='pca', **kwargs):
    """
    Main function to reduce dimensionality.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray, optional
        Labels (needed for supervised band selection)
    method : str
        'pca', 'band_selection', or 'both'
    **kwargs : additional arguments
        
    Returns:
    --------
    X_reduced : ndarray
        Reduced feature matrix
    reducer_info : dict
        Information about the reduction method
    """
    reducer_info = {'method': method}
    
    if method == 'pca':
        X_reduced, pca_model, variance = apply_pca(
            X,
            n_components=kwargs.get('n_components', config.PCA_COMPONENTS),
            variance_threshold=kwargs.get('variance_threshold', config.PCA_VARIANCE_THRESHOLD)
        )
        reducer_info['pca_model'] = pca_model
        reducer_info['variance_explained'] = variance
        plot_pca_variance(variance)
        
    elif method == 'band_selection':
        if y is None:
            raise ValueError("Band selection requires labels (y)")
        
        selection_method = kwargs.get('selection_method', config.BAND_SELECTION_METHOD)
        n_bands = kwargs.get('n_bands', config.N_SELECTED_BANDS)
        
        if selection_method == 'importance':
            selected_idx, scores = select_bands_by_importance(X, y, n_bands)
        elif selection_method == 'correlation':
            selected_idx, scores = select_bands_by_correlation(X, y, n_bands)
        elif selection_method == 'mutual_info':
            selected_idx, scores = select_bands_by_mutual_info(X, y, n_bands)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
        
        X_reduced = X[:, selected_idx]
        reducer_info['selected_bands'] = selected_idx
        reducer_info['band_scores'] = scores
        plot_band_importance(scores, selected_idx, selection_method.replace('_', ' ').title())
        
    elif method == 'both':
        # First select bands, then apply PCA
        if y is None:
            raise ValueError("Band selection requires labels (y)")
        
        selection_method = kwargs.get('selection_method', config.BAND_SELECTION_METHOD)
        n_bands = kwargs.get('n_bands', config.N_SELECTED_BANDS)
        
        if selection_method == 'importance':
            selected_idx, scores = select_bands_by_importance(X, y, n_bands)
        elif selection_method == 'correlation':
            selected_idx, scores = select_bands_by_correlation(X, y, n_bands)
        else:
            selected_idx, scores = select_bands_by_mutual_info(X, y, n_bands)
        
        X_bands = X[:, selected_idx]
        
        X_reduced, pca_model, variance = apply_pca(
            X_bands,
            n_components=kwargs.get('n_components', config.PCA_COMPONENTS)
        )
        
        reducer_info['selected_bands'] = selected_idx
        reducer_info['band_scores'] = scores
        reducer_info['pca_model'] = pca_model
        reducer_info['variance_explained'] = variance
        
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    
    return X_reduced, reducer_info

