"""
Advanced evaluation metrics for hyperspectral pavement classification.
Implements comprehensive metrics including IoU, spatial metrics, and P4-metric.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, cohen_kappa_score,
                            balanced_accuracy_score, matthews_corrcoef)
from scipy import ndimage


def compute_iou(y_true, y_pred, class_id=1):
    """
    Compute Intersection over Union (IoU) for a specific class.
    
    Parameters:
    -----------
    y_true : ndarray
        True binary labels
    y_pred : ndarray
        Predicted binary labels
    class_id : int
        Class ID to compute IoU for (default: 1 for pavement)
        
    Returns:
    --------
    iou : float
        IoU score
    """
    # Create binary masks for the class
    true_mask = (y_true == class_id)
    pred_mask = (y_pred == class_id)
    
    # Compute intersection and union
    intersection = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou


def compute_per_class_iou(y_true, y_pred, n_classes=2):
    """
    Compute IoU for each class.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    n_classes : int
        Number of classes
        
    Returns:
    --------
    iou_per_class : ndarray
        IoU score for each class
    """
    iou_per_class = np.zeros(n_classes)
    for i in range(n_classes):
        iou_per_class[i] = compute_iou(y_true, y_pred, class_id=i)
    return iou_per_class


def compute_mean_iou(y_true, y_pred, n_classes=2):
    """
    Compute mean IoU across all classes.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    n_classes : int
        Number of classes
        
    Returns:
    --------
    mean_iou : float
        Mean IoU score
    """
    iou_per_class = compute_per_class_iou(y_true, y_pred, n_classes)
    return np.mean(iou_per_class)


def compute_p4_metric(y_true, y_pred, class_id=1):
    """
    Compute P4-metric: geometric mean of precision, recall, specificity, and NPV.
    
    P4 = (Precision * Recall * Specificity * NPV)^(1/4)
    
    Parameters:
    -----------
    y_true : ndarray
        True binary labels
    y_pred : ndarray
        Predicted binary labels
    class_id : int
        Class ID to compute P4 for
        
    Returns:
    --------
    p4 : float
        P4-metric score
    """
    # Create binary masks
    true_mask = (y_true == class_id)
    pred_mask = (y_pred == class_id)
    
    # Compute confusion matrix components
    tp = np.logical_and(true_mask, pred_mask).sum()
    fp = np.logical_and(~true_mask, pred_mask).sum()
    fn = np.logical_and(true_mask, ~pred_mask).sum()
    tn = np.logical_and(~true_mask, ~pred_mask).sum()
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    # P4-metric: geometric mean
    p4 = (precision * recall * specificity * npv) ** 0.25
    
    return p4, {'precision': precision, 'recall': recall, 
                'specificity': specificity, 'npv': npv}


def compute_spatial_accuracy(y_true_map, y_pred_map, window_size=5):
    """
    Compute spatial accuracy: percentage of correctly classified spatial regions.
    
    Parameters:
    -----------
    y_true_map : ndarray
        2D true classification map
    y_pred_map : ndarray
        2D predicted classification map
    window_size : int
        Size of spatial window for region-based evaluation
        
    Returns:
    --------
    spatial_acc : float
        Spatial accuracy score
    """
    from scipy.ndimage import uniform_filter
    
    # Compute local accuracy in windows
    correct = (y_true_map == y_pred_map).astype(float)
    local_acc = uniform_filter(correct, size=window_size, mode='reflect')
    
    # Spatial accuracy: mean of local accuracies
    spatial_acc = local_acc.mean()
    
    return spatial_acc


def compute_boundary_accuracy(y_true_map, y_pred_map):
    """
    Compute boundary accuracy: accuracy at class boundaries.
    
    Parameters:
    -----------
    y_true_map : ndarray
        2D true classification map
    y_pred_map : ndarray
        2D predicted classification map
        
    Returns:
    --------
    boundary_acc : float
        Boundary accuracy score
    """
    from scipy.ndimage import binary_dilation
    
    # Find boundaries in true map
    true_boundaries = np.zeros_like(y_true_map, dtype=bool)
    for class_id in np.unique(y_true_map):
        if class_id < 0:  # Skip invalid pixels
            continue
        class_mask = (y_true_map == class_id)
        dilated = binary_dilation(class_mask, iterations=1)
        boundaries = dilated & ~class_mask
        true_boundaries = true_boundaries | boundaries
    
    # Compute accuracy only at boundaries
    if true_boundaries.sum() == 0:
        return 1.0
    
    boundary_correct = (y_true_map[true_boundaries] == y_pred_map[true_boundaries]).sum()
    boundary_acc = boundary_correct / true_boundaries.sum()
    
    return boundary_acc


def compute_comprehensive_metrics(y_true, y_pred, y_true_map=None, y_pred_map=None, 
                                   class_names=None, verbose=True):
    """
    Compute comprehensive evaluation metrics.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels (1D)
    y_pred : ndarray
        Predicted labels (1D)
    y_true_map : ndarray, optional
        2D true classification map (for spatial metrics)
    y_pred_map : ndarray, optional
        2D predicted classification map (for spatial metrics)
    class_names : list, optional
        Names of classes
    verbose : bool
        Whether to print metrics
        
    Returns:
    --------
    metrics : dict
        Dictionary of all computed metrics
    """
    if class_names is None:
        class_names = ['Non-pavement', 'Pavement']
    
    n_classes = len(class_names)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Weighted and macro averages
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Average Accuracy (AA)
    aa = np.mean(recall_per_class)
    
    # IoU metrics
    iou_per_class = compute_per_class_iou(y_true, y_pred, n_classes)
    mean_iou = compute_mean_iou(y_true, y_pred, n_classes)
    
    # P4-metric for pavement class
    p4_pavement, p4_components = compute_p4_metric(y_true, y_pred, class_id=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Spatial metrics (if maps provided)
    spatial_acc = None
    boundary_acc = None
    if y_true_map is not None and y_pred_map is not None:
        try:
            spatial_acc = compute_spatial_accuracy(y_true_map, y_pred_map)
            boundary_acc = compute_boundary_accuracy(y_true_map, y_pred_map)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not compute spatial metrics: {e}")
    
    # Compile all metrics
    metrics = {
        # Overall metrics
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'average_accuracy': aa,
        'kappa': kappa,
        'matthews_corrcoef': mcc,
        'mean_iou': mean_iou,
        
        # Weighted metrics
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        
        # Macro metrics
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        
        # Per-class metrics
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'iou_per_class': iou_per_class,
        
        # Advanced metrics
        'p4_metric_pavement': p4_pavement,
        'p4_components': p4_components,
        
        # Spatial metrics
        'spatial_accuracy': spatial_acc,
        'boundary_accuracy': boundary_acc,
        
        # Confusion matrix
        'confusion_matrix': cm,
        
        # Class names
        'class_names': class_names
    }
    
    if verbose:
        print_metrics_report(metrics)
    
    return metrics


def compare_ground_truth_sources(y_land_cover, y_kmeans, verbose=True):
    """
    Compare two ground truth sources (e.g., land cover vs k-means).
    
    Parameters:
    -----------
    y_land_cover : ndarray
        Labels from land cover data (1D or 2D)
    y_kmeans : ndarray
        Labels from k-means clustering (1D or 2D)
    verbose : bool
        Print comparison statistics
        
    Returns:
    --------
    comparison_metrics : dict
        Dictionary with agreement statistics
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
    
    # Flatten if 2D
    if y_land_cover.ndim > 1:
        y_lc_flat = y_land_cover.flatten()
    else:
        y_lc_flat = y_land_cover
    
    if y_kmeans.ndim > 1:
        y_km_flat = y_kmeans.flatten()
    else:
        y_km_flat = y_kmeans
    
    # Ensure same length
    min_len = min(len(y_lc_flat), len(y_km_flat))
    y_lc_flat = y_lc_flat[:min_len]
    y_km_flat = y_km_flat[:min_len]
    
    # Calculate agreement metrics
    agreement = (y_lc_flat == y_km_flat).sum()
    total = len(y_lc_flat)
    agreement_rate = agreement / total if total > 0 else 0
    
    accuracy = accuracy_score(y_lc_flat, y_km_flat)
    kappa = cohen_kappa_score(y_lc_flat, y_km_flat)
    cm = confusion_matrix(y_lc_flat, y_km_flat)
    
    # Per-class agreement
    lc_pavement = (y_lc_flat == 1).sum()
    km_pavement = (y_km_flat == 1).sum()
    lc_nonpavement = (y_lc_flat == 0).sum()
    km_nonpavement = (y_km_flat == 0).sum()
    
    # Agreement on pavement pixels
    both_pavement = ((y_lc_flat == 1) & (y_km_flat == 1)).sum()
    both_nonpavement = ((y_lc_flat == 0) & (y_km_flat == 0)).sum()
    
    comparison_metrics = {
        'total_pixels': total,
        'agreement_count': agreement,
        'agreement_rate': agreement_rate,
        'accuracy': accuracy,
        'kappa': kappa,
        'confusion_matrix': cm,
        'land_cover_pavement': lc_pavement,
        'kmeans_pavement': km_pavement,
        'land_cover_nonpavement': lc_nonpavement,
        'kmeans_nonpavement': km_nonpavement,
        'both_pavement': both_pavement,
        'both_nonpavement': both_nonpavement,
        'pavement_agreement': both_pavement / max(lc_pavement, km_pavement, 1),
        'nonpavement_agreement': both_nonpavement / max(lc_nonpavement, km_nonpavement, 1)
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("GROUND TRUTH SOURCE COMPARISON")
        print(f"{'='*60}")
        print(f"Total pixels: {total:,}")
        print(f"Agreement: {agreement:,} ({100*agreement_rate:.2f}%)")
        print(f"Accuracy: {100*accuracy:.2f}%")
        print(f"Kappa: {kappa:.4f}")
        print(f"\nPavement pixels:")
        print(f"  Land Cover: {lc_pavement:,} ({100*lc_pavement/total:.2f}%)")
        print(f"  K-means:    {km_pavement:,} ({100*km_pavement/total:.2f}%)")
        print(f"  Both agree: {both_pavement:,}")
        print(f"\nNon-pavement pixels:")
        print(f"  Land Cover: {lc_nonpavement:,} ({100*lc_nonpavement/total:.2f}%)")
        print(f"  K-means:    {km_nonpavement:,} ({100*km_nonpavement/total:.2f}%)")
        print(f"  Both agree: {both_nonpavement:,}")
        print(f"\nConfusion Matrix:")
        print(f"              Predicted (K-means)")
        print(f"              Non-pave  Pavement")
        print(f"Actual (LC) Non-pave  {cm[0,0]:8d} {cm[0,1]:8d}")
        print(f"Actual (LC) Pavement  {cm[1,0]:8d} {cm[1,1]:8d}")
    
    return comparison_metrics


def print_metrics_report(metrics):
    """
    Print comprehensive metrics report.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics from compute_comprehensive_metrics
    """
    print(f"\n{'='*70}")
    print("COMPREHENSIVE EVALUATION METRICS")
    print(f"{'='*70}")
    
    print(f"\nOverall Metrics:")
    print(f"  Overall Accuracy (OA):        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Balanced Accuracy:             {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy']*100:.2f}%)")
    print(f"  Average Accuracy (AA):         {metrics['average_accuracy']:.4f} ({metrics['average_accuracy']*100:.2f}%)")
    print(f"  Kappa Coefficient (K):         {metrics['kappa']:.4f}")
    print(f"  Matthews Correlation Coef:     {metrics['matthews_corrcoef']:.4f}")
    print(f"  Mean IoU:                      {metrics['mean_iou']:.4f} ({metrics['mean_iou']*100:.2f}%)")
    
    if metrics['spatial_accuracy'] is not None:
        print(f"  Spatial Accuracy:              {metrics['spatial_accuracy']:.4f} ({metrics['spatial_accuracy']*100:.2f}%)")
    if metrics['boundary_accuracy'] is not None:
        print(f"  Boundary Accuracy:             {metrics['boundary_accuracy']:.4f} ({metrics['boundary_accuracy']*100:.2f}%)")
    
    print(f"\nWeighted Metrics:")
    print(f"  Precision (Weighted):          {metrics['precision_weighted']:.4f}")
    print(f"  Recall (Weighted):              {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score (Weighted):           {metrics['f1_weighted']:.4f}")
    
    print(f"\nMacro Metrics:")
    print(f"  Precision (Macro):              {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro):                 {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (Macro):               {metrics['f1_macro']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    class_names = metrics['class_names']
    for i, name in enumerate(class_names):
        print(f"\n  {name}:")
        print(f"    Precision:  {metrics['precision_per_class'][i]:.4f} ({metrics['precision_per_class'][i]*100:.2f}%)")
        print(f"    Recall:     {metrics['recall_per_class'][i]:.4f} ({metrics['recall_per_class'][i]*100:.2f}%)")
        print(f"    F1-Score:   {metrics['f1_per_class'][i]:.4f} ({metrics['f1_per_class'][i]*100:.2f}%)")
        print(f"    IoU:        {metrics['iou_per_class'][i]:.4f} ({metrics['iou_per_class'][i]*100:.2f}%)")
    
    print(f"\nAdvanced Metrics:")
    print(f"  P4-Metric (Pavement):           {metrics['p4_metric_pavement']:.4f}")
    print(f"    Components:")
    p4_comp = metrics['p4_components']
    print(f"      Precision:    {p4_comp['precision']:.4f}")
    print(f"      Recall:       {p4_comp['recall']:.4f}")
    print(f"      Specificity:  {p4_comp['specificity']:.4f}")
    print(f"      NPV:          {p4_comp['npv']:.4f}")
    
    print(f"\n{'='*70}")


def save_metrics_report(metrics, filename):
    """
    Save comprehensive metrics to file.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics
    filename : str
        Output filename
    """
    import os
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    try:
        with open(filename, 'w') as f:
            f.write("="*70 + "\n")
            f.write("COMPREHENSIVE EVALUATION METRICS\n")
            f.write("="*70 + "\n\n")
            
            f.write("Overall Metrics:\n")
            f.write(f"  Overall Accuracy (OA):        {metrics['accuracy']:.6f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"  Balanced Accuracy:             {metrics['balanced_accuracy']:.6f} ({metrics['balanced_accuracy']*100:.2f}%)\n")
            f.write(f"  Average Accuracy (AA):         {metrics['average_accuracy']:.6f} ({metrics['average_accuracy']*100:.2f}%)\n")
            f.write(f"  Kappa Coefficient (K):         {metrics['kappa']:.6f}\n")
            f.write(f"  Matthews Correlation Coef:     {metrics['matthews_corrcoef']:.6f}\n")
            f.write(f"  Mean IoU:                      {metrics['mean_iou']:.6f} ({metrics['mean_iou']*100:.2f}%)\n")
            
            if metrics['spatial_accuracy'] is not None:
                f.write(f"  Spatial Accuracy:              {metrics['spatial_accuracy']:.6f} ({metrics['spatial_accuracy']*100:.2f}%)\n")
            if metrics['boundary_accuracy'] is not None:
                f.write(f"  Boundary Accuracy:             {metrics['boundary_accuracy']:.6f} ({metrics['boundary_accuracy']*100:.2f}%)\n")
            
            f.write("\nWeighted Metrics:\n")
            f.write(f"  Precision (Weighted):          {metrics['precision_weighted']:.6f}\n")
            f.write(f"  Recall (Weighted):              {metrics['recall_weighted']:.6f}\n")
            f.write(f"  F1-Score (Weighted):           {metrics['f1_weighted']:.6f}\n")
            
            f.write("\nMacro Metrics:\n")
            f.write(f"  Precision (Macro):              {metrics['precision_macro']:.6f}\n")
            f.write(f"  Recall (Macro):                 {metrics['recall_macro']:.6f}\n")
            f.write(f"  F1-Score (Macro):               {metrics['f1_macro']:.6f}\n")
            
            f.write("\nPer-Class Metrics:\n")
            class_names = metrics['class_names']
            for i, name in enumerate(class_names):
                f.write(f"\n  {name}:\n")
                f.write(f"    Precision:  {metrics['precision_per_class'][i]:.6f} ({metrics['precision_per_class'][i]*100:.2f}%)\n")
                f.write(f"    Recall:     {metrics['recall_per_class'][i]:.6f} ({metrics['recall_per_class'][i]*100:.2f}%)\n")
                f.write(f"    F1-Score:   {metrics['f1_per_class'][i]:.6f} ({metrics['f1_per_class'][i]*100:.2f}%)\n")
                f.write(f"    IoU:        {metrics['iou_per_class'][i]:.6f} ({metrics['iou_per_class'][i]*100:.2f}%)\n")
            
            f.write("\nAdvanced Metrics:\n")
            f.write(f"  P4-Metric (Pavement):           {metrics['p4_metric_pavement']:.6f}\n")
            p4_comp = metrics['p4_components']
            f.write(f"    Precision:    {p4_comp['precision']:.6f}\n")
            f.write(f"    Recall:       {p4_comp['recall']:.6f}\n")
            f.write(f"    Specificity:  {p4_comp['specificity']:.6f}\n")
            f.write(f"    NPV:          {p4_comp['npv']:.6f}\n")
            
            f.write("\nConfusion Matrix:\n")
            cm = metrics['confusion_matrix']
            f.write("          Predicted\n")
            f.write("         ")
            for i in range(cm.shape[1]):
                f.write(f"  {class_names[i][:8]}")
            f.write("\n")
            for i in range(cm.shape[0]):
                f.write(f"Actual {class_names[i][:8]}")
                for j in range(cm.shape[1]):
                    f.write(f"  {cm[i,j]:6d}")
                f.write("\n")
            
            f.write("\n" + "="*70 + "\n")
        
        # Verify file was created
        if os.path.exists(filename):
            try:
                import advanced_config as config
                if config.VERBOSE:
                    print(f"Metrics report saved to: {filename}")
            except:
                print(f"Metrics report saved to: {filename}")
        else:
            print(f"ERROR: File {filename} was not created!")
    except Exception as e:
        print(f"ERROR saving metrics report to {filename}: {e}")

