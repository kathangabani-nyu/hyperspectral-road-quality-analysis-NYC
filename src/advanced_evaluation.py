"""
Advanced evaluation metrics and visualizations for research-grade analysis.

Implements:
- ROC curves and AUC
- Precision-Recall curves
- Comprehensive metrics
- Classifier comparison visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                            classification_report, confusion_matrix)
import advanced_config as config


def compute_roc_curve(y_true, y_pred_proba, class_names=None):
    """
    Compute ROC curve and AUC.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred_proba : ndarray
        Predicted probabilities for positive class
    class_names : list
        Class names
        
    Returns:
    --------
    fpr : ndarray
        False positive rates
    tpr : ndarray
        True positive rates
    roc_auc : float
        Area under ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    if config.VERBOSE:
        print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    return fpr, tpr, roc_auc


def plot_roc_curves(classifiers_results, y_true, filename=None):
    """
    Plot ROC curves for multiple classifiers.
    
    Parameters:
    -----------
    classifiers_results : dict
        Dictionary with {classifier_name: y_pred_proba}
    y_true : ndarray
        True labels
    filename : str
        Output filename
    """
    if filename is None:
        filename = config.ROC_CURVE_FILE
    
    plt.figure(figsize=(10, 8))
    
    for name, y_pred_proba in classifiers_results.items():
        fpr, tpr, roc_auc = compute_roc_curve(y_true, y_pred_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Classifier Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    if config.VERBOSE:
        print(f"ROC curves saved to: {filename}")


def plot_precision_recall_curves(classifiers_results, y_true, filename=None):
    """
    Plot Precision-Recall curves for multiple classifiers.
    
    Parameters:
    -----------
    classifiers_results : dict
        Dictionary with {classifier_name: y_pred_proba}
    y_true : ndarray
        True labels
    filename : str
        Output filename
    """
    if filename is None:
        filename = config.PR_CURVE_FILE
    
    plt.figure(figsize=(10, 8))
    
    for name, y_pred_proba in classifiers_results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        plt.plot(recall, precision, linewidth=2, label=f'{name} (AP = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Classifier Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    if config.VERBOSE:
        print(f"Precision-Recall curves saved to: {filename}")


def plot_classifier_comparison(results, metric='f1', filename=None):
    """
    Create bar plot comparing classifier performance.
    
    Parameters:
    -----------
    results : dict
        Dictionary with classifier results
    metric : str
        Metric to compare ('accuracy', 'f1', 'precision', 'recall')
    filename : str
        Output filename
    """
    if filename is None:
        filename = config.CLASSIFIER_COMPARISON_FILE
    
    names = list(results.keys())
    scores = [results[name][metric] for name in names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(len(names)), scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f'Classifier Comparison - {metric.capitalize()} Score', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim([0.9, 1.0])  # Zoom in on high scores
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    if config.VERBOSE:
        print(f"Classifier comparison plot saved to: {filename}")


def compute_comprehensive_metrics(y_true, y_pred, y_pred_proba=None, class_names=None):
    """
    Compute comprehensive evaluation metrics.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predictions
    y_pred_proba : ndarray, optional
        Predicted probabilities
    class_names : list
        Class names
        
    Returns:
    --------
    metrics : dict
        Comprehensive metrics
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                matthews_corrcoef, balanced_accuracy_score)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # If probabilities available, compute ROC-AUC
    if y_pred_proba is not None:
        from sklearn.metrics import roc_auc_score
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
        except:
            pass
    
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("COMPREHENSIVE METRICS")
        print(f"{'='*60}")
        print(f"Accuracy:            {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy:   {metrics['balanced_accuracy']:.4f}")
        print(f"Weighted Precision:  {metrics['precision_weighted']:.4f}")
        print(f"Weighted Recall:     {metrics['recall_weighted']:.4f}")
        print(f"Weighted F1:         {metrics['f1_weighted']:.4f}")
        print(f"Matthews Corr Coef:  {metrics['matthews_corrcoef']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC-AUC:             {metrics['roc_auc']:.4f}")
            print(f"Avg Precision:       {metrics['avg_precision']:.4f}")
    
    return metrics


def save_comprehensive_report(metrics, classifiers_results=None, filename=None):
    """
    Save comprehensive evaluation report to file.
    
    Parameters:
    -----------
    metrics : dict
        Metrics dictionary
    classifiers_results : dict, optional
        Results from multiple classifiers
    filename : str
        Output filename
    """
    if filename is None:
        filename = config.COMPREHENSIVE_METRICS_FILE
    
    with open(filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("COMPREHENSIVE EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write("-"*40 + "\n")
        f.write(f"Accuracy:            {metrics['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy:   {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Weighted Precision:  {metrics['precision_weighted']:.4f}\n")
        f.write(f"Weighted Recall:     {metrics['recall_weighted']:.4f}\n")
        f.write(f"Weighted F1-Score:   {metrics['f1_weighted']:.4f}\n")
        f.write(f"Matthews Corr Coef:  {metrics['matthews_corrcoef']:.4f}\n")
        
        if 'roc_auc' in metrics:
            f.write(f"ROC-AUC:             {metrics['roc_auc']:.4f}\n")
            f.write(f"Average Precision:   {metrics['avg_precision']:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Per-Class Metrics:\n")
        f.write("="*60 + "\n")
        
        class_names = ['Non-pavement', 'Pavement']
        for i, name in enumerate(class_names):
            if i < len(metrics['precision_per_class']):
                f.write(f"\n{name}:\n")
                f.write(f"  Precision: {metrics['precision_per_class'][i]:.4f}\n")
                f.write(f"  Recall:    {metrics['recall_per_class'][i]:.4f}\n")
                f.write(f"  F1-Score:  {metrics['f1_per_class'][i]:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Confusion Matrix:\n")
        f.write("="*60 + "\n")
        cm = metrics['confusion_matrix']
        f.write("          Predicted\n")
        f.write("         " + "  ".join([f"{name[:8]:>8s}" for name in class_names]) + "\n")
        f.write("Actual\n")
        for i, name in enumerate(class_names):
            if i < cm.shape[0]:
                f.write(f"{name[:8]:>8s} " + "  ".join([f"{cm[i,j]:>8d}" for j in range(cm.shape[1])]) + "\n")
        
        if classifiers_results:
            f.write("\n" + "="*60 + "\n")
            f.write("Classifier Comparison:\n")
            f.write("="*60 + "\n")
            for clf_name, clf_metrics in classifiers_results.items():
                f.write(f"\n{clf_name}:\n")
                for metric_name, value in clf_metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {metric_name}: {value:.4f}\n")
    
    if config.VERBOSE:
        print(f"\nComprehensive report saved to: {filename}")

