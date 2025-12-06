"""
Classification module for hyperspectral pavement identification.
Includes improved training strategies and evaluation.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix)
from sklearn.utils import resample
import config


def balance_dataset(X, y, strategy='undersample', max_samples_per_class=None):
    """
    Balance the dataset to handle class imbalance.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix (n_samples, n_features)
    y : ndarray
        Labels (n_samples,)
    strategy : str
        'undersample' - reduce majority class
        'oversample' - increase minority class
    max_samples_per_class : int
        Maximum samples per class
        
    Returns:
    --------
    X_balanced : ndarray
        Balanced feature matrix
    y_balanced : ndarray
        Balanced labels
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("BALANCING DATASET")
        print(f"{'='*60}")
    
    if max_samples_per_class is None:
        max_samples_per_class = config.MAX_SAMPLES_PER_CLASS
    
    unique_classes = np.unique(y)
    
    # Count samples per class
    class_counts = {}
    for c in unique_classes:
        class_counts[c] = np.sum(y == c)
    
    if config.VERBOSE:
        print("Original class distribution:")
        for c, count in class_counts.items():
            pct = 100 * count / len(y)
            print(f"  Class {int(c)}: {count} samples ({pct:.1f}%)")
    
    # Determine target size
    if strategy == 'undersample':
        target_size = min(min(class_counts.values()), max_samples_per_class)
    else:  # oversample
        target_size = min(max(class_counts.values()), max_samples_per_class)
    
    if config.VERBOSE:
        print(f"\nBalancing strategy: {strategy}")
        print(f"Target samples per class: {target_size}")
    
    # Resample each class
    X_balanced_list = []
    y_balanced_list = []
    
    for c in unique_classes:
        class_mask = y == c
        X_class = X[class_mask]
        y_class = y[class_mask]
        
        if len(X_class) > target_size:
            # Undersample
            X_resampled, y_resampled = resample(
                X_class, y_class,
                n_samples=target_size,
                random_state=config.RANDOM_STATE,
                replace=False
            )
        elif len(X_class) < target_size and strategy == 'oversample':
            # Oversample
            X_resampled, y_resampled = resample(
                X_class, y_class,
                n_samples=target_size,
                random_state=config.RANDOM_STATE,
                replace=True
            )
        else:
            X_resampled = X_class
            y_resampled = y_class
        
        X_balanced_list.append(X_resampled)
        y_balanced_list.append(y_resampled)
    
    X_balanced = np.vstack(X_balanced_list)
    y_balanced = np.hstack(y_balanced_list)
    
    # Shuffle
    shuffle_idx = np.random.RandomState(config.RANDOM_STATE).permutation(len(X_balanced))
    X_balanced = X_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]
    
    if config.VERBOSE:
        print("\nBalanced class distribution:")
        for c in unique_classes:
            count = np.sum(y_balanced == c)
            pct = 100 * count / len(y_balanced)
            print(f"  Class {int(c)}: {count} samples ({pct:.1f}%)")
    
    return X_balanced, y_balanced


def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier.
    
    Parameters:
    -----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
        
    Returns:
    --------
    classifier : RandomForestClassifier
        Trained classifier
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print(f"{'='*60}")
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"\nRandom Forest parameters:")
        print(f"  n_estimators: {config.RF_N_ESTIMATORS}")
        print(f"  max_depth: {config.RF_MAX_DEPTH}")
        print(f"  min_samples_split: {config.RF_MIN_SAMPLES_SPLIT}")
    
    # Train classifier
    classifier = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        random_state=config.RANDOM_STATE,
        n_jobs=config.RF_N_JOBS,
        verbose=0
    )
    
    if config.VERBOSE:
        print("\nTraining in progress...")
    
    classifier.fit(X_train, y_train)
    
    if config.VERBOSE:
        print("Training complete!")
    
    return classifier


def evaluate_classifier(classifier, X_test, y_test, class_names=None):
    """
    Evaluate classifier performance.
    
    Parameters:
    -----------
    classifier : trained classifier
        Classifier to evaluate
    X_test : ndarray
        Test features
    y_test : ndarray
        Test labels
    class_names : list
        Names of classes
        
    Returns:
    --------
    metrics : dict
        Dictionary with evaluation metrics
    y_pred : ndarray
        Predictions on test set
    """
    if class_names is None:
        class_names = ['Non-pavement', 'Pavement']
    
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("EVALUATING CLASSIFIER")
        print(f"{'='*60}")
        print(f"Test samples: {len(X_test)}")
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("CLASSIFICATION METRICS")
        print(f"{'='*60}")
        print(f"Overall Accuracy:  {accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall:    {recall:.4f}")
        print(f"Weighted F1-Score:  {f1:.4f}")
        
        print(f"\n{'='*60}")
        print("PER-CLASS METRICS")
        print(f"{'='*60}")
        for i, name in enumerate(class_names):
            if i < len(precision_per_class):
                print(f"\n{name}:")
                print(f"  Precision: {precision_per_class[i]:.4f}")
                print(f"  Recall:    {recall_per_class[i]:.4f}")
                print(f"  F1-Score:  {f1_per_class[i]:.4f}")
        
        print(f"\n{'='*60}")
        print("CONFUSION MATRIX")
        print(f"{'='*60}")
        print("          Predicted")
        print("         ", "  ".join([f"{name[:8]:>8s}" for name in class_names]))
        print("Actual")
        for i, name in enumerate(class_names):
            if i < cm.shape[0]:
                print(f"{name[:8]:>8s}", "  ".join([f"{cm[i,j]:>8d}" for j in range(cm.shape[1])]))
        
        print(f"\n{'='*60}")
        print("DETAILED CLASSIFICATION REPORT")
        print(f"{'='*60}")
        
        # Check if we have multiple classes
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        if len(unique_classes) > 1:
            print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
        else:
            print(f"WARNING: Only one class present in predictions: {class_names[int(unique_classes[0])]}")
            print("Cannot generate meaningful classification report with single class.")
            print("This typically means:")
            print("  - Data region has no variation (all same surface type)")
            print("  - Try a different START_X/START_Y location in config.py")
            print("  - Or increase test area size to capture more variety")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm
    }
    
    return metrics, y_pred


def apply_classifier(classifier, X, batch_size=10000):
    """
    Apply classifier to full dataset in batches.
    
    Parameters:
    -----------
    classifier : trained classifier
        Classifier to apply
    X : ndarray
        Full feature matrix (may include invalid pixels)
    batch_size : int
        Batch size for prediction
        
    Returns:
    --------
    predictions : ndarray
        Predictions for all pixels (-999 for invalid pixels)
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("APPLYING CLASSIFIER TO FULL IMAGE")
        print(f"{'='*60}")
        print(f"Total pixels: {len(X)}")
        print(f"Batch size: {batch_size}")
    
    predictions = []
    n_batches = int(np.ceil(len(X) / batch_size))
    
    for i in range(0, len(X), batch_size):
        batch = X[i:min(i+batch_size, len(X))]
        
        # Check for invalid pixels
        batch_valid = ~(np.isnan(batch).any(axis=1) | np.isinf(batch).any(axis=1))
        
        batch_pred = np.full(len(batch), -999, dtype=int)
        
        if np.any(batch_valid):
            batch_pred[batch_valid] = classifier.predict(batch[batch_valid])
        
        predictions.append(batch_pred)
        
        if config.VERBOSE and (i // batch_size + 1) % 5 == 0:
            print(f"  Processed batch {i // batch_size + 1}/{n_batches}")
    
    all_predictions = np.concatenate(predictions)
    
    if config.VERBOSE:
        n_valid = np.sum(all_predictions != -999)
        n_pavement = np.sum(all_predictions == 1)
        n_non_pavement = np.sum(all_predictions == 0)
        
        print(f"\nPrediction summary:")
        print(f"  Valid pixels: {n_valid}")
        print(f"  Pavement: {n_pavement} ({100*n_pavement/n_valid:.1f}%)")
        print(f"  Non-pavement: {n_non_pavement} ({100*n_non_pavement/n_valid:.1f}%)")
    
    return all_predictions


def save_metrics_to_file(metrics, filename=None):
    """
    Save classification metrics to text file.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with metrics
    filename : str
        Output filename
    """
    if filename is None:
        filename = config.METRICS_FILE
    
    with open(filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("PAVEMENT CLASSIFICATION METRICS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Overall Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Weighted Precision: {metrics['precision']:.4f}\n")
        f.write(f"Weighted Recall:    {metrics['recall']:.4f}\n")
        f.write(f"Weighted F1-Score:  {metrics['f1']:.4f}\n\n")
        
        f.write("="*60 + "\n")
        f.write("PER-CLASS METRICS\n")
        f.write("="*60 + "\n\n")
        
        class_names = ['Non-pavement', 'Pavement']
        for i, name in enumerate(class_names):
            if i < len(metrics['precision_per_class']):
                f.write(f"{name}:\n")
                f.write(f"  Precision: {metrics['precision_per_class'][i]:.4f}\n")
                f.write(f"  Recall:    {metrics['recall_per_class'][i]:.4f}\n")
                f.write(f"  F1-Score:  {metrics['f1_per_class'][i]:.4f}\n\n")
        
        f.write("="*60 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("="*60 + "\n\n")
        
        cm = metrics['confusion_matrix']
        f.write("          Predicted\n")
        f.write("         " + "  ".join([f"{name[:8]:>8s}" for name in class_names]) + "\n")
        f.write("Actual\n")
        for i, name in enumerate(class_names):
            if i < cm.shape[0]:
                f.write(f"{name[:8]:>8s} " + "  ".join([f"{cm[i,j]:>8d}" for j in range(cm.shape[1])]) + "\n")
    
    if config.VERBOSE:
        print(f"\nMetrics saved to: {filename}")

