"""
Ensemble classification methods for hyperspectral data.

Implements:
- Random Forest (enhanced)
- Support Vector Machine
- XGBoost
- Ensemble voting/stacking
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

import advanced_config as config


def create_random_forest():
    """Create enhanced Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
        random_state=config.RANDOM_STATE,
        n_jobs=config.RF_N_JOBS,
        class_weight='balanced'
    )


def create_svm():
    """Create SVM classifier."""
    return SVC(
        kernel=config.SVM_KERNEL,
        C=config.SVM_C,
        gamma=config.SVM_GAMMA,
        random_state=config.RANDOM_STATE,
        probability=True,  # Needed for soft voting
        class_weight='balanced'
    )


def create_xgboost():
    """Create XGBoost classifier."""
    if not HAS_XGBOOST:
        return None
    
    return xgb.XGBClassifier(
        n_estimators=config.XGB_N_ESTIMATORS,
        max_depth=config.XGB_MAX_DEPTH,
        learning_rate=config.XGB_LEARNING_RATE,
        subsample=config.XGB_SUBSAMPLE,
        random_state=config.RANDOM_STATE,
        n_jobs=config.RF_N_JOBS,
        eval_metric='logloss'
    )


def create_ensemble_classifier(method='voting'):
    """
    Create ensemble classifier combining multiple algorithms.
    
    Parameters:
    -----------
    method : str
        'voting' or 'stacking'
        
    Returns:
    --------
    ensemble : ensemble classifier
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print(f"CREATING ENSEMBLE CLASSIFIER ({method.upper()})")
        print(f"{'='*60}")
    
    # Create base classifiers
    estimators = []
    
    if config.USE_RANDOM_FOREST:
        rf = create_random_forest()
        estimators.append(('rf', rf))
        if config.VERBOSE:
            print(f"  ✓ Random Forest ({config.RF_N_ESTIMATORS} trees)")
    
    if config.USE_SVM:
        svm = create_svm()
        estimators.append(('svm', svm))
        if config.VERBOSE:
            print(f"  ✓ SVM (kernel={config.SVM_KERNEL})")
    
    if config.USE_XGBOOST and HAS_XGBOOST:
        xgb_clf = create_xgboost()
        estimators.append(('xgb', xgb_clf))
        if config.VERBOSE:
            print(f"  ✓ XGBoost ({config.XGB_N_ESTIMATORS} trees)")
    
    if len(estimators) == 0:
        raise ValueError("No classifiers enabled in config")
    
    if len(estimators) == 1:
        if config.VERBOSE:
            print(f"\nOnly one classifier enabled, returning that classifier")
        return estimators[0][1]
    
    # Create ensemble
    if method == 'voting':
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=config.ENSEMBLE_VOTING,
            n_jobs=config.N_JOBS
        )
        if config.VERBOSE:
            print(f"\n→ Ensemble method: {config.ENSEMBLE_VOTING} voting")
    
    elif method == 'stacking':
        # Use logistic regression as meta-classifier
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=config.RANDOM_STATE),
            n_jobs=config.N_JOBS
        )
        if config.VERBOSE:
            print(f"\n→ Ensemble method: stacking with logistic regression")
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return ensemble


def train_multiple_classifiers(X_train, y_train):
    """
    Train multiple classifiers separately for comparison.
    
    Parameters:
    -----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
        
    Returns:
    --------
    classifiers : dict
        Dictionary of trained classifiers
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("TRAINING MULTIPLE CLASSIFIERS")
        print(f"{'='*60}")
    
    classifiers = {}
    
    # Random Forest
    if config.USE_RANDOM_FOREST:
        if config.VERBOSE:
            print("\nTraining Random Forest...")
        rf = create_random_forest()
        rf.fit(X_train, y_train)
        classifiers['Random_Forest'] = rf
        if config.VERBOSE:
            print("  ✓ Complete")
    
    # SVM
    if config.USE_SVM:
        if config.VERBOSE:
            print("\nTraining SVM...")
        svm = create_svm()
        # Sample if dataset too large for SVM
        if len(X_train) > 10000:
            if config.VERBOSE:
                print(f"  Sampling 10000 samples for faster SVM training...")
            sample_idx = np.random.choice(len(X_train), 10000, replace=False)
            svm.fit(X_train[sample_idx], y_train[sample_idx])
        else:
            svm.fit(X_train, y_train)
        classifiers['SVM'] = svm
        if config.VERBOSE:
            print("  ✓ Complete")
    
    # XGBoost
    if config.USE_XGBOOST and HAS_XGBOOST:
        if config.VERBOSE:
            print("\nTraining XGBoost...")
        xgb_clf = create_xgboost()
        xgb_clf.fit(X_train, y_train)
        classifiers['XGBoost'] = xgb_clf
        if config.VERBOSE:
            print("  ✓ Complete")
    
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print(f"Trained {len(classifiers)} classifiers")
        print(f"{'='*60}")
    
    return classifiers


def evaluate_multiple_classifiers(classifiers, X_test, y_test):
    """
    Evaluate multiple classifiers and compare performance.
    
    Parameters:
    -----------
    classifiers : dict
        Dictionary of trained classifiers
    X_test : ndarray
        Test features
    y_test : ndarray
        Test labels
        
    Returns:
    --------
    results : dict
        Dictionary with results for each classifier
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("COMPARING CLASSIFIER PERFORMANCE")
        print(f"{'='*60}")
    
    results = {}
    
    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'predictions': y_pred
        }
        
        if config.VERBOSE:
            print(f"\n{name}:")
            print(f"  Accuracy:  {results[name]['accuracy']:.4f}")
            print(f"  Precision: {results[name]['precision']:.4f}")
            print(f"  Recall:    {results[name]['recall']:.4f}")
            print(f"  F1-Score:  {results[name]['f1']:.4f}")
    
    # Find best classifier
    best_clf = max(results.keys(), key=lambda k: results[k]['f1'])
    
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print(f"Best classifier: {best_clf} (F1={results[best_clf]['f1']:.4f})")
        print(f"{'='*60}")
    
    return results


def get_feature_importance(classifier, feature_names=None):
    """
    Extract feature importance from classifier.
    
    Parameters:
    -----------
    classifier : trained classifier
        Must support feature_importances_
    feature_names : list, optional
        Names of features
        
    Returns:
    --------
    importances : ndarray or None
        Feature importances if available
    """
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        
        if config.VERBOSE and feature_names is not None:
            indices = np.argsort(importances)[::-1]
            print(f"\nTop 10 most important features:")
            for i in range(min(10, len(indices))):
                idx = indices[i]
                feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                print(f"  {i+1}. {feat_name}: {importances[idx]:.4f}")
        
        return importances
    
    return None

