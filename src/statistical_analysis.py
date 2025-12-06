"""
Statistical analysis and significance testing for classifier comparison.

Implements:
- McNemar's test for comparing classifiers
- Cohen's Kappa for agreement
- Confidence intervals
- Cross-validation with statistical testing
"""

import numpy as np
from scipy.stats import chi2, norm
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import advanced_config as config


def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    McNemar's test to determine if two classifiers are significantly different.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred1 : ndarray
        Predictions from classifier 1
    y_pred2 : ndarray
        Predictions from classifier 2
        
    Returns:
    --------
    statistic : float
        McNemar test statistic
    p_value : float
        P-value
    significant : bool
        Whether difference is significant at config.CONFIDENCE_LEVEL
    """
    # Create contingency table
    # n01: clf1 correct, clf2 wrong
    # n10: clf1 wrong, clf2 correct
    n01 = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    n10 = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    
    # McNemar's test statistic (with continuity correction)
    statistic = (abs(n01 - n10) - 1)**2 / (n01 + n10 + 1e-10)
    
    # P-value from chi-square distribution with 1 degree of freedom
    p_value = 1 - chi2.cdf(statistic, df=1)
    
    # Significant if p-value < (1 - confidence_level)
    alpha = 1 - config.CONFIDENCE_LEVEL
    significant = p_value < alpha
    
    if config.VERBOSE:
        print(f"\nMcNemar's Test:")
        print(f"  n01 (clf1 correct, clf2 wrong): {n01}")
        print(f"  n10 (clf1 wrong, clf2 correct): {n10}")
        print(f"  Test statistic: {statistic:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant at {config.CONFIDENCE_LEVEL*100}% level: {significant}")
    
    return statistic, p_value, significant


def compute_kappa(y_true, y_pred):
    """
    Compute Cohen's Kappa score.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
        
    Returns:
    --------
    kappa : float
        Kappa score (-1 to 1)
    """
    kappa = cohen_kappa_score(y_true, y_pred)
    
    if config.VERBOSE:
        print(f"\nCohen's Kappa: {kappa:.4f}")
        if kappa > 0.8:
            print("  Interpretation: Almost perfect agreement")
        elif kappa > 0.6:
            print("  Interpretation: Substantial agreement")
        elif kappa > 0.4:
            print("  Interpretation: Moderate agreement")
        elif kappa > 0.2:
            print("  Interpretation: Fair agreement")
        else:
            print("  Interpretation: Poor agreement")
    
    return kappa


def confidence_interval(accuracy, n_samples, confidence=0.95):
    """
    Compute confidence interval for accuracy using normal approximation.
    
    Parameters:
    -----------
    accuracy : float
        Observed accuracy
    n_samples : int
        Number of samples
    confidence : float
        Confidence level (e.g., 0.95 for 95%)
        
    Returns:
    --------
    lower : float
        Lower bound
    upper : float
        Upper bound
    """
    # Z-score for confidence level
    z = norm.ppf((1 + confidence) / 2)
    
    # Standard error
    se = np.sqrt(accuracy * (1 - accuracy) / n_samples)
    
    # Confidence interval
    lower = accuracy - z * se
    upper = accuracy + z * se
    
    # Clip to [0, 1]
    lower = max(0, lower)
    upper = min(1, upper)
    
    if config.VERBOSE:
        print(f"\n{confidence*100}% Confidence Interval for Accuracy:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  CI: [{lower:.4f}, {upper:.4f}]")
        print(f"  Margin of error: ±{z*se:.4f}")
    
    return lower, upper


def cross_validation_with_stats(classifier, X, y, cv_folds=None):
    """
    Perform cross-validation and compute statistics.
    
    Parameters:
    -----------
    classifier : sklearn classifier
        Classifier to evaluate
    X : ndarray
        Features
    y : ndarray
        Labels
    cv_folds : int, optional
        Number of CV folds
        
    Returns:
    --------
    scores : ndarray
        Cross-validation scores
    mean_score : float
        Mean score
    std_score : float
        Standard deviation
    ci_lower : float
        Lower confidence bound
    ci_upper : float
        Upper confidence bound
    """
    if cv_folds is None:
        cv_folds = config.CV_FOLDS
    
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION ({cv_folds}-FOLD)")
        print(f"{'='*60}")
    
    # Perform cross-validation
    if config.CV_STRATIFIED:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    
    scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy', n_jobs=config.N_JOBS)
    
    mean_score = scores.mean()
    std_score = scores.std()
    
    # Confidence interval using t-distribution
    from scipy.stats import t
    alpha = 1 - config.CONFIDENCE_LEVEL
    t_val = t.ppf(1 - alpha/2, cv_folds - 1)
    margin = t_val * std_score / np.sqrt(cv_folds)
    
    ci_lower = mean_score - margin
    ci_upper = mean_score + margin
    
    if config.VERBOSE:
        print(f"\nCross-validation results:")
        print(f"  Scores: {[f'{s:.4f}' for s in scores]}")
        print(f"  Mean accuracy: {mean_score:.4f}")
        print(f"  Std deviation: {std_score:.4f}")
        print(f"  {config.CONFIDENCE_LEVEL*100}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return scores, mean_score, std_score, ci_lower, ci_upper


def compare_classifiers_statistical(classifiers_results, y_true):
    """
    Statistical comparison of multiple classifiers.
    
    Parameters:
    -----------
    classifiers_results : dict
        Dictionary with classifier names as keys and predictions as values
    y_true : ndarray
        True labels
        
    Returns:
    --------
    comparison_table : dict
        Pairwise comparison results
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("PAIRWISE CLASSIFIER COMPARISON (McNEMAR'S TEST)")
        print(f"{'='*60}")
    
    classifier_names = list(classifiers_results.keys())
    comparison_table = {}
    
    for i, name1 in enumerate(classifier_names):
        for j, name2 in enumerate(classifier_names):
            if i < j:  # Only compare each pair once
                y_pred1 = classifiers_results[name1]
                y_pred2 = classifiers_results[name2]
                
                statistic, p_value, significant = mcnemar_test(y_true, y_pred1, y_pred2)
                
                comparison_table[f"{name1}_vs_{name2}"] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': significant
                }
                
                if config.VERBOSE:
                    print(f"\n{name1} vs {name2}:")
                    print(f"  McNemar statistic: {statistic:.4f}")
                    print(f"  P-value: {p_value:.4f}")
                    if significant:
                        print(f"  → Significantly different at {config.CONFIDENCE_LEVEL*100}% level")
                    else:
                        print(f"  → Not significantly different")
    
    return comparison_table


def save_statistical_tests(results, filename=None):
    """
    Save statistical test results to file.
    
    Parameters:
    -----------
    results : dict
        Statistical test results
    filename : str, optional
        Output filename
    """
    if filename is None:
        filename = config.STATISTICAL_TESTS_FILE
    
    with open(filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("STATISTICAL ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")
        
        for test_name, test_results in results.items():
            f.write(f"\n{test_name}:\n")
            f.write("-"*40 + "\n")
            for key, value in test_results.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
    
    if config.VERBOSE:
        print(f"\nStatistical tests saved to: {filename}")


def analyze_error_patterns(y_true, y_pred, X=None):
    """
    Analyze patterns in classification errors.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predictions
    X : ndarray, optional
        Features (for analyzing error characteristics)
        
    Returns:
    --------
    error_analysis : dict
        Dictionary with error analysis
    """
    errors = y_true != y_pred
    n_errors = errors.sum()
    error_rate = n_errors / len(y_true)
    
    analysis = {
        'total_errors': n_errors,
        'error_rate': error_rate,
        'false_positives': np.sum((y_true == 0) & (y_pred == 1)),
        'false_negatives': np.sum((y_true == 1) & (y_pred == 0))
    }
    
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("ERROR ANALYSIS")
        print(f"{'='*60}")
        print(f"  Total errors: {n_errors} / {len(y_true)} ({error_rate*100:.2f}%)")
        print(f"  False positives: {analysis['false_positives']}")
        print(f"  False negatives: {analysis['false_negatives']}")
    
    if X is not None and n_errors > 0:
        # Analyze error characteristics
        error_features = X[errors]
        correct_features = X[~errors]
        
        analysis['error_feature_mean'] = error_features.mean(axis=0)
        analysis['correct_feature_mean'] = correct_features.mean(axis=0)
        
        if config.VERBOSE:
            print(f"\n  Error patterns in features (mean difference):")
            diff = np.abs(error_features.mean(axis=0) - correct_features.mean(axis=0))
            top_diff_indices = np.argsort(diff)[-5:][::-1]
            for idx in top_diff_indices:
                print(f"    Feature {idx}: {diff[idx]:.4f}")
    
    return analysis

