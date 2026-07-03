def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    # Count basic outcomes
    tp = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 1)
    fn = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 0)
    fp = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 1)
    tn = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 0)
    
    # Construct the confusion matrix matching the example's shape
    confusion_matrix = [[tp, fn], [fp, tn]]
    
    total = tp + tn + fp + fn
    
    # Calculate performance metrics with fallback for division by zero
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    f1_denominator = (2 * tp + fp + fn)
    f1 = (2 * tp) / f1_denominator if f1_denominator > 0 else 0.0
    
    spec_denominator = (tn + fp)
    specificity = tn / spec_denominator if spec_denominator > 0 else 0.0
    
    npv_denominator = (tn + fn)
    negativePredictive = tn / npv_denominator if npv_denominator > 0 else 0.0
    
    return (
        confusion_matrix, 
        round(accuracy, 3), 
        round(f1, 3), 
        round(specificity, 3), 
        round(negativePredictive, 3)
    )

