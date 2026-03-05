import numpy as np


def precision(y_true, y_pred):
    """
    Calculate the precision of the predictions.
    
    Parameters:
    y_true : array-like, shape (n_samples,)
        True binary labels in range {0, 1} or {-1, 1}. 
    y_pred : array-like, shape (n_samples,)
        Predicted binary labels in range {0, 1} or {-1, 1}.

    Returns:
    float
        The precision score.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    """
    Calculate the recall of the predictions.
    
    Parameters:
    y_true : array-like, shape (n_samples,)
        True binary labels in range {0, 1} or {-1, 1}. 
    y_pred : array-like, shape (n_samples,)
        Predicted binary labels in range {0, 1} or {-1, 1}.

    Returns:
    float
        The recall score.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    """
    Calculate the F1 score of the predictions.
    
    Parameters:
    y_true : array-like, shape (n_samples,)
        True binary labels in range {0, 1} or {-1, 1}. 
    y_pred : array-like, shape (n_samples,)
        Predicted binary labels in range {0, 1} or {-1, 1}.

    Returns:
    float
        The F1 score.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


# Example usage:
# y_true = np.array([1, 0, 1, 1, 0])
# y_pred = np.array([1, 0, 1, 0, 0])
# print(f'Precision: {precision(y_true, y_pred)}')
# print(f'Recall: {recall(y_true, y_pred)}')
# print(f'F1 Score: {f1_score(y_true, y_pred)}')
