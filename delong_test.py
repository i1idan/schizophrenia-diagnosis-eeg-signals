import os
import numpy as np
from scipy import stats
import pandas as pd


def delong_test_fn(y_true, y_pred1, y_pred2):
    """
    Calculate the DeLong test statistic for comparing two ROC curves.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        The true binary labels.

    y_pred1 : array-like, shape (n_samples,)
        The predicted probabilities for the first classifier.

    y_pred2 : array-like, shape (n_samples,)
        The predicted probabilities for the second classifier.

    Returns
    -------
    z : float
        The DeLong test statistic.
    p : float
        The two-sided p-value for the DeLong test.
    """
    # Calculate the true positive rates (sensitivity) and false positive rates (1 - specificity)
    tpr1 = sum((y_true == 1) & (y_pred1 > 0.5)) / sum(y_true == 1)
    fpr1 = sum((y_true == 0) & (y_pred1 > 0.5)) / sum(y_true == 0)
    tpr2 = sum((y_true == 1) & (y_pred2 > 0.5)) / sum(y_true == 1)
    fpr2 = sum((y_true == 0) & (y_pred2 > 0.5)) / sum(y_true == 0)

    # Calculate the DeLong test statistic
    if tpr1 - tpr2 == 0:
        z = 0
    else:
        z = (tpr1 - tpr2) / np.sqrt(fpr1 * (1 - fpr1) + fpr2 * (1 - fpr2))

    # Calculate the two-sided p-value
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p


if __name__ == '__main__':
    checkpoints = "./checkpoints"
    model_name_f = "Transformer"
    model_name_s = "Transformer"
    train_n = 0
    df1 = pd.read_csv(os.path.join(checkpoints, model_name_f, f"{train_n}", "y_test_pred.csv"))
    df2 = pd.read_csv(os.path.join(checkpoints, model_name_s, f"{train_n}", "y_test_pred.csv"))
    assert np.all(df1['y_test'].values == df2['y_test'].values)
    z, p = delong_test_fn(df1['y_test'].values, df1['y_pred'].values, df2['y_pred'].values)
    print(z, p)
