"""
Evaluation metrics for forecast comparison.

Includes the Diebold-Mariano test, which is the standard way finance ML
papers establish that one model's forecast errors are *significantly*
different from another's (not just numerically different by chance).
Without this test, a claim like "XGBoost beat LSTM" is just an anecdote
from one dataset; reviewers in this subfield will often ask for it.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2_Score": r2_score(y_true, y_pred),
        "Directional_Accuracy": directional_accuracy(y_true, y_pred),
    }


def directional_accuracy(y_true, y_pred) -> float:
    """
    Fraction of days where predicted sign matches actual sign.
    Often more meaningful than R^2 for a trading-relevant claim.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def diebold_mariano(y_true, pred_a, pred_b, h=1, power=2):
    """
    Diebold-Mariano test comparing forecast accuracy of two models.

    H0: both models have equal predictive accuracy.
    Returns (DM_statistic, p_value). A significant p-value (e.g. < 0.05)
    means the accuracy difference is unlikely to be due to chance.

    h: forecast horizon (1 for one-step-ahead, which is our case)
    power: 1 for absolute-error loss differential, 2 for squared-error
    """
    y_true = np.asarray(y_true)
    e_a = y_true - np.asarray(pred_a)
    e_b = y_true - np.asarray(pred_b)

    if power == 1:
        d = np.abs(e_a) - np.abs(e_b)
    else:
        d = e_a ** 2 - e_b ** 2

    n = len(d)
    d_bar = np.mean(d)

    # Newey-West style variance estimate accounting for autocorrelation
    # up to lag h-1 (h=1 -> just the variance, no autocovariance terms)
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        gamma_lag = np.cov(d[:-lag], d[lag:])[0, 1]
        var_d += 2 * (1 - lag / h) * gamma_lag
    var_d /= n

    if var_d <= 0:
        return np.nan, np.nan

    dm_stat = d_bar / np.sqrt(var_d)
    # Harvey, Leybourne, Newbold (1997) small-sample correction
    hln_correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * hln_correction

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), df=n - 1))
    return dm_stat_corrected, p_value


def pairwise_dm_matrix(predictions: dict, y_true) -> "pd.DataFrame":
    """
    Runs DM test for every pair of models, returns a p-value matrix.
    Use this to build the significance table in the Results section.
    """
    import pandas as pd
    names = list(predictions.keys())
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)

    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if i == j:
                matrix.loc[name_a, name_b] = np.nan
                continue
            _, p = diebold_mariano(y_true, predictions[name_a], predictions[name_b])
            matrix.loc[name_a, name_b] = p

    return matrix
