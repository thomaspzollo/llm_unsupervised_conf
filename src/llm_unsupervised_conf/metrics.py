from __future__ import annotations

import pandas as pd
import numpy as np
import calibration as cal

import torch

from sklearn.calibration import calibration_curve




def get_ece(conf, labels, n_bins=12, p=2):
    ece = cal.lower_bound_scaling_ce(conf, labels, p=p, debias=False, num_bins=n_bins, binning_scheme=cal.get_equal_bins, mode='top-label')
    return ece

def get_ece1(conf, labels, n_bins=12):
    return get_ece(conf, labels, n_bins=12, p=1)

def get_ece2(conf, labels, n_bins=12):
    return get_ece(conf, labels, n_bins=12, p=2)


def get_nll(y_pred, y_true):
    """
    Compute binary cross-entropy loss between target values and predictions.

    Args:
    y_true (numpy array): Ground truth labels (1D array, values 0 or 1).
    y_pred (numpy array): Predicted probabilities (1D array, values between 0 and 1).

    Returns:
    float: Binary cross-entropy loss.
    """
    epsilon = 1e-6  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  # Clip predictions
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def get_mce(y_pred, y_true, n_bins=12, strategy="quantile"):
    """
    Maximum Calibration Error (MCE) for binary probabilistic predictions.

    MCE = max_b | acc_b - conf_b |
    where acc_b is prob_true (fraction of positives in bin b),
          conf_b is prob_pred (mean predicted prob in bin b).

    Notes:
      - Uses sklearn.calibration.calibration_curve, so bins with zero samples
        are omitted automatically.
      - y_pred should be probabilities in [0, 1].
    """
    prob_true, prob_pred = calibration_curve(
        y_true, y_pred, n_bins=n_bins, strategy=strategy
    )
    return float(np.max(np.abs(prob_true - prob_pred)))

