"""
Walk-forward (rolling-origin) validation.

A single 80/20 chronological split answers "did this model work on one
window of history?" Walk-forward validation answers "does this model's
edge persist across many different market regimes?" — which is what a
reviewer will actually ask about a finance ML paper.

Each fold:
  [ ---- train (TRAIN_WINDOW_DAYS) ---- ][ test (TEST_WINDOW_DAYS) ]
                       roll forward by STEP_DAYS
  [       ---- train ----          ][ test ]
"""
from dataclasses import dataclass
from typing import List
import numpy as np
from . import config


@dataclass
class Fold:
    fold_id: int
    train_start: int
    train_end: int   # exclusive
    test_start: int
    test_end: int     # exclusive


def generate_folds(n_rows: int) -> List[Fold]:
    """
    Generate walk-forward fold index boundaries for a series of length n_rows.
    """
    folds = []
    fold_id = 0
    train_start = 0
    train_end = config.TRAIN_WINDOW_DAYS

    while train_end + config.TEST_WINDOW_DAYS <= n_rows:
        test_start = train_end
        test_end = test_start + config.TEST_WINDOW_DAYS

        folds.append(Fold(fold_id, train_start, train_end, test_start, test_end))

        fold_id += 1
        train_end += config.STEP_DAYS
        # NOTE: train_start stays 0 -> "expanding window". Set train_start
        # to (train_end - config.TRAIN_WINDOW_DAYS) instead for a fixed-size
        # "rolling window" if you'd rather cap training data recency.

    return folds


def rolling_window_folds(n_rows: int) -> List[Fold]:
    """
    Alternative to generate_folds: fixed-size rolling window instead of
    expanding window. Use this variant if you want to test robustness to
    a fixed amount of training history (common ablation in the paper).
    """
    folds = []
    fold_id = 0
    train_start = 0
    train_end = config.TRAIN_WINDOW_DAYS

    while train_end + config.TEST_WINDOW_DAYS <= n_rows:
        test_start = train_end
        test_end = test_start + config.TEST_WINDOW_DAYS

        folds.append(Fold(fold_id, train_start, train_end, test_start, test_end))

        fold_id += 1
        train_start += config.STEP_DAYS
        train_end += config.STEP_DAYS

    return folds
