"""Benchmark the CV stage of the real CRO fitness with threaded folds."""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cro_utils import sample_bounded_integers, threaded_cross_val_score


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data" / "Paper"
MAX_LAG = 180
MAX_WINDOW = 60
MAX_SHIFT = MAX_LAG + MAX_WINDOW
LAST_TRAIN_YEAR = 2010
CV_FOLDS = 5
LOGREG_PARAMS = {"class_weight": "balanced", "tol": 1e-2}


def prepare_real_candidate(seed):
    """Build only the columns used by one reproducible, unbiased candidate."""

    predictors = pd.read_csv(DATA_DIR / "predictors_dataset.csv", index_col=0)
    predictors.index = pd.to_datetime(predictors.index)
    target = pd.read_csv(DATA_DIR / "target.csv", index_col=0)
    target.index = pd.to_datetime(target.index)

    valid_index = predictors.index[MAX_SHIFT:].intersection(target.index)
    train_mask = valid_index.year <= LAST_TRAIN_YEAR
    train_dates = valid_index[train_mask]
    train_positions = predictors.index.get_indexer(train_dates)
    y_train = target.reindex(train_dates)["Target"].to_numpy()

    driver_count = predictors.shape[1]
    lower_bounds = np.concatenate(
        (
            np.ones(driver_count, dtype=np.int64),
            np.zeros(driver_count, dtype=np.int64),
            np.zeros(driver_count, dtype=np.int64),
        )
    )
    upper_bounds = np.concatenate(
        (
            np.full(driver_count, MAX_WINDOW, dtype=np.int64),
            np.full(driver_count, MAX_LAG, dtype=np.int64),
            np.ones(driver_count, dtype=np.int64),
        )
    )
    solution = sample_bounded_integers(
        lower_bounds, upper_bounds, rng=np.random.default_rng(seed)
    )

    windows = solution[:driver_count]
    starting_lags = solution[driver_count : 2 * driver_count]
    selected = solution[2 * driver_count :]
    variable_indices = []
    lag_indices = []
    for driver_index in np.flatnonzero(selected):
        for lag in range(
            starting_lags[driver_index],
            starting_lags[driver_index] + windows[driver_index],
        ):
            if 1 <= lag <= MAX_SHIFT:
                variable_indices.append(driver_index)
                lag_indices.append(lag)

    variable_indices = np.asarray(variable_indices, dtype=np.int64)
    lag_indices = np.asarray(lag_indices, dtype=np.int64)
    values = predictors.to_numpy(dtype=np.float32)
    X_train = values[
        train_positions[:, None] - lag_indices[None, :],
        variable_indices[None, :],
    ]
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale == 0] = 1
    X_train = (X_train - mean) / scale

    return X_train, y_train, int(selected.sum()), solution


def evaluate(X, y, n_jobs):
    estimator = LogisticRegression(**LOGREG_PARAMS)
    start = time.perf_counter()
    scores = threaded_cross_val_score(
        estimator,
        X,
        y,
        cv=CV_FOLDS,
        scoring="f1",
        n_jobs=n_jobs,
        blas_threads=1,
    )
    return scores, time.perf_counter() - start


def benchmark(seed, repeats, workers):
    if workers < 2:
        raise ValueError("The parallel benchmark needs at least 2 workers.")

    X, y, active_drivers, _ = prepare_real_candidate(seed)
    print(
        f"Candidate: {X.shape[0]} rows, {X.shape[1]} lagged columns, "
        f"{active_drivers} active drivers"
    )
    print(f"Comparing 1 worker with {workers} threaded fold workers")
    print("Timing scope: cross-validation only (candidate preparation excluded)")

    # Warm both implementation paths and the underlying numerical libraries.
    evaluate(X, y, 1)
    evaluate(X, y, workers)

    sequential_times = []
    parallel_times = []
    reference_scores = None
    for repetition in range(repeats):
        order = (1, workers) if repetition % 2 == 0 else (workers, 1)
        for jobs in order:
            scores, duration = evaluate(X, y, jobs)
            if reference_scores is None:
                reference_scores = scores
            else:
                np.testing.assert_allclose(
                    reference_scores, scores, rtol=1e-12, atol=1e-12
                )
            if jobs == 1:
                sequential_times.append(duration)
            else:
                parallel_times.append(duration)

    sequential_median = float(np.median(sequential_times))
    parallel_median = float(np.median(parallel_times))
    speedup = sequential_median / parallel_median
    print(f"Fold F1 scores: {np.array2string(reference_scores, precision=8)}")
    print(f"Mean F1: {reference_scores.mean():.10f}")
    print(f"Sequential median: {sequential_median:.4f} s")
    print(f"Threaded median:   {parallel_median:.4f} s")
    print(f"Speedup:           {speedup:.2f}x")
    print("Score equivalence: within 1e-12")

    return speedup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--workers", type=int, default=min(CV_FOLDS, os.cpu_count() or 1)
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.workers < 2:
        raise SystemExit("--workers must be at least 2")
    benchmark(args.seed, args.repeats, min(args.workers, CV_FOLDS))
