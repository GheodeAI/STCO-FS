"""GPU prototype for the CRO objective: batched logistic-regression CV.

Validates a batched (candidates x folds) GPU implementation of the exact
production objective -- StandardScaler-free global standardization, 5-fold
StratifiedKFold, LogisticRegression(class_weight='balanced') with sklearn's
L2 objective, f1 scoring -- against sklearn on the real dataset, then
benchmarks throughput. Device-agnostic: picks cuda > mps > cpu, so the same
script runs on this Mac (MPS, float32) and on the NVIDIA server (CUDA,
optionally float64 for a tighter numerical match).

Usage:
    python gpu_cv_prototype.py --candidates 24 --chunk 4
    python gpu_cv_prototype.py --device cuda --dtype float64 --candidates 64
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import spearmanr

from cro_utils import sample_bounded_integers

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data" / "Paper"
MAX_LAG = 180
MAX_WINDOW = 60
MAX_SHIFT = MAX_LAG + MAX_WINDOW
FIRST_TRAIN_YEAR = 1950
LAST_TRAIN_YEAR = 2010
CV_FOLDS = 5
LOGREG_PARAMS = {"class_weight": "balanced"}
L2_C = 1.0  # sklearn LogisticRegression default C


def load_training_matrix():
    """Build X_train_full/y/mu/sigma exactly like CRO_Spatiotemporal_FS.py."""

    predictors = pd.read_csv(DATA_DIR / "predictors_dataset.csv", index_col=0)
    predictors.index = pd.to_datetime(predictors.index)
    target = pd.read_csv(DATA_DIR / "target.csv", index_col=0)
    target.index = pd.to_datetime(target.index)

    n_len = len(predictors)
    idx_pred = predictors.index[MAX_SHIFT:]
    valid_idx = idx_pred.intersection(target.index)
    col_index = {}
    n_lags = MAX_SHIFT
    pos = idx_pred.get_indexer(valid_idx)
    X_full = np.empty(
        (len(valid_idx), predictors.shape[1] * n_lags), dtype=np.float32
    )
    col_id = 0
    for var_i, col in enumerate(predictors.columns):
        s = predictors[col].to_numpy()
        for lag in range(1, n_lags + 1):
            X_full[:, col_id] = s[MAX_SHIFT - lag : n_len - lag][pos]
            col_index[(var_i, lag)] = col_id
            col_id += 1

    y_full = target.reindex(valid_idx)["Target"].to_numpy()
    train_mask = (valid_idx.year >= FIRST_TRAIN_YEAR) & (
        valid_idx.year <= LAST_TRAIN_YEAR
    )
    X_train = X_full[train_mask]
    y_train = y_full[train_mask].astype(np.int64)
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1
    return X_train, y_train, mu, sigma, col_index, predictors.shape[1]


def sample_candidates(count, driver_count, col_index, seed0=0):
    """Reproducible candidates from the same generator the CRO uses."""

    lower = np.concatenate(
        (
            np.ones(driver_count, dtype=np.int64),
            np.zeros(driver_count, dtype=np.int64),
            np.zeros(driver_count, dtype=np.int64),
        )
    )
    upper = np.concatenate(
        (
            np.full(driver_count, MAX_WINDOW, dtype=np.int64),
            np.full(driver_count, MAX_LAG, dtype=np.int64),
            np.ones(driver_count, dtype=np.int64),
        )
    )
    candidates = []
    seed = seed0
    while len(candidates) < count:
        solution = sample_bounded_integers(
            lower, upper, rng=np.random.default_rng(seed)
        )
        seed += 1
        windows = solution[:driver_count]
        lags0 = solution[driver_count : 2 * driver_count]
        selected = solution[2 * driver_count :]
        cols = []
        for i in np.flatnonzero(selected):
            for lag in range(lags0[i], lags0[i] + windows[i]):
                if 1 <= lag <= MAX_SHIFT:
                    cols.append(col_index[(i, lag)])
        if cols:
            candidates.append(np.asarray(cols, dtype=np.int64))
    return candidates


def gpu_batched_cv(
    candidates, X_dev, y_np, mu_dev, sigma_dev, folds, device, dtype, chunk,
    max_iter=200,
):
    """Batched 5-fold CV for a list of candidates. Returns (n_cand, 5) f1."""

    y_signed = torch.as_tensor(
        2.0 * y_np - 1.0, dtype=dtype, device=device
    )
    fold_rows = [
        (
            torch.as_tensor(tr, dtype=torch.long, device=device),
            torch.as_tensor(va, dtype=torch.long, device=device),
        )
        for tr, va in folds
    ]
    # sklearn's balanced weights are computed on each fold's own training y.
    fold_weights = []
    for tr, _ in folds:
        y_tr = y_np[tr]
        n, n_pos = len(y_tr), int(y_tr.sum())
        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * (n - n_pos))
        w = np.where(y_tr == 1, w_pos, w_neg)
        fold_weights.append(torch.as_tensor(w, dtype=dtype, device=device))

    all_f1 = np.zeros((len(candidates), CV_FOLDS), dtype=np.float64)
    for chunk_start in range(0, len(candidates), chunk):
        chunk_cands = candidates[chunk_start : chunk_start + chunk]
        d_max = max(len(c) for c in chunk_cands)
        n_chunk = len(chunk_cands)

        # Standardized, zero-padded candidate matrices: (n_chunk, n_rows, d_max).
        # Padded columns keep coefficient exactly 0 under the L2 penalty.
        X_std = torch.zeros(
            (n_chunk, X_dev.shape[0], d_max), dtype=dtype, device=device
        )
        for j, cols in enumerate(chunk_cands):
            idx = torch.as_tensor(cols, dtype=torch.long, device=device)
            X_std[j, :, : len(cols)] = (
                X_dev.index_select(1, idx) - mu_dev.index_select(0, idx)
            ) / sigma_dev.index_select(0, idx)

        # One optimization problem per (candidate, fold): B = n_chunk * folds.
        # The losses are independent, so optimizing their sum solves each
        # strictly convex problem to its unique minimum (same as sklearn's).
        problems = []
        for f, (tr_rows, va_rows) in enumerate(fold_rows):
            problems.append(
                (
                    X_std.index_select(1, tr_rows),
                    y_signed.index_select(0, tr_rows),
                    fold_weights[f],
                )
            )

        beta = torch.zeros(
            (n_chunk, CV_FOLDS, d_max), dtype=dtype, device=device,
            requires_grad=True,
        )
        bias = torch.zeros(
            (n_chunk, CV_FOLDS), dtype=dtype, device=device, requires_grad=True
        )
        optimizer = torch.optim.LBFGS(
            [beta, bias],
            max_iter=max_iter,
            history_size=20,
            tolerance_grad=1e-7 if dtype == torch.float64 else 1e-6,
            tolerance_change=0.0,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            total = beta.pow(2).sum() * 0.5  # L2 penalty, intercept excluded
            for f, (X_tr, y_tr, w_tr) in enumerate(problems):
                logits = (
                    torch.einsum("cnd,cd->cn", X_tr, beta[:, f, :])
                    + bias[:, f].unsqueeze(1)
                )
                margins = y_tr.unsqueeze(0) * logits
                total = total + L2_C * (
                    w_tr.unsqueeze(0) * torch.nn.functional.softplus(-margins)
                ).sum()
            total.backward()
            return total

        optimizer.step(closure)

        with torch.no_grad():
            for f, (_, va_rows) in enumerate(fold_rows):
                X_va = X_std.index_select(1, va_rows)
                logits = (
                    torch.einsum("cnd,cd->cn", X_va, beta[:, f, :])
                    + bias[:, f].unsqueeze(1)
                )
                predictions = (logits > 0).to(torch.int64).cpu().numpy()
                y_va = y_np[va_rows.cpu().numpy()]
                for j in range(n_chunk):
                    all_f1[chunk_start + j, f] = f1_score(
                        y_va, predictions[j], zero_division=0.0
                    )
    return all_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float64"])
    parser.add_argument("--skip-cpu", action="store_true",
                        help="GPU throughput only (no sklearn reference).")
    parser.add_argument("--seed0", type=int, default=0)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if device.type == "mps" and dtype == torch.float64:
        raise SystemExit("MPS does not support float64; use --dtype float32.")
    print(f"device={device.type}  dtype={args.dtype}  torch={torch.__version__}")

    X_train, y_train, mu, sigma, col_index, driver_count = load_training_matrix()
    print(f"X_train={X_train.shape}  positives={y_train.mean():.3%}")
    candidates = sample_candidates(
        args.candidates, driver_count, col_index, seed0=args.seed0
    )
    sizes = [len(c) for c in candidates]
    print(f"candidates={len(candidates)}  n_cols min/med/max="
          f"{min(sizes)}/{int(np.median(sizes))}/{max(sizes)}")

    # Same folds sklearn's cross_val_score uses for a classifier: stratified,
    # no shuffling. y is shared by every candidate, so folds are shared too.
    skf = StratifiedKFold(n_splits=CV_FOLDS)
    folds = [
        (tr.astype(np.int64), va.astype(np.int64))
        for tr, va in skf.split(np.zeros_like(y_train), y_train)
    ]

    # --- GPU path (upload once, then batched fits) ---
    t0 = time.perf_counter()
    X_dev = torch.as_tensor(X_train, dtype=dtype, device=device)
    mu_dev = torch.as_tensor(mu, dtype=dtype, device=device)
    sigma_dev = torch.as_tensor(sigma, dtype=dtype, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    upload_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    gpu_f1 = gpu_batched_cv(
        candidates, X_dev, y_train, mu_dev, sigma_dev, folds, device, dtype,
        chunk=args.chunk,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu_s = time.perf_counter() - t0
    gpu_cv_mean = gpu_f1.mean(axis=1)
    print(f"\nGPU: upload={upload_s:.2f}s  fit+score={gpu_s:.2f}s  "
          f"({gpu_s / len(candidates):.3f} s/eval)")

    if args.skip_cpu:
        return

    # --- CPU reference: the exact production objective ---
    t0 = time.perf_counter()
    cpu_f1 = np.zeros_like(gpu_f1)
    for i, cols in enumerate(candidates):
        X_std = (X_train[:, cols] - mu[cols]) / sigma[cols]
        clf = LogisticRegression(**LOGREG_PARAMS)
        cpu_f1[i] = cross_val_score(
            clf, X_std, y_train, cv=CV_FOLDS, scoring="f1"
        )
    cpu_s = time.perf_counter() - t0
    cpu_cv_mean = cpu_f1.mean(axis=1)
    print(f"CPU sklearn reference: {cpu_s:.2f}s  "
          f"({cpu_s / len(candidates):.3f} s/eval)")

    fold_diff = np.abs(gpu_f1 - cpu_f1)
    mean_diff = np.abs(gpu_cv_mean - cpu_cv_mean)
    rho = spearmanr(1.0 / cpu_cv_mean, 1.0 / gpu_cv_mean).statistic
    print("\n--- Validation vs sklearn ---")
    print(f"per-fold |dF1|:  max={fold_diff.max():.2e}  "
          f"mean={fold_diff.mean():.2e}")
    print(f"cv-mean  |dF1|:  max={mean_diff.max():.2e}  "
          f"mean={mean_diff.mean():.2e}")
    print(f"fitness rank correlation (Spearman): {rho:.6f}")
    print(f"speedup vs sequential sklearn: {cpu_s / gpu_s:.2f}x")


if __name__ == "__main__":
    main()
