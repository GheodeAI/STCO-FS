"""Utilities shared by the CRO runner and its performance benchmarks."""

from concurrent.futures import Future
from contextlib import nullcontext
from threading import Lock, get_ident
from typing import Any, Optional

import numpy as np
from joblib import parallel_backend
from sklearn.model_selection import cross_val_score
from threadpoolctl import threadpool_limits


class ConcurrentMemo:
    """Thread-safe memoization that coalesces concurrent work for one key."""

    def __init__(self):
        self._lock = Lock()
        self._values = {}
        self._inflight = {}
        self._calls = 0
        self._ready_hits = 0
        self._waits = 0
        self._misses = 0
        self._errors = 0

    def get_or_compute(self, key, compute):
        """Return a cached value or compute it once across concurrent callers."""

        thread_id = get_ident()
        with self._lock:
            self._calls += 1
            if key in self._values:
                self._ready_hits += 1
                return self._values[key]

            pending = self._inflight.get(key)
            if pending is None:
                future = Future()
                self._inflight[key] = (future, thread_id)
                self._misses += 1
                owner = True
            else:
                future, owner_thread_id = pending
                if owner_thread_id == thread_id:
                    raise RuntimeError("Recursive computation for the same cache key.")
                self._waits += 1
                owner = False

        if not owner:
            return future.result()

        try:
            value = compute()
        except BaseException as error:
            with self._lock:
                self._inflight.pop(key, None)
                self._errors += 1
                future.set_exception(error)
            raise

        with self._lock:
            self._values[key] = value
            self._inflight.pop(key, None)
            future.set_result(value)
        return value

    def snapshot(self):
        """Return an atomic copy of cache metrics."""

        with self._lock:
            return {
                "calls": self._calls,
                "hits": self._ready_hits + self._waits,
                "ready_hits": self._ready_hits,
                "waits": self._waits,
                "misses": self._misses,
                "errors": self._errors,
                "currsize": len(self._values),
                "inflight": len(self._inflight),
            }


def sample_bounded_integers(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample every gene uniformly from its own inclusive integer bounds.

    When ``rng`` is omitted the legacy NumPy RNG is used deliberately: the
    vendored PyCROSL operators use that same RNG, so ``np.random.seed`` keeps a
    complete optimization run reproducible.
    """

    lower = np.asarray(lower_bounds, dtype=np.int64)
    upper = np.asarray(upper_bounds, dtype=np.int64)

    if lower.shape != upper.shape:
        raise ValueError("Lower and upper bounds must have the same shape.")
    if np.any(lower > upper):
        raise ValueError("Every lower bound must be <= its upper bound.")

    exclusive_upper = upper + 1
    if rng is None:
        return np.random.randint(lower, exclusive_upper, size=lower.shape).astype(
            np.int64, copy=False
        )
    return rng.integers(
        lower, exclusive_upper, size=lower.shape, dtype=np.int64
    )


def flip_driver_selection(solution, population, objfunc, params):
    """Symmetrically flip selection bits without corrupting lag/window genes.

    The signature is the one expected by ``SubstrateInt('Custom', ...)``.
    ``population`` and ``objfunc`` are intentionally unused.
    """

    del population, objfunc
    result = np.rint(np.asarray(solution)).astype(np.int64, copy=True)
    if result.size % 3 != 0:
        raise ValueError("A driver solution must contain three equally sized blocks.")

    driver_count = result.size // 3
    flip_count = min(max(int(params.get("N", 1)), 0), driver_count)
    if flip_count == 0:
        return result

    driver_indices = np.random.choice(
        driver_count, size=flip_count, replace=False
    )
    selection_indices = 2 * driver_count + driver_indices
    current_bits = np.clip(result[selection_indices], 0, 1)
    result[selection_indices] = 1 - current_bits
    return result


def threaded_cross_val_score(
    estimator,
    X,
    y,
    *,
    cv: int,
    scoring: str,
    n_jobs: int,
    blas_threads: int = 1,
    limit_blas: bool = True,
    error_score: Any = "raise",
) -> np.ndarray:
    """Run CV folds in shared-memory threads without BLAS oversubscription."""

    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1.")
    if blas_threads < 1:
        raise ValueError("blas_threads must be at least 1.")

    kwargs = {
        "cv": cv,
        "scoring": scoring,
        "n_jobs": n_jobs,
        "pre_dispatch": n_jobs,
        "error_score": error_score,
    }
    blas_context = (
        threadpool_limits(limits=blas_threads) if limit_blas else nullcontext()
    )
    with blas_context:
        if n_jobs == 1:
            return cross_val_score(estimator, X, y, **kwargs)
        with parallel_backend("threading", n_jobs=n_jobs):
            return cross_val_score(estimator, X, y, **kwargs)
