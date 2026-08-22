from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from cro_utils import (
    ConcurrentMemo,
    flip_driver_selection,
    sample_bounded_integers,
    threaded_cross_val_score,
)


DRIVER_COUNT = 71
LOWER_BOUNDS = np.concatenate(
    (
        np.ones(DRIVER_COUNT, dtype=np.int64),
        np.zeros(DRIVER_COUNT, dtype=np.int64),
        np.zeros(DRIVER_COUNT, dtype=np.int64),
    )
)
UPPER_BOUNDS = np.concatenate(
    (
        np.full(DRIVER_COUNT, 60, dtype=np.int64),
        np.full(DRIVER_COUNT, 180, dtype=np.int64),
        np.ones(DRIVER_COUNT, dtype=np.int64),
    )
)


def test_sampler_respects_bounds_and_is_not_driver_biased():
    rng = np.random.default_rng(20260710)
    samples = np.stack(
        [
            sample_bounded_integers(LOWER_BOUNDS, UPPER_BOUNDS, rng=rng)
            for _ in range(10_000)
        ]
    )
    windows = samples[:, :DRIVER_COUNT]
    lags = samples[:, DRIVER_COUNT : 2 * DRIVER_COUNT]
    selection = samples[:, 2 * DRIVER_COUNT :]

    assert windows.min() == 1
    assert windows.max() == 60
    assert abs(windows.mean() - 30.5) < 0.1
    assert lags.min() == 0
    assert lags.max() == 180
    assert abs(lags.mean() - 90.0) < 0.3
    assert set(np.unique(selection)) == {0, 1}
    assert abs(selection.mean() - 0.5) < 0.002


def test_sampler_is_reproducible_with_the_global_numpy_seed():
    previous_state = np.random.get_state()
    try:
        np.random.seed(1234)
        first = sample_bounded_integers(LOWER_BOUNDS, UPPER_BOUNDS)
        np.random.seed(1234)
        second = sample_bounded_integers(LOWER_BOUNDS, UPPER_BOUNDS)
    finally:
        np.random.set_state(previous_state)

    np.testing.assert_array_equal(first, second)


def test_driver_mutation_flips_only_selection_bits_symmetrically():
    previous_state = np.random.get_state()
    try:
        base = np.concatenate(
            (
                np.full(DRIVER_COUNT, 20),
                np.full(DRIVER_COUNT, 90),
                np.zeros(DRIVER_COUNT),
            )
        )
        np.random.seed(7)
        activated = flip_driver_selection(base, None, None, {"N": 5})
        all_selected = base.copy()
        all_selected[2 * DRIVER_COUNT :] = 1
        np.random.seed(7)
        deactivated = flip_driver_selection(all_selected, None, None, {"N": 5})
    finally:
        np.random.set_state(previous_state)

    np.testing.assert_array_equal(
        activated[: 2 * DRIVER_COUNT], base[: 2 * DRIVER_COUNT]
    )
    np.testing.assert_array_equal(
        deactivated[: 2 * DRIVER_COUNT], all_selected[: 2 * DRIVER_COUNT]
    )
    assert activated[2 * DRIVER_COUNT :].sum() == 5
    assert deactivated[2 * DRIVER_COUNT :].sum() == DRIVER_COUNT - 5


def test_threaded_folds_match_sequential_scores():
    X, y = make_classification(
        n_samples=800,
        n_features=20,
        n_informative=10,
        weights=[0.85, 0.15],
        random_state=42,
    )
    estimator = LogisticRegression(
        class_weight="balanced", tol=1e-2, random_state=42
    )
    sequential = threaded_cross_val_score(
        estimator, X, y, cv=5, scoring="f1", n_jobs=1
    )
    threaded = threaded_cross_val_score(
        estimator, X, y, cv=5, scoring="f1", n_jobs=2
    )

    np.testing.assert_allclose(sequential, threaded, rtol=1e-12, atol=1e-12)


def test_concurrent_memo_computes_one_key_once():
    memo = ConcurrentMemo()
    compute_lock = Lock()
    compute_calls = 0

    def compute():
        nonlocal compute_calls
        with compute_lock:
            compute_calls += 1
        time.sleep(0.05)
        return 42

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(memo.get_or_compute, "shared", compute)
            for _ in range(8)
        ]
    assert [future.result() for future in futures] == [42] * 8
    assert compute_calls == 1
    assert memo.snapshot() == {
        "calls": 8,
        "hits": 7,
        "ready_hits": 0,
        "waits": 7,
        "misses": 1,
        "errors": 0,
        "currsize": 1,
        "inflight": 0,
    }


def test_concurrent_memo_propagates_errors_and_allows_retry():
    memo = ConcurrentMemo()

    def fail():
        time.sleep(0.05)
        raise ValueError("expected")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(memo.get_or_compute, "key", fail) for _ in range(4)]
    for future in futures:
        try:
            future.result()
        except ValueError as error:
            assert str(error) == "expected"
        else:
            raise AssertionError("The memoized exception was not propagated.")

    assert memo.get_or_compute("key", lambda: 7) == 7
    metrics = memo.snapshot()
    assert metrics["misses"] == 2
    assert metrics["errors"] == 1
    assert metrics["inflight"] == 0
