"""Compare real CRO candidate-level threading with deterministic short runs."""

import argparse
import contextlib
import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "CRO_Spatiotemporal_FS.py"
PYCRO_ROOT = ROOT / "Experiments" / "Paper"
RESULT_PREFIX = "CRO_BENCHMARK_RESULT "


def benchmark_environment(
    workers, cv_workers, neval, seed, temporary_directory
):
    environment = os.environ.copy()
    python_path = os.pathsep.join((str(PYCRO_ROOT), str(ROOT)))
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]

    environment.update(
        {
            "BLIS_NUM_THREADS": "1",
            "CRO_CV_N_JOBS": str(cv_workers),
            "CRO_NEVAL": str(neval),
            "CRO_N_JOBS": str(workers),
            "CRO_SEED": str(seed),
            "CRO_VERBOSE": "0",
            "MKL_NUM_THREADS": "1",
            "MPLCONFIGDIR": str(temporary_directory / "matplotlib"),
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": python_path,
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    return environment


def runner_wrapper():
    """Return a small script that reports metrics outside the production runner."""

    runner = repr(str(RUNNER))
    return f"""
import contextlib
import hashlib
import io
import json
import resource
import runpy
import sys
import numpy as np

buffer = io.StringIO()
try:
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        namespace = runpy.run_path({runner}, run_name='__main__')
except BaseException:
    print(buffer.getvalue(), file=sys.stderr)
    raise

cro_alg = namespace['cro_alg']
objfunc = namespace['objfunc']
solution = namespace['solution']
population = np.vstack([
    np.asarray(coral.solution, dtype=np.int64)
    for coral in cro_alg.population.population
])
result = {{
    'best_fitness': float(namespace['obj_value']),
    'cache': objfunc.fitness_cache.snapshot(),
    'candidate_requests': int(objfunc.counter),
    'cro_workers': int(namespace['CRO_N_JOBS']),
    'cv_workers': int(namespace['CV_N_JOBS']),
    'elapsed_s': float(cro_alg.real_time_spent),
    'generations': len(cro_alg.history),
    'history_sha256': hashlib.sha256(
        np.asarray(cro_alg.history, dtype=np.float64).tobytes()
    ).hexdigest(),
    'population_sha256': hashlib.sha256(population.tobytes()).hexdigest(),
    'solution_sha256': hashlib.sha256(
        np.asarray(solution, dtype=np.int64).tobytes()
    ).hexdigest(),
    'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20),
}}
print({RESULT_PREFIX!r} + json.dumps(result, sort_keys=True))
"""


def run_once(workers, neval, seed, timeout, cv_workers=1):
    with tempfile.TemporaryDirectory(
        prefix=f"stco-cro-{workers}x{cv_workers}-"
    ) as temp_name:
        temporary_directory = Path(temp_name)
        (temporary_directory / "Data").symlink_to(
            ROOT / "Data", target_is_directory=True
        )
        (temporary_directory / "matplotlib").mkdir()
        environment = benchmark_environment(
            workers, cv_workers, neval, seed, temporary_directory
        )

        started = time.perf_counter()
        process = subprocess.Popen(
            [sys.executable, "-c", runner_wrapper()],
            cwd=temporary_directory,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        monitored = psutil.Process(process.pid)
        peak_rss = 0
        peak_threads = 0
        peak_children = 0
        deadline = started + timeout
        while process.poll() is None:
            children = []
            try:
                children = monitored.children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                pass
            try:
                rss = monitored.memory_info().rss
                for child in children:
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                        pass
                peak_rss = max(peak_rss, rss)
                peak_threads = max(peak_threads, monitored.num_threads())
                peak_children = max(peak_children, len(children))
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                pass
            if time.perf_counter() > deadline:
                process.kill()
                stdout, stderr = process.communicate()
                raise TimeoutError(
                    f"CRO benchmark with {workers} workers timed out.\n"
                    f"stdout:\n{stdout[-2000:]}\nstderr:\n{stderr[-2000:]}"
                )
            time.sleep(0.05)

        stdout, stderr = process.communicate()
        wall_time = time.perf_counter() - started
        if process.returncode != 0:
            raise RuntimeError(
                f"CRO benchmark with {workers} workers failed ({process.returncode}).\n"
                f"stdout:\n{stdout[-4000:]}\nstderr:\n{stderr[-4000:]}"
            )

        result_lines = [
            line for line in stdout.splitlines() if line.startswith(RESULT_PREFIX)
        ]
        if len(result_lines) != 1:
            raise RuntimeError(
                f"Expected one benchmark result for {workers} workers; got "
                f"{len(result_lines)}.\nstdout:\n{stdout[-4000:]}"
            )
        result = json.loads(result_lines[0][len(RESULT_PREFIX) :])
        observed_rss_mib = (
            peak_rss / 2**20 if peak_rss else result["peak_rss_mib"]
        )
        result.update(
            {
                "e2e_wall_s": wall_time,
                "observed_peak_children": peak_children,
                "observed_peak_rss_mib": observed_rss_mib,
                "observed_peak_threads": peak_threads,
            }
        )
        return result


def assert_equivalent(reference, candidate):
    exact_fields = (
        "candidate_requests",
        "generations",
        "history_sha256",
        "population_sha256",
        "solution_sha256",
    )
    for field in exact_fields:
        if candidate[field] != reference[field]:
            raise AssertionError(
                f"Parallel run changed {field}: "
                f"{reference[field]!r} != {candidate[field]!r}"
            )

    for field in ("calls", "hits", "misses", "errors", "currsize", "inflight"):
        if candidate["cache"][field] != reference["cache"][field]:
            raise AssertionError(
                f"Parallel run changed cache.{field}: "
                f"{reference['cache'][field]!r} != {candidate['cache'][field]!r}"
            )

    np.testing.assert_allclose(
        candidate["best_fitness"],
        reference["best_fitness"],
        rtol=1e-12,
        atol=1e-12,
    )


def benchmark(workers, fold_workers, neval, seed, timeout):
    results = []
    for worker_count in workers:
        print(f"Running CRO with {worker_count} candidate worker(s)...", flush=True)
        result = run_once(worker_count, neval, seed, timeout, cv_workers=1)
        results.append(result)
        print(
            f"  CRO={result['elapsed_s']:.3f}s, "
            f"E2E={result['e2e_wall_s']:.3f}s, "
            f"RSS={result['observed_peak_rss_mib']:.1f} MiB, "
            f"misses={result['cache']['misses']}"
        )

    if fold_workers > 1:
        print(
            f"Running CRO with 1 candidate worker and {fold_workers} fold workers...",
            flush=True,
        )
        result = run_once(1, neval, seed, timeout, cv_workers=fold_workers)
        results.append(result)
        print(
            f"  CRO={result['elapsed_s']:.3f}s, "
            f"E2E={result['e2e_wall_s']:.3f}s, "
            f"RSS={result['observed_peak_rss_mib']:.1f} MiB, "
            f"misses={result['cache']['misses']}"
        )

    reference = results[0]
    for result in results[1:]:
        assert_equivalent(reference, result)

    print("\nDeterminism: solution, history, population and fitness match.")
    print("CROxCV  cro_s   speedup  efficiency  misses/s  rss_mib  threads  children")
    for result in results:
        speedup = reference["elapsed_s"] / result["elapsed_s"]
        parallel_workers = max(result["cro_workers"], result["cv_workers"])
        efficiency = speedup / parallel_workers
        throughput = result["cache"]["misses"] / result["elapsed_s"]
        label = f"{result['cro_workers']}x{result['cv_workers']}"
        print(
            f"{label:>5}  {result['elapsed_s']:>6.2f}  "
            f"{speedup:>7.2f}x  {efficiency:>10.2%}  {throughput:>8.2f}  "
            f"{result['observed_peak_rss_mib']:>7.1f}  "
            f"{result['observed_peak_threads']:>7}  "
            f"{result['observed_peak_children']:>8}"
        )
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--fold-workers", type=int, default=5)
    parser.add_argument("--neval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--timeout", type=float, default=600)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.workers or any(worker < 1 for worker in args.workers):
        raise SystemExit("--workers values must be positive")
    if args.workers[0] != 1:
        raise SystemExit("The first --workers value must be 1 as the baseline")
    if args.fold_workers < 1:
        raise SystemExit("--fold-workers must be positive")
    if not 1 <= args.neval <= 1000:
        raise SystemExit("--neval must be between 1 and 1000")
    benchmark(
        args.workers,
        args.fold_workers,
        args.neval,
        args.seed,
        args.timeout,
    )
