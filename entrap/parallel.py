"""Bounded, deterministic thread scheduling for independent ENTRAP simulations.

Tasks are produced on the calling thread, so seeded RNG draws keep a stable order.
Only a small, bounded number of task payloads are retained at once, avoiding the
memory blow-up caused by eagerly submitting every Monte Carlo replicate.
"""
from __future__ import annotations
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os


def resolve_workers(n_jobs, n_tasks, parallel_max_workers=8,
                    parallel_memory_mb=1024, bytes_per_worker=0):
    """Resolve a safe worker count from task, CPU, and memory limits."""
    n_tasks = int(n_tasks)
    if n_tasks <= 1:
        return 1
    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be zero, use -1 or a positive integer")
    if n_jobs == 1:
        return 1
    requested = (os.cpu_count() or 1) if n_jobs < 0 else n_jobs
    hard_cap = max(1, int(parallel_max_workers))
    workers = min(requested, hard_cap, n_tasks)
    budget = max(1, int(parallel_memory_mb)) * 1024 * 1024
    if bytes_per_worker > 0:
        memory_cap = max(1, budget // int(bytes_per_worker))
        workers = min(workers, int(memory_cap))
    return max(1, int(workers))


def ordered_thread_map(function, items, workers):
    """Execute independent tasks in parallel while preserving input order."""
    workers = max(1, int(workers))
    iterator = iter(items)
    if workers == 1:
        for item in iterator:
            yield function(item)
        return

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="entrap-mc") as pool:
        pending = deque()
        for _ in range(workers):
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            future = pending.popleft()
            yield future.result()
            try:
                item = next(iterator)
            except StopIteration:
                continue
            pending.append(pool.submit(function, item))
