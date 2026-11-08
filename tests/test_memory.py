"""Tests for optimizer memory caching and warm start functionality."""

import numpy as np
import pandas as pd
import pytest

from gradient_free_optimizers import RandomSearchOptimizer


def test_memory_caching_avoids_recomputation():
    """Verify that memory=True caches calls and avoids redundant evaluations."""
    search_space = {"x1": np.arange(10)}

    call_count_mem = 0

    def objective_mem(para):
        nonlocal call_count_mem
        call_count_mem += 1
        return -(para["x1"] ** 2)

    opt_mem = RandomSearchOptimizer(search_space, random_state=42)
    opt_mem.search(objective_mem, n_iter=50, memory=True)

    # Search space only has 10 points; evaluations cannot exceed unique points
    unique_positions = len(set(opt_mem.search_data["x1"]))
    assert call_count_mem == unique_positions
    assert call_count_mem <= 10

    # With memory=False, every single iteration re-evaluates
    call_count_no_mem = 0

    def objective_no_mem(para):
        nonlocal call_count_no_mem
        call_count_no_mem += 1
        return -(para["x1"] ** 2)

    opt_no_mem = RandomSearchOptimizer(search_space, random_state=42)
    opt_no_mem.search(objective_no_mem, n_iter=50, memory=False)

    assert call_count_no_mem == 50
    assert call_count_mem < call_count_no_mem


def test_memory_warm_start_all_cached():
    """Verify that when all positions are pre-cached, objective is never called."""
    search_space = {"x1": np.arange(10)}

    warm_start = pd.DataFrame({
        "x1": np.arange(10),
        "score": np.full(10, 42.0),
    })

    call_count = 0

    def objective(para):
        nonlocal call_count
        call_count += 1
        return -(para["x1"] ** 2)

    opt = RandomSearchOptimizer(search_space, random_state=42)
    opt.search(objective, n_iter=30, memory_warm_start=warm_start)

    # All evaluations should come from warm start; objective must not be called
    assert call_count == 0
    assert len(opt.search_data) == 30
    assert (opt.search_data["score"] == 42.0).all()
    assert opt.best_score == 42.0


def test_memory_warm_start_from_previous_search():
    """Verify that search_data from an earlier run acts as warm start."""
    search_space = {"x1": np.arange(10)}

    def objective(para):
        return -(para["x1"] ** 2)

    opt0 = RandomSearchOptimizer(search_space, random_state=1)
    opt0.search(objective, n_iter=20)

    call_count1 = 0

    def objective1(para):
        nonlocal call_count1
        call_count1 += 1
        return -(para["x1"] ** 2)

    opt1 = RandomSearchOptimizer(search_space, random_state=2)
    opt1.search(objective1, n_iter=20, memory_warm_start=opt0.search_data)

    # Any position previously evaluated in opt0 should be retrieved from cache
    cached_positions = set(opt0.search_data["x1"])
    new_positions = set(opt1.search_data["x1"]) - cached_positions
    assert call_count1 == len(new_positions)
    assert call_count1 <= 20


def test_memory_warm_start_manual():
    """Verify that manual warm-start values are used instead of re-evaluating."""
    search_space = {"x1": np.arange(5)}
    memory_warm_start = pd.DataFrame({
        "x1": [0, 1],
        "score": [99.0, 100.0],
    })

    call_count = 0

    def objective(para):
        nonlocal call_count
        call_count += 1
        return -(para["x1"] ** 2)

    opt = RandomSearchOptimizer(search_space, random_state=42)
    opt.search(objective, n_iter=20, memory_warm_start=memory_warm_start)

    # Check that positions 0 and 1 got scores from memory_warm_start
    for _, row in opt.search_data.iterrows():
        if row["x1"] == 0:
            assert row["score"] == 99.0
        elif row["x1"] == 1:
            assert row["score"] == 100.0


def test_memory_warm_start_wrong_type(caplog):
    """Verify invalid warm start types log warning and search completes cleanly."""
    search_space = {"x1": np.arange(5)}
    opt = RandomSearchOptimizer(search_space, random_state=42)

    opt.search(
        lambda p: -(p["x1"] ** 2),
        n_iter=10,
        memory_warm_start=np.array([1, 2, 3]),
    )

    assert "Memory warm start must be of type pandas.DataFrame" in caplog.text
    assert len(opt.search_data) == 10


def test_memory_warm_start_wrong_search_space(caplog):
    """Verify that warm start with non-matching parameters completes cleanly."""
    search_space = {"x1": np.arange(5)}
    wrong_df = pd.DataFrame({"unrelated_col": [1, 2], "score": [1.0, 2.0]})

    call_count = 0

    def objective(para):
        nonlocal call_count
        call_count += 1
        return -(para["x1"] ** 2)

    opt = RandomSearchOptimizer(search_space, random_state=42)
    opt.search(objective, n_iter=10, memory_warm_start=wrong_df)

    assert "is in search_space but not in memory dataframe" in caplog.text
    assert len(opt.search_data) == 10
    assert call_count == len(set(opt.search_data["x1"]))
