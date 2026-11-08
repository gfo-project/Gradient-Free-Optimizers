"""Tests for optimizer memory warm start across optimizer families."""

import numpy as np
import pytest

from gradient_free_optimizers import GridSearchOptimizer

from ._parametrize import optimizers_non_smbo, optimizers_smbo


@pytest.mark.parametrize(*optimizers_non_smbo)
def test_memory_warm_start_0(Optimizer_non_smbo):
    """Verify non-SMBO optimizers skip evaluations when warm-started on full grid."""
    search_space = {
        "x1": np.arange(0, 10, 1),
    }
    n_iter = 20

    # GridSearch systematically visits all 10 positions in 20 iterations
    opt0 = GridSearchOptimizer(search_space)
    opt0.search(lambda p: -(p["x1"] ** 2), n_iter=n_iter, verbosity=False)

    call_count = 0

    def objective(para):
        nonlocal call_count
        call_count += 1
        return -(para["x1"] ** 2)

    opt1 = Optimizer_non_smbo(search_space)
    opt1.search(
        objective,
        n_iter=n_iter,
        memory_warm_start=opt0.search_data,
        verbosity=False,
    )

    # Every position was cached by opt0; objective must never be re-evaluated
    assert call_count == 0
    assert len(opt1.search_data) == n_iter


@pytest.mark.parametrize(*optimizers_smbo)
def test_memory_warm_start_1(Optimizer_smbo):
    """Verify SMBO optimizers skip evaluations when warm-started on cached points."""
    search_space = {
        "x1": np.arange(0, 1, 1),
    }
    n_iter = 2

    opt0 = GridSearchOptimizer(search_space)
    opt0.search(lambda p: -(p["x1"] ** 2), n_iter=n_iter, verbosity=False)

    call_count = 0

    def objective(para):
        nonlocal call_count
        call_count += 1
        return -(para["x1"] ** 2)

    opt1 = Optimizer_smbo(search_space)
    opt1.search(
        objective,
        n_iter=n_iter,
        memory_warm_start=opt0.search_data,
        verbosity=False,
    )

    # All candidate positions were pre-cached; objective must not be re-evaluated
    assert call_count == 0
    assert len(opt1.search_data) == n_iter


def test_memory_warm_start_partial():
    """Verify that only uncached positions trigger objective evaluation."""
    search_space = {
        "x1": np.arange(0, 10, 1),
    }

    opt0 = GridSearchOptimizer(search_space)
    opt0.search(lambda p: -(p["x1"] ** 2), n_iter=5, verbosity=False)

    call_count = 0

    def objective(para):
        nonlocal call_count
        call_count += 1
        return -(para["x1"] ** 2)

    opt1 = GridSearchOptimizer(search_space)
    opt1.search(
        objective,
        n_iter=10,
        memory_warm_start=opt0.search_data,
        verbosity=False,
    )

    cached_x1 = set(opt0.search_data["x1"])
    new_x1 = set(opt1.search_data["x1"]) - cached_x1
    assert call_count == len(new_x1)
