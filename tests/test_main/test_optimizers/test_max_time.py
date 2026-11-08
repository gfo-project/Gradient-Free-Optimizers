"""Tests for the max_time stopping criterion across optimizers."""

import time

import numpy as np
import pytest

from gradient_free_optimizers import RandomSearchOptimizer
from gradient_free_optimizers._stopping_conditions import (
    StoppingContext,
    TimeExceededCondition,
)

from ._parametrize import optimizers


def objective_function(para):
    return -(para["x1"] ** 2)


search_space = {
    "x1": np.arange(0, 100, 1),
}


@pytest.mark.parametrize(*optimizers)
def test_max_time_0(Optimizer, monkeypatch):
    """Verify max_time stops all optimizers deterministically via simulated time."""
    sim_time = 1000.0

    def fake_time():
        nonlocal sim_time
        sim_time += 0.05
        return sim_time

    monkeypatch.setattr(time, "time", fake_time)

    opt = Optimizer(search_space)
    opt.search(objective_function, n_iter=1000000, max_time=0.5, verbosity=False)

    assert len(opt.search_data) < 1000000
    assert len(opt.search_data) > 0
    assert any(
        c.name == "TimeExceeded" and c.triggered for c in opt.stopper.conditions
    )


@pytest.mark.parametrize(*optimizers)
def test_max_time_1(Optimizer):
    """Verify max_time stops real-clock search loop and triggers TimeExceeded."""
    opt = Optimizer(search_space)
    opt.search(objective_function, n_iter=1000000, max_time=0.05, verbosity=False)

    assert len(opt.search_data) < 1000000
    assert len(opt.search_data) > 0
    assert any(
        c.name == "TimeExceeded" and c.triggered for c in opt.stopper.conditions
    )


def test_max_time_scaling():
    """Verify that a longer max_time allows at least as many iterations."""
    opt_short = RandomSearchOptimizer(search_space, random_state=42)
    opt_short.search(
        objective_function, n_iter=1000000, max_time=0.02, verbosity=False
    )

    opt_long = RandomSearchOptimizer(search_space, random_state=42)
    opt_long.search(
        objective_function, n_iter=1000000, max_time=0.1, verbosity=False
    )

    assert len(opt_long.search_data) >= len(opt_short.search_data)
    assert any(
        c.name == "TimeExceeded" and c.triggered
        for c in opt_short.stopper.conditions
    )
    assert any(
        c.name == "TimeExceeded" and c.triggered
        for c in opt_long.stopper.conditions
    )


def test_time_exceeded_condition_unit():
    """Unit tests for TimeExceededCondition boundary checks and debug info."""
    cond = TimeExceededCondition(max_time=1.0)
    assert cond.name == "TimeExceeded"
    assert not cond.triggered

    ctx_early = StoppingContext(
        iteration=1,
        score_current=0.0,
        score_best=0.0,
        score_history=[0.0],
        start_time=100.0,
        current_time=100.5,
    )
    assert not cond.should_stop(ctx_early)
    assert not cond.triggered

    ctx_late = StoppingContext(
        iteration=2,
        score_current=0.0,
        score_best=0.0,
        score_history=[0.0, 0.0],
        start_time=100.0,
        current_time=101.5,
    )
    assert cond.should_stop(ctx_late)
    assert cond.triggered
    assert "Time limit exceeded" in cond.trigger_reason

    debug = cond.get_debug_info(ctx_late)
    assert debug["condition"] == "TimeExceeded"
    assert debug["max_time"] == 1.0
    assert debug["triggered"] is True

    cond.reset()
    assert not cond.triggered
    assert cond.trigger_reason == ""
