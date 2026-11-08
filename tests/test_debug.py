"""Check if there are any debug prints left in optimizer code."""

import numpy as np
import pytest

from gradient_free_optimizers import (
    BayesianOptimizer,
    EvolutionStrategyOptimizer,
    ForestOptimizer,
    HillClimbingOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    RandomAnnealingOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomSearchOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    StochasticHillClimbingOptimizer,
    TreeStructuredParzenEstimators,
)

optimizer_list = [
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    RandomSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
    SimulatedAnnealingOptimizer,
    ParallelTemperingOptimizer,
    ParticleSwarmOptimizer,
    EvolutionStrategyOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
]


def objective_function(para):
    """Simple quadratic objective function."""
    score = -para["x1"] * para["x1"]
    return score


search_space = {
    "x1": np.arange(0, 5, 1),
}


@pytest.mark.parametrize("optimizer", optimizer_list)
def test_no_debug_prints_left_in_code(optimizer, capsys=None):
    """Verify optimizer runs cleanly with verbosity=False and leaves no debug prints."""
    opt0 = optimizer(search_space)
    opt0.search(objective_function, n_iter=15, verbosity=False, memory=False)

    opt1 = optimizer(search_space)
    opt1.search(objective_function, n_iter=15, verbosity=False)

    opt2 = optimizer(search_space)
    opt2.search(
        objective_function,
        n_iter=15,
        verbosity=False,
        memory_warm_start=opt1.search_data,
    )

    opt3 = optimizer(search_space, initialize={"warm_start": [{"x1": 1}]})
    opt3.search(
        objective_function,
        n_iter=15,
        verbosity=False,
    )

    if capsys is not None:
        captured = capsys.readouterr()
        assert captured.out == ""
