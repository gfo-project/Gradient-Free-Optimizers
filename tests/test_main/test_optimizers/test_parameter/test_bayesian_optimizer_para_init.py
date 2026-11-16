"""Tests for BayesianOptimizer parameter initialization."""

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

from gradient_free_optimizers import BayesianOptimizer

from ._base_para_test import (
    _base_para_test_func,
    search_data1,
    search_data2,
    search_data3,
    search_data4,
    search_data5,
    search_data6,
)

search_space = {"x1": np.arange(-10, 11, 1)}


class GPR:
    def __init__(self):
        nu_param = 0.5
        matern = Matern(nu=nu_param)

        self.gpr = GaussianProcessRegressor(
            kernel=matern + RBF() + WhiteKernel(), n_restarts_optimizer=1
        )

    def fit(self, X, y):
        self.gpr.fit(X, y)

    def predict(self, X, return_std=False):
        return self.gpr.predict(X, return_std=return_std)


bayesian_optimizer_para = [
    ({"gpr": GPR()}),
    ({"xi": 0.001}),
    ({"xi": 0.5}),
    ({"xi": 0.9}),
    ({"acquisition_function": "expected_improvement"}),
    ({"acquisition_function": "probability_of_improvement"}),
    ({"acquisition_function": "thompson_sampling"}),
    ({"warm_start_smbo": None}),
    ({"warm_start_smbo": search_data1}),
    ({"warm_start_smbo": search_data2}),
    ({"warm_start_smbo": search_data3}),
    ({"warm_start_smbo": search_data4}),
    ({"warm_start_smbo": search_data5}),
    ({"warm_start_smbo": search_data6}),
    ({"max_sample_size": 10000000}),
    ({"max_sample_size": 10000}),
    ({"max_sample_size": 1000000000}),
    ({"sampling": False}),
    ({"sampling": {"random": 1}}),
    ({"sampling": {"random": 100000000}}),
    ({"rand_rest_p": 0}),
    ({"rand_rest_p": 0.5}),
    ({"rand_rest_p": 1}),
    ({"rand_rest_p": 10}),
    ({"replacement": True}),
    ({"replacement": False}),
]


pytest_wrapper = ("opt_para", bayesian_optimizer_para)


@pytest.mark.parametrize(*pytest_wrapper)
def test_bayesian_para(opt_para):
    _base_para_test_func(opt_para, BayesianOptimizer)


def test_warm_start_0():
    opt = BayesianOptimizer(search_space, warm_start_smbo=search_data1)

    assert len(opt.X_sample) == 30
