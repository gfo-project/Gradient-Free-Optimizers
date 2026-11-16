"""Tests for LipschitzOptimizer parameter initialization."""

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pytest

from gradient_free_optimizers import LipschitzOptimizer

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


lipschitz_para = [
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
    ({"replacement": True}),
    ({"replacement": False}),
]


pytest_wrapper = ("opt_para", lipschitz_para)


@pytest.mark.parametrize(*pytest_wrapper)
def test_lipschitz_para(opt_para):
    _base_para_test_func(opt_para, LipschitzOptimizer)


def test_warm_start_0():
    opt = LipschitzOptimizer(search_space, warm_start_smbo=search_data1)

    assert len(opt.X_sample) == 30
