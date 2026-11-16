"""Tests for ForestOptimizer parameter initialization."""

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pytest

from gradient_free_optimizers import ForestOptimizer

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


dto_para = [
    ({"tree_regressor": "random_forest"}),
    ({"tree_regressor": "extra_tree"}),
    ({"tree_regressor": "gradient_boost"}),
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


pytest_wrapper = ("opt_para", dto_para)


@pytest.mark.parametrize(*pytest_wrapper)
def test_dto_para(opt_para):
    _base_para_test_func(opt_para, ForestOptimizer)


def test_warm_start_0():
    opt = ForestOptimizer(search_space, warm_start_smbo=search_data1)

    assert len(opt.X_sample) == 30
