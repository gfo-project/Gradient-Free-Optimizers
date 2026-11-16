"""Base helper and shared warm-start data for parameter initialization tests."""

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd


def objective_function(para):
    score = -para["x1"] * para["x1"]
    return score


search_space = {"x1": np.arange(-100, 101, 1)}


def _base_para_test_func(opt_para, optimizer):
    opt = optimizer(search_space, initialize={"vertices": 1}, **opt_para)
    opt.search(
        objective_function,
        n_iter=30,
        memory=False,
        verbosity=False,
    )

    para_key = list(opt_para.keys())[0]
    para_value = getattr(opt, para_key)

    if para_key == "acquisition_function":
        assert para_value == opt_para[para_key]
    else:
        assert para_value is opt_para[para_key]


# Pre-computed warm-start datasets for SMBO parameter initialization tests.
# Created statically to avoid executing redundant RandomSearchOptimizer
# search runs at module import time.
_x1_1 = np.tile(np.arange(-10, 11), 2)[:30]
search_data1 = pd.DataFrame({"x1": _x1_1, "score": -(_x1_1**2).astype(float)})

_x1_2 = np.linspace(-10, 50, 30, dtype=int)
search_data2 = pd.DataFrame({"x1": _x1_2, "score": -(_x1_2**2).astype(float)})

_x1_3 = np.linspace(-50, 10, 30, dtype=int)
search_data3 = pd.DataFrame({"x1": _x1_3, "score": -(_x1_3**2).astype(float)})

_x1_4 = np.arange(30) % 20
search_data4 = pd.DataFrame(
    {
        "x1": _x1_4,
        "score": np.where(np.arange(30) % 2 == 0, 1.0, np.nan),
    }
)

search_data5 = pd.DataFrame(
    {
        "x1": _x1_4,
        "score": np.where(np.arange(30) % 2 == 0, 1.0, -np.inf),
    }
)

search_data6 = pd.DataFrame(
    {
        "x1": _x1_4,
        "score": np.where(np.arange(30) % 2 == 0, 1.0, np.inf),
    }
)
