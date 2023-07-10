from functools import partial

import scipy.stats as st
from scipy.optimize import root

from .internal import _error_RV


def generic(
    rv,
    mean,
    std: float | None,
    fixed_params: dict[str:float],
    search_params: list[str],
    x0: list[float] | None = None,
    method="lm",
    tol=1e-4,
):
    if std is None:
        n_x = 1
    else:
        n_x = 2
    if x0 is None:
        x0 = [4.2, 2.4][:n_x]
    assert len(x0) == len(search_params) == n_x
    sol = root(
        _error_RV,
        x0=x0,
        args=(rv, mean, std, fixed_params, search_params),
        method=method,
    )
    sp = {s: v for (s, v) in zip(search_params, sol.x)}
    random_var = partial(rv, **fixed_params)(**sp)
    if n_x == 1:
        assert abs(mean - random_var.mean()) < tol
    else:
        assert (abs(mean - random_var.mean()) < tol) and (
            abs(std - random_var.std()) < tol
        )
    return random_var


def normal(mean, std):
    return st.norm(mean, std)


def lognormal(mean, std):
    return generic(
        st.lognorm, mean, std, fixed_params={"loc": 0}, search_params=["s", "scale"]
    )


def gumbel(mean, std):
    return generic(
        st.gumbel_r, mean, std, fixed_params={}, search_params=["loc", "scale"]
    )


def weibull(mean, std):
    return generic(
        st.weibull_min, mean, std, fixed_params={"loc": 0}, search_params=["c", "scale"]
    )


def frechet(mean, std):
    return generic(
        st.invweibull, mean, std, fixed_params={"loc": 0}, search_params=["c", "scale"]
    )


def beta(mean, std, lower_bound=0, upper_bound=1):
    return generic(
        st.beta,
        mean,
        std,
        fixed_params={"loc": lower_bound, "scale": upper_bound - lower_bound},
        search_params=["a", "b"],
    )


def gamma(mean, std):
    return generic(
        st.gamma, mean, std, fixed_params={"loc": 0}, search_params=["a", "scale"]
    )


def uniform(mean, std):
    return generic(
        st.uniform, mean, std, fixed_params={}, search_params=["loc", "scale"]
    )


def rayleigh(mean):
    return generic(
        st.rayleigh, mean, None, fixed_params={"loc": 0}, search_params=["scale"]
    )


def maxwell(mean):
    return generic(
        st.maxwell, mean, None, fixed_params={"loc": 0}, search_params=["scale"]
    )


def exponential(mean):
    return generic(
        st.expon, mean, None, fixed_params={"loc": 0}, search_params=["scale"]
    )
