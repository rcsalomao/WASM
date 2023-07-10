from functools import partial

from scipy.optimize import root


def error_RV(x, rv, mean, std, fixed_params, search_params):
    sp = {s: v for (s, v) in zip(search_params, x)}
    random_var = partial(rv, **fixed_params)(**sp)
    return [mean - random_var.mean(), std - random_var.std()]


def generate_RV(
    rv,
    mean,
    std,
    fixed_params: dict[str:float],
    search_params: list[str],
    x0=None,
    method="lm",
    tol=1e-4,
):
    if x0 is None:
        x0 = [4.2, 2.4][: len(search_params)]
    assert len(x0) == len(search_params)
    sol = root(
        error_RV,
        x0=x0,
        args=(rv, mean, std, fixed_params, search_params),
        method=method,
    )
    sp = {s: v for (s, v) in zip(search_params, sol.x)}
    random_var = partial(rv, **fixed_params)(**sp)
    assert (abs(random_var.mean() - mean) < tol) and (abs(random_var.std() - std) < tol)
    return random_var
