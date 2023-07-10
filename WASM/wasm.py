import math
import warnings
from collections import namedtuple
from itertools import product

import numpy as np
import scipy.stats as st
from scipy.stats.qmc import Halton, LatinHypercube, Sobol, scale
from tqdm import tqdm

# import matplotlib.pyplot as plt


def serial_system(values):
    return min(values)


def parallel_system(values):
    return max(values)


def calc_system_value(system_definition: dict, values_gX):
    values_system = []
    for k in system_definition:
        values = system_definition[k]
        for val in values:
            if isinstance(val, int):
                values_system.append(values_gX[val])
            if isinstance(val, dict):
                values_system.append(calc_system_value(val, values_gX))
        if k == "serial":
            return serial_system(values_system)
        if k == "parallel":
            return parallel_system(values_system)


def inferior_superior_limits(rv, inf_sup_exponent):
    inf_prob = np.power(10.0, -inf_sup_exponent)
    sup_prob = 1.0 - inf_prob
    return (rv.ppf(inf_prob), rv.ppf(sup_prob))


def uniform_sampling(n_samples, bounds):
    n_bounds = len(bounds)
    lb = [lo for lo, _ in bounds]
    ub = [up for _, up in bounds]
    return np.random.uniform(lb, ub, (n_samples, n_bounds))


def jittering_sampling(n_samples, bounds):
    n_dim = len(bounds)
    n_div = math.ceil(np.power(n_samples, 1.0 / n_dim))
    grid_indexes = list(product(range(n_div), repeat=n_dim))
    ddivs = [(ub - lb) / n_div for lb, ub in bounds]
    n_samples = len(grid_indexes)
    xs = np.zeros((n_samples, n_dim))
    for i in range(n_samples):
        grid_idx_tuple = grid_indexes[i]
        for j in range(n_dim):
            lb, ub = bounds[j]
            xs[i, j] = (
                lb + grid_idx_tuple[j] * ddivs[j] + np.random.uniform(0, ddivs[j])
            )
    return xs


def sobol_sampling(n_samples, bounds):
    n_dim = len(bounds)
    sampler = Sobol(n_dim)
    lb = [lo for lo, _ in bounds]
    ub = [up for _, up in bounds]
    warnings.filterwarnings(
        "ignore",
        message="The balance properties of Sobol' points require n to be a power of 2.",
    )
    return scale(sampler.random(n_samples), lb, ub)


def halton_sampling(n_samples, bounds):
    n_dim = len(bounds)
    sampler = Halton(n_dim)
    lb = [lo for lo, _ in bounds]
    ub = [up for _, up in bounds]
    return scale(sampler.random(n_samples), lb, ub)


def lhc_sampling(n_samples, bounds):
    n_dim = len(bounds)
    sampler = LatinHypercube(n_dim)
    lb = [lo for lo, _ in bounds]
    ub = [up for _, up in bounds]
    return scale(sampler.random(n_samples), lb, ub)


def calc_correlated_samples(ui_uncorr, correlation_matrix, bounds):
    n_bounds = len(bounds)
    # Jzy = np.linalg.cholesky(correlation_matrix)
    w, A_barra = np.linalg.eig(correlation_matrix)
    lambda_square_rooted = np.eye(len(w)) * (np.sqrt(w))
    Jzy = np.dot(A_barra, lambda_square_rooted)
    y = st.norm.ppf(ui_uncorr)
    z = np.dot(y, Jzy.T)
    probs = st.norm.cdf(z)
    ui = np.zeros_like(probs)
    for i in range(n_bounds):
        lb, ub = bounds[i]
        ui[:, i] = st.uniform(lb, ub - lb).ppf(probs[:, i])
    return ui


class WASM(object):
    def __init__(
        self,
        Xi=None,
        Xd_lbub=None,
        correlation_matrix=None,
        n_samples=20000,
        inferior_superior_exponent=6.0,
        sampling_method="jitter",
    ):
        sampling_map = {
            "jitter": jittering_sampling,
            "uniform": uniform_sampling,
            "sobol": sobol_sampling,
            "halton": halton_sampling,
            "lhs": lhc_sampling,
        }
        assert (
            sampling_method in sampling_map.keys()
        ), "Please enter either 'jitter', 'uniform', 'sobol', 'halton' or 'lhs'."
        if Xi is None:
            Xi = []
        if Xd_lbub is None:
            Xd_lbub = []
        self.Xi = Xi
        self.n_Xi = len(Xi)
        self.n_Xd = len(Xd_lbub)
        self.n_rv = self.n_Xi + self.n_Xd
        bounds = []
        for rv in Xi:
            bounds.append(inferior_superior_limits(rv, inferior_superior_exponent))
        for rv_tuple in Xd_lbub:
            lbLb, _ = inferior_superior_limits(rv_tuple[0], inferior_superior_exponent)
            _, ubUb = inferior_superior_limits(rv_tuple[1], inferior_superior_exponent)
            bounds.append((lbLb, ubUb))
        n_bounds = len(bounds)
        if correlation_matrix is not None:
            assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
            assert n_bounds == correlation_matrix.shape[0]
            assert np.all((correlation_matrix - correlation_matrix.T) < 1e-8)
            zero_approx = 1e-15
            bounds_uncorr = [(zero_approx, 1.0 - zero_approx)] * n_bounds
            ui_uncorr = sampling_map[sampling_method](n_samples, bounds_uncorr)
            self.ui = calc_correlated_samples(ui_uncorr, correlation_matrix, bounds)
        else:
            self.ui = sampling_map[sampling_method](n_samples, bounds)
        self.actual_n_samples = self.ui.shape[0]
        # plt.plot(self.ui[:, 0], self.ui[:, 1], "ko", markersize=1)
        # plt.xlim(bounds[0])
        # plt.ylim(bounds[1])
        # plt.show()

    def write_samples(self, filename, d_lbub=None):
        with open(filename, "w") as f:
            for i in range(self.n_rv - 1):
                f.write("RV%s," % (i + 1))
            f.write("RV%s" % (self.n_rv))
            if d_lbub is not None:
                n_d = len(d_lbub)
                n_samples, _ = self.ui.shape
                lb = [lo for lo, _ in d_lbub]
                ub = [up for _, up in d_lbub]
                ui_ds = np.random.uniform(lb, ub, (n_samples, n_d))
                ui = np.concatenate((self.ui, ui_ds), axis=1)
                for i in range(n_d - 1):
                    f.write("d%s," % (i + 1))
                f.write("d%s" % (n_d))
            else:
                ui = self.ui
            f.write("\n")
            rows, cols = ui.shape
            for i in range(rows):
                for j in range(cols - 1):
                    f.write("%s," % (ui[i, j]))
                f.write("%s" % (ui[i, cols - 1]))
                f.write("\n")

    def compute_limit_state_functions(
        self,
        limit_state_functions,
        system_definitions=None,
        d=None,
        disable_progress_bar=False,
    ):
        if d is None:
            d = []
        self.n_limit_state_functions = len(limit_state_functions)
        gX_values = np.zeros((self.actual_n_samples, self.n_limit_state_functions))
        for i in tqdm(
            range(self.actual_n_samples),
            disable=disable_progress_bar,
            desc="Evaluating g(Xi, Xd, d)",
            unit="samples",
            ascii=True,
        ):
            for j in range(self.n_limit_state_functions):
                gX_values[i, j] = limit_state_functions[j](
                    self.ui[i, 0 : self.n_Xi], self.ui[i, self.n_Xi :], d
                )
        if system_definitions is not None:
            n_system_definitions = len(system_definitions)
            systems_values = np.zeros((self.actual_n_samples, n_system_definitions))
            for i in range(self.actual_n_samples):
                for j in range(n_system_definitions):
                    systems_values[i, j] = calc_system_value(
                        system_definitions[j], gX_values[i, :]
                    )
            gX_system_values = np.concatenate((gX_values, systems_values), axis=1)
        else:
            gX_system_values = gX_values
        self.indicadora = 1.0 * (gX_system_values < 0.0)

    def compute_Beta_Rashki(self, Xd=None):
        if Xd is None:
            Xd = []
        assert len(Xd) == self.n_Xd
        pdfs = np.zeros_like(self.ui)
        random_vars = self.Xi + Xd
        for i in range(self.n_rv):
            pdfs[:, i] = random_vars[i].pdf(self.ui[:, i])
        w_i = np.multiply.reduce(pdfs, axis=1)
        w_total = w_i.sum()
        w_failure = np.zeros_like(self.indicadora)
        n_indicadora_cols = self.indicadora.shape[1]
        for i in range(n_indicadora_cols):
            w_failure[:, i] = self.indicadora[:, i] * w_i
        pfs = np.zeros(n_indicadora_cols)
        for i in range(n_indicadora_cols):
            pfs[i] = min(w_failure[:, i].sum() / w_total, 1.0 - 1e-15)
        gXs_results = namedtuple("gXs_results", "pfs, betas")
        systems_results = namedtuple("systems_results", "pfs, betas")
        result = namedtuple("result", "gXs_results, systems_results")
        return result(
            gXs_results(
                pfs[0 : self.n_limit_state_functions],
                -st.norm.ppf(pfs[0 : self.n_limit_state_functions]),
            ),
            systems_results(
                pfs[self.n_limit_state_functions :],
                -st.norm.ppf(pfs[self.n_limit_state_functions :]),
            ),
        )
