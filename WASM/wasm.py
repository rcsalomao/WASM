from collections import namedtuple
import numpy as np
import scipy.stats as st
from tqdm import tqdm
from itertools import product
from scipy.optimize import root

# import matplotlib.pyplot as plt


def serial_system(*gX_values):
    return min(gX_values)


def parallel_system(*gX_values):
    return max(gX_values)


def error_RV_2_param(x, rv, mean, std):
    random_var = rv(x[0], scale=x[1])
    return [mean - random_var.mean(), std - random_var.std()]


def generate_RV_2_param(rv, mean, std, x0=(4.2, 2.4), method="lm", tol=1e-5):
    sol = root(error_RV_2_param, x0=x0, args=(rv, mean, std), method=method)
    random_var = rv(sol.x[0], scale=sol.x[1])
    assert (abs(random_var.mean() - mean) < tol) and (abs(random_var.std() - std) < tol)
    return random_var


def inferior_superior_limits(rv, inf_sup_exponent):
    inf_prob = np.power(10.0, -inf_sup_exponent)
    sup_prob = 1.0 - inf_prob
    return (rv.ppf(inf_prob), rv.ppf(sup_prob))


def jittering_sampling(n_samples, bounds):
    n_dim = len(bounds)
    n_div = int(np.power(n_samples, 1.0 / n_dim)) + 1
    dim_ranges = []
    for i in range(n_dim):
        lb, ub = bounds[i]
        dx = (ub - lb) / n_div
        dim_range = []
        for j in range(n_div):
            lbx = lb + j * dx
            ubx = lb + (j + 1) * dx
            dim_range.append((lbx, ubx))
        dim_ranges.append(dim_range)
    products = list(product(range(n_div), repeat=n_dim))
    xs = np.zeros((len(products), n_dim))
    for j in range(len(products)):
        for i in range(n_dim):
            lbx, ubx = dim_ranges[i][products[j][i]]
            xs[j, i] = np.random.uniform(lbx, ubx)
    return xs


class WASM(object):
    def __init__(
        self,
        limit_state_functions,
        Xi=None,
        Xd_lbub=None,
        correlation_matrix=None,
        n_samples=10000,
        inferior_superior_exponent=5.0,
        sampling_method="jitter",
    ):
        self.limit_state_functions = limit_state_functions
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
            if sampling_method == "jitter":
                bounds_corr = [(1e-16, 1.0 - 1e-16)] * n_bounds
                ui_corr = jittering_sampling(n_samples, bounds_corr)
                # Jzy = np.linalg.cholesky(correlation_matrix)
                w, A_barra = np.linalg.eig(correlation_matrix)
                lambda_square_rooted = np.eye(len(w)) * (np.sqrt(w))
                Jzy = np.dot(A_barra, lambda_square_rooted)
                y = st.norm.ppf(ui_corr)
                z = np.dot(y, Jzy.T)
                probs = st.norm.cdf(z)
                self.ui = np.zeros_like(probs)
                for i in range(n_bounds):
                    lb, ub = bounds[i]
                    self.ui[:, i] = st.uniform(lb, ub - lb).ppf(probs[:, i])
            elif sampling_method == "uniform":
                # Jzy = np.linalg.cholesky(correlation_matrix)
                w, A_barra = np.linalg.eig(correlation_matrix)
                lambda_square_rooted = np.eye(len(w)) * (np.sqrt(w))
                Jzy = np.dot(A_barra, lambda_square_rooted)
                y = np.random.randn(n_samples, n_bounds)
                z = np.dot(y, Jzy.T)
                probs = st.norm.cdf(z)
                self.ui = np.zeros_like(probs)
                for i in range(n_bounds):
                    lb, ub = bounds[i]
                    self.ui[:, i] = st.uniform(lb, ub - lb).ppf(probs[:, i])
            else:
                print(">>>Error. Please enter either 'jitter' or 'uniform'.")
                exit(2)
        else:
            if sampling_method == "jitter":
                self.ui = jittering_sampling(n_samples, bounds)
            elif sampling_method == "uniform":
                lb = [lo for lo, _ in bounds]
                ub = [up for _, up in bounds]
                self.ui = np.random.uniform(lb, ub, (n_samples, n_bounds))
            else:
                print(">>>Error. Please enter either 'jitter' or 'uniform'.")
                exit(2)
        self.actual_n_samples = self.ui.shape[0]
        # plt.plot(self.ui[:, 0], self.ui[:, 1], 'ko', markersize=1)
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
        self, d=None, system_functions=None, disable_progress_bar=False
    ):
        if d is None:
            d = []
        n_limit_state_functions = len(self.limit_state_functions)
        gX_values = np.zeros((self.actual_n_samples, n_limit_state_functions))
        for i in tqdm(
            range(self.actual_n_samples),
            disable=disable_progress_bar,
            desc="Evaluating g(Xi, Xd, d)",
            unit="samples",
            ascii=True,
        ):
            for j in range(n_limit_state_functions):
                gX_values[i, j] = self.limit_state_functions[j](
                    self.ui[i, 0 : self.n_Xi], self.ui[i, self.n_Xi :], d
                )
        if system_functions is not None:
            n_system_functions = len(system_functions)
            system_functions_values = np.zeros(
                (self.actual_n_samples, n_system_functions)
            )
            for i in range(self.actual_n_samples):
                for j in range(n_system_functions):
                    system_functions_values[i, j] = system_functions[j](gX_values[i, :])
            gX_system_values = np.concatenate(
                (gX_values, system_functions_values), axis=1
            )
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
        wi = np.multiply.reduce(pdfs, axis=1)
        w_total = wi.sum()
        w_normalized = wi / w_total
        w_failure = np.zeros_like(self.indicadora)
        indicadora_cols = self.indicadora.shape[1]
        for i in range(indicadora_cols):
            w_failure[:, i] = self.indicadora[:, i] * w_normalized
        pfs = np.zeros(indicadora_cols)
        for i in range(indicadora_cols):
            pfs[i] = min(w_failure[:, i].sum(), 1.0 - 1e-16)
        n_limit_state_functions = len(self.limit_state_functions)
        gX_pf_beta = namedtuple("gx", "pf, beta")
        system_pf_beta = namedtuple("system", "pf, beta")
        return (
            gX_pf_beta(
                pfs[0:n_limit_state_functions],
                -st.norm.ppf(pfs[0:n_limit_state_functions]),
            ),
            system_pf_beta(
                pfs[n_limit_state_functions:],
                -st.norm.ppf(pfs[n_limit_state_functions:]),
            ),
        )
