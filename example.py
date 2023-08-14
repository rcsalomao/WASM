from WASM import WASM, generate_RV
import scipy.stats as st
import numpy as np


def rcbeam():
    def g1(Xi, Xd, d):
        Fy, As, Fc, Q = Xi
        width, height = d
        return As * Fy * height - 0.59 * (As * Fy) ** 2 / (Fc * width) - Q

    limit_state_functions = [g1]

    Xi = [
        st.norm(44, 0.105 * 44),
        st.norm(4.08, 0.02 * 4.08),
        st.norm(3.12, 0.14 * 3.12),
        st.norm(2052, 0.12 * 2052),
    ]

    d = [12, 19]

    correlation_matrix = np.eye(len(Xi))

    wasm = WASM(
        Xi,
        correlation_matrix=correlation_matrix,
        n_samples=30000,
        inferior_superior_exponent=5,
        sampling_method="jitter",
        # sampling_method="uniform",
        # sampling_method="antithetic",
        # sampling_method="sobol",
        # sampling_method="halton",
        # sampling_method="lhs",
    )
    # wasm.write_samples('./samples.csv')
    wasm.compute_limit_state_functions(
        limit_state_functions,
        d=d,
        disable_progress_bar=False,
    )

    result = wasm.compute_Beta_Rashki()
    print(result.gXs_results)
    print(result.systems_results)
    print(result.gXs_results.betas)
    print(result.systems_results.betas)


def generate_random_vars():
    rv = generate_RV.normal(42, 2.4)
    print(rv.mean(), rv.std())

    rv = generate_RV.generic(
        st.lognorm, 42, 2.4, fixed_params={"loc": 0}, search_params=["s", "scale"]
    )
    print(rv.mean(), rv.std())

    rv = generate_RV.lognormal(42, 2.4)
    print(rv.mean(), rv.std())

    rv = generate_RV.gumbel(42, 2.4)
    print(rv.mean(), rv.std(), rv.stats(moments="s"), rv.interval(1))

    rv = generate_RV.type_I_smallest_value(42, 2.4)
    print(rv.mean(), rv.std(), rv.stats(moments="s"), rv.interval(1))

    rv = generate_RV.weibull(42, 2.4)
    print(rv.mean(), rv.std(), rv.stats(moments="s"))

    rv = generate_RV.frechet(42, 2.4)
    print(rv.mean(), rv.std(), rv.stats(moments="s"))

    rv = generate_RV.beta(42, 2.4, lower_bound=30, upper_bound=50)
    print(rv.mean(), rv.std(), rv.stats(moments="s"), rv.interval(1))

    rv = generate_RV.uniform(42, 2.4)
    print(rv.mean(), rv.std(), rv.stats(moments="s"), rv.interval(1))

    rv = generate_RV.gamma(42, 2.4, loc=0)
    print(rv.mean(), rv.std(), rv.stats(moments="s"))

    rv = generate_RV.rayleigh(42)
    print(rv.mean(), rv.std(), rv.stats(moments="s"), rv.interval(1))

    rv = generate_RV.maxwell(42)
    print(rv.mean(), rv.std(), rv.stats(moments="s"), rv.interval(1))

    rv = generate_RV.exponential(42)
    print(rv.mean(), rv.std(), rv.stats(moments="s"))


if __name__ == "__main__":
    rcbeam()
    # generate_random_vars()
