from WASM import WASM, generate_RV_2_param
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


if __name__ == "__main__":
    rcbeam()
