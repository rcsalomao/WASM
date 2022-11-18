from WASM import WASM, generate_RV_2_param
import scipy.stats as st
import numpy as np


def g1(Xi, Xd, d):
    return Xi[1] - Xi[0]


def example1():
    limit_state_functions = [g1]
    system_definitions = [
        {"parallel": [0, 0]},
        {"serial": [0, 0]},
        {"serial": [{"parallel": [0, 0]}, 0]},
    ]

    Xi = [
        generate_RV_2_param(st.uniform, 5.0, 0.5),
        generate_RV_2_param(st.uniform, 5.0, 0.5),
    ]
    # Xd_lbub = [
    #     (generate_RV_2_param(st.norm, 4.9, 0.5), generate_RV_2_param(st.norm, 5.1, 0.5))
    # ]
    # Xd = [generate_RV_2_param(st.norm, 5.0, 0.5)]
    # d = [0.3]

    corr_matrix = np.eye(len(Xi))
    corr_matrix[0, 1] = 0.6
    corr_matrix[1, 0] = 0.6

    wasm = WASM(
        Xi,
        correlation_matrix=corr_matrix,
        n_samples=500,
        sampling_method="jitter",
        # sampling_method="uniform",
        # sampling_method="sobol",
        # sampling_method="halton",
        # sampling_method="lhs",
    )
    # wasm.write_samples('./samples.csv')
    wasm.compute_limit_state_functions(
        limit_state_functions,
        system_definitions=system_definitions,
        disable_progress_bar=False,
    )

    result = wasm.compute_Beta_Rashki()
    print(result.gXs_results)
    print(result.systems_results)
    print(result.gXs_results.betas)
    print(result.systems_results.betas)


if __name__ == "__main__":
    example1()
