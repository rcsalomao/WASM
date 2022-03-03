from WASM import WASM, generate_RV_2_param
import scipy.stats as st
import numpy as np


def g1(xi, xd, d):
    return xi[1]-xi[0]


# def g1(xi, xd, d):
#     return xd[0]-xi[0]


limit_state_functions = [g1]


def sys1(gs):
    return min(gs)


system_functions = [sys1]

Xi = [
    generate_RV_2_param(st.uniform, 5.0, 0.5),
    generate_RV_2_param(st.uniform, 5.0, 0.5)
]
Xd_lbub = [
    (generate_RV_2_param(st.norm, 4.9, 0.5), generate_RV_2_param(st.norm, 5.1, 0.5))
]
Xd = [
    generate_RV_2_param(st.norm, 5.0, 0.5)
]
d = [0.3]

corr_matrix = np.eye(len(Xi))
corr_matrix[0, 1] = 0.6
corr_matrix[1, 0] = 0.6

wasm = WASM(limit_state_functions, Xi, correlation_matrix=corr_matrix, n_samples=500, sampling_method="jitter")
# wasm = WASM(limit_state_functions, Xi, correlation_matrix=corr_matrix, n_samples=10, sampling_method="uniform")
# wasm.write_samples('./samples.csv')
# wasm = WASM(limit_state_functions, Xi, n_samples=20000)
wasm.compute_limit_state_functions(disable_progress_bar=False)
print(wasm.compute_Beta_Rashki())
