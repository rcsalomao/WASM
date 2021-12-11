from WASM import WASM, genRV2Param
import scipy.stats as st
import numpy as np


def g1(xi, xd, d):
    return xi[1]-xi[0]


# def g1(xi, xd, d):
#     return xd[0]-xi[0]


limitStateFunctions = [g1]


def sys1(gs):
    return min(gs)


systemFunctions = [sys1]

Xi = [
    genRV2Param(st.uniform, 5.0, 0.5),
    genRV2Param(st.uniform, 5.0, 0.5)
]
XdLbUb = [
    (genRV2Param(st.norm, 4.9, 0.5), genRV2Param(st.norm, 5.1, 0.5))
]
Xd = [
    genRV2Param(st.norm, 5.0, 0.5)
]
d = [0.3]

corrMatrix = np.eye(len(Xi))
corrMatrix[0, 1] = 0.6
corrMatrix[1, 0] = 0.6

wasm = WASM(limitStateFunctions, Xi, correlationMatrix=corrMatrix, nSamples=500, samplingMethod="jitter")
# wasm = WASM(limitStateFunctions, Xi, correlationMatrix=corrMatrix, nSamples=10, samplingMethod="jitter")
# wasm.writeSamples('./samples.csv')
# wasm = WASM(limitStateFunctions, Xi, nSamples=20000)
wasm.calcLimitStateFunctions(disableProgressBar=False)
print(wasm.calcBeta_Rashki())
