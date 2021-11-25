import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from scipy.optimize import root


def errorRV2Param(x, rv, mean, std):
    randomVar = rv(x[0], scale=x[1])
    return [mean-randomVar.mean(), std-randomVar.std()]


def genRV2Param(rv, mean, std, x0=(4.2, 2.4), method='lm', tol=1e-5):
    sol = root(errorRV2Param, x0=x0, args=(rv, mean, std), method=method)
    randomVar = rv(sol.x[0], scale=sol.x[1])
    assert((abs(randomVar.mean()-mean) < tol) and (abs(randomVar.std() - std) < tol))
    return randomVar


def limInfSup(rv, expInfSup):
    pInf = np.power(10.0, -expInfSup)
    pSup = 1.0 - pInf
    return (rv.ppf(pInf), rv.ppf(pSup))


def jitteringSampling(nSamples, bounds):
    nDim = len(bounds)
    nDiv = int(np.power(nSamples, 1.0/nDim))+1

    dimRanges = []
    for i in range(nDim):
        lb, ub = bounds[i]
        dx = (ub-lb)/nDiv
        dimRange = []
        for j in range(nDiv):
            lbx = lb + j*dx
            ubx = lb + (j+1)*dx
            dimRange.append((lbx, ubx))
        dimRanges.append(dimRange)

    products = list(product(range(nDiv), repeat=nDim))

    xs = np.zeros((len(products), nDim))
    for j in range(len(products)):
        for i in range(nDim):
            lbx, ubx = dimRanges[i][products[j][i]]
            xs[j, i] = np.random.uniform(lbx, ubx)
    return xs


class WASM(object):
    def __init__(
            self,
            limitStateFunctions,
            Xi=None,
            ZLbUbi=None,
            correlationMatrix=None,
            nSamples=10000,
            expInfSup=5.0,
            samplingMethod="jitter"):

        self.limitStateFunctions = limitStateFunctions

        if Xi is None:
            Xi = []
        if ZLbUbi is None:
            ZLbUbi = []
        self.Xi = Xi

        self.nX = len(Xi)
        self.nZ = len(ZLbUbi)
        self.nRV = self.nX+self.nZ

        bounds = []
        for rv in Xi:
            bounds.append(limInfSup(rv, expInfSup))
        for rvTuple in ZLbUbi:
            lbLb, _ = limInfSup(rvTuple[0], expInfSup)
            _, ubUb = limInfSup(rvTuple[1], expInfSup)
            bounds.append((lbLb, ubUb))

        nBounds = len(bounds)
        if correlationMatrix is not None:
            assert(correlationMatrix.shape[0] == correlationMatrix.shape[1])
            assert(nBounds == correlationMatrix.shape[0])
            assert(np.all((correlationMatrix-correlationMatrix.T) < 1e-8))
            if samplingMethod == "jitter":
                boundsCorr = [(1e-16, 1.0-1e-16)]*nBounds
                uiCorr = jitteringSampling(nSamples, boundsCorr)
                # Jzy = np.linalg.cholesky(correlationMatrix)
                w, ABarra = np.linalg.eig(correlationMatrix)
                LambdaSquareRooted = np.eye(len(w))*(np.sqrt(w))
                Jzy = np.dot(ABarra, LambdaSquareRooted)
                y = st.norm.ppf(uiCorr)
                z = np.dot(y, Jzy.T)

                probs = st.norm.cdf(z)

                self.ui = np.zeros_like(probs)
                for i in range(nBounds):
                    lb, ub = bounds[i]
                    self.ui[:, i] = st.uniform(lb, ub-lb).ppf(probs[:, i])
            elif samplingMethod == "uniform":
                # Jzy = np.linalg.cholesky(correlationMatrix)
                w, ABarra = np.linalg.eig(correlationMatrix)
                LambdaSquareRooted = np.eye(len(w))*(np.sqrt(w))
                Jzy = np.dot(ABarra, LambdaSquareRooted)
                y = np.random.randn(nSamples, nBounds)
                z = np.dot(y, Jzy.T)

                probs = st.norm.cdf(z)

                self.ui = np.zeros_like(probs)
                for i in range(nBounds):
                    lb, ub = bounds[i]
                    self.ui[:, i] = st.uniform(lb, ub-lb).ppf(probs[:, i])
            else:
                print("Error. Please enter either 'jitter' or 'uniform'.")
                exit(2)
        else:
            if samplingMethod == "jitter":
                self.ui = jitteringSampling(nSamples, bounds)
            elif samplingMethod == "uniform":
                lb = [lo for lo, up in bounds]
                ub = [up for lo, up in bounds]
                self.ui = np.random.uniform(lb, ub, (nSamples, nBounds))
            else:
                print("Error. Please enter either 'jitter' or 'uniform'.")
                exit(2)

        self.actualNSamples = self.ui.shape[0]

        plt.plot(self.ui[:, 0], self.ui[:, 1], 'ko', markersize=1)
        plt.xlim(bounds[0])
        plt.ylim(bounds[1])
        plt.show()

    def writeSamples(self, fileName):
        with open(fileName, "w") as f:
            for i in range(self.nRV-1):
                f.write("RV%s" % (i+1))
                f.write(",")
            f.write("RV%s" % (self.nRV))
            f.write("\n")

            rows, cols = self.ui.shape
            for i in range(rows):
                for j in range(cols-1):
                    f.write(str(self.ui[i, j]))
                    f.write(",")
                f.write(str(self.ui[i, cols-1]))
                f.write("\n")

    def calcLimitStateFunctions(self, d=None, systemFunctions=None, disableProgressBar=False):
        if d is None:
            d = []

        nLimitStateFunctions = len(self.limitStateFunctions)
        GXZdValues = np.zeros((self.actualNSamples, nLimitStateFunctions))
        for i in tqdm(range(self.actualNSamples), disable=disableProgressBar, desc="Evaluating g(X,Z,d)", unit="samples", ascii=True):
            # sleep(0.00025)
            for j in range(nLimitStateFunctions):
                GXZdValues[i, j] = self.limitStateFunctions[j](
                    self.ui[i, 0:self.nX],
                    self.ui[i, self.nX:],
                    d
                )

        if systemFunctions is not None:
            nSystemFunctions = len(systemFunctions)
            systemFunctionsValues = np.zeros((self.actualNSamples, nSystemFunctions))
            for i in range(self.actualNSamples):
                for j in range(nSystemFunctions):
                    systemFunctionsValues[i, j] = systemFunctions[j](GXZdValues[i, :])
            GXZdSystemValues = np.concatenate((GXZdValues, systemFunctionsValues), axis=1)
        else:
            GXZdSystemValues = GXZdValues

        self.Indic = 1.0*(GXZdSystemValues < 0.0)

    def calcBeta_Rashki(self, Zi=None):
        if Zi is None:
            Zi = []
        assert(len(Zi) == self.nZ)

        pdf = np.zeros_like(self.ui)
        randomVars = self.Xi+Zi
        for i in range(self.nRV):
            pdf[:, i] = randomVars[i].pdf(self.ui[:, i])

        wi = np.multiply.reduce(pdf, axis=1)
        wTotal = wi.sum()
        wNormalized = wi/wTotal

        numerator = np.zeros_like(self.Indic)
        indicCols = self.Indic.shape[1]
        for i in range(indicCols):
            numerator[:, i] = self.Indic[:, i]*wNormalized

        pfs = np.zeros(indicCols)
        for i in range(indicCols):
            pfs[i] = min(numerator[:, i].sum(), 1.0-1e-16)

        return (pfs, -st.norm.ppf(pfs))
