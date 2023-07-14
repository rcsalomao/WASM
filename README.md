# Weighted Average Simulation Method - WASM

This library represents the implementation of the simulation method presented in [1] to approximate the probability of failure of a system.
As such, it is possible to numerically solve a reliability problem by Monte Carlo, taking in account diverse types of correlated random variables, analytical and numerical limit state functions and system definitions.

## Installation

Just copy/git clone the WASM repository into your project root or someplace discoverable by your python environment (PYTHONPATH, sys.path, site-packages, etc).
Creating a symbolic link on those places pointing to the WASM repo is also another option.

## Requirements

This repo makes use of the following libraries:

- numpy
- scipy
- tqdm

Please install them beforehand on your python environment.

## Usage

### Random Variables

The random variables to supply as input must be random variable objects from scipy.stats module.
In this manner, the user is responsible for the random variables specific parameters and fitting process.
The module `generate_RV` is also supplied to help finding a random variable with desired mean $\mu$ and/or standard deviation $\sigma$.
This module contains random variable generator functions, that tries to find a desired random distribution with the desired parameters.

The implemented generator functions are:
```python
generate_RV.generic(...)
generate_RV.normal(...)
generate_RV.lognormal(...)
generate_RV.gumbel(...)
generate_RV.type_I_largest_value(...)
generate_RV.type_I_smallest_value(...)
generate_RV.weibull(...)
generate_RV.frechet(...)
generate_RV.beta(...)
generate_RV.gamma(...)
generate_RV.uniform(...)
generate_RV.rayleigh(...)
generate_RV.maxwell(...)
generate_RV.exponential(...)
```

The input description for the `generic(...)` function is:
```python
generate_RV.generic(rv, mean, std: float | None, fixed_params: dict[str:float], search_params: list[str], x0=None, method="lm", tol=1e-4):

# rv: Random variable object from scipy.stats.

# mean: Mean value of the resulting random variable.

# std: Standard deviation value or a None value, according to the random distribution.

# fixed_params: Dictionary defining the values of the random variable parameters that will be adopted throught the root finding process to fit the scipy.stats random variable object.

# search_params: List of parameter names of the scipy.stats random variable object that will be fitted by the root finding algorithm, resulting in an object with the desired mean and/or standard deviation value(s).

# x0: A initial values list of [mean] or [mean, standard deviation]. Those are used only as starting value for the root finding algorithm. They can, and should be, changed if there is some difficulty on convergence.

# method: Method for root finding (see "root" function from scipy.optimize).

# tol: Tolerance value. After the root function invocation, there is a verification to check if the mean and standard deviation of the resulting random variable object has the desired mean and std. That is, abs(rv.mean() - mean) < tol and abs(rv.std() - std) < tol.
```

The `lognormal`, `gumbel`, `weibull`, `beta`, `exponential` and other random distribution generator functions have, in general, the following signatures:
```python
generate_RV.lognormal( mean, std, x0: list[float] | None = None, method="lm", tol=1e-4):
generate_RV.gumbel(mean, std, x0: list[float] | None = None, method="lm", tol=1e-4):
generate_RV.weibull(mean, std, loc: float = 0, x0: list[float] | None = None, method="lm", tol=1e-4):
generate_RV.beta(mean, std, lower_bound=0, upper_bound=1, x0: list[float] | None = None, method="lm", tol=1e-4):
generate_RV.exponential( mean, loc: float = 0, x0: list[float] | None = None, method="lm", tol=1e-4):
```

#### Examples

```python
X1 = scipy.stats.norm()
X2 = scipy.stats.norm(9.9, 0.8)
X3 = generate_RV.generic(
    scipy.stats.gumbel_r,
    5.8,
    1.9,
    fixed_params={},
    search_params=["loc", "scale"]
)
X4 = generate_RV.weibull(4.2, 1.4)

Xi = [X1, X2, X3, X4]

Xd = [
    generate_RV.generic(
        scipy.stats.lognorm,
        5.0,
        1.2,
        fixed_params={"loc": 0},
        search_params=["s", "scale"]
    ),
    generate_RV.uniform(6.3, 1.2)
]

Xd_lbub = [
    (
        generate_RV.generic(
            scipy.stats.weibull_min,
            4.2,
            0.25*4.2,
            {"loc": 0},
            ["c", "scale"]
        ),
        generate_RV.weibull(9.1, 0.25*9.1)
    )
]
```

### Limit State Functions

Limit state functions define the modes of failure of interest.
Those are a function of the random and deterministic variables of the problem.
The vector of random variables is denoted (in index notation) as $X_m$, whereas the vector of deterministic variables is described as $d_n$.
A distinction is made for the random variables, resulting in 2 groups of random variables:

- $Xi_m$ is a vector of *independent* random variables.
- $Xd_m$ is a vector of *design* random variables.

$d_n$ is also described as a vector of *design* deterministic variables.

The *design* variables, both random and deterministic, are employed in the context of numerical optimization.
The use of this library for solving optimization problems with reliability constraints won't be discussed in this repo.
For that case, please see the [RBDO_examples](https://github.com/rcsalomao/RBDO_examples) repo.

With that said, the description of a limit state function $g$ is given as:
```python
g(Xi, Xd, d):

# Xi: A list of independent random variable objects.

# Xd: A list of design random variable objects.

# d: A list of design deterministic variable.
```

The limit state function can be defined as analytical or numerical (by wrapping the numerical method function).
As input, WASM must be supplied with a list of limit state functions (even if there is only one).

#### Examples

```python
def g1(Xi, Xd, d):
    X1, X2, X3 = Xi
    return X1**3 + X2**3 - X3/10.0

def g2(Xi, Xd, d):
    X1, X2, X3 = Xi
    d1, d2 = d
    return X1*X3*d1 - X2*d2

def g3(Xi, Xd, d):
    X1, X2, X3 = Xi
    any_other_variable = 42
    max_threshold_value = 3.3
    return max_threshold_value - my_awesome_FEM_model_result(X1, X2, X3, any_other_variable)

limit_state_functions = [g1, g2, g3]
```

### System Definitions

A system definition is expressed as a dictionary with key either "serial" or "parallel" and a value representing a list made of integers and/or subsystem definition dictionaries.
The integer values map out to the index position of a limit state function contained in the limit state functions input list.
This way, it's possible to represent "serial", "parallel" and mixed/hybrid systems.

#### Examples

```python
system1 = {"serial": [0, 1, 2]}
system2 = {"parallel": range(len(limit_state_functions))}
system3 = {"parallel": [
    {"serial": [0, 1]},
    2
]}

system_definitions = [system1, system2, system3]
```

### WASM Interface

The main numerical method is contained in the WASM object.
It's methods compose the interface available to the user to solve the reliability problem.
The following methods are exposed to the user:
```python
WASM(...)
WASM.write_samples(...)
WASM.compute_limit_state_functions(...)
WASM.compute_beta_Rashki(...)
```

The constructor has the following signature and accepts the input parameters of:
```python
WASM(Xi=None, Xd_lbub=None, correlation_matrix=None, n_samples=10000, inferior_superior_exponent=5, sampling_method="jitter"):

# Xi: A list of independent random variable objects.

# Xd_lbub: A list of tuples of design random variable objects that define the lower bound and upper bound for each one of the design random variable problem dimension.

# correlation_matrix: A correlation matrix respective to both independent and design random variables. This matrix must be symmetrical and square. If None, it's assumed that all random variables are completely uncorrelated.

# n_samples: Number of samples (or at least approximately) to solve the reliability problem.

# inferior_superior_exponent: In the WASM method it is necessary to estipulate minimum and maximum numerical bounds for the sampling process. In this implementation, as defined in [1], those bounds are obtained by the inverse of the random distribution CDF with inferior and superior probabilities. Those probabilities are (1e-n) and (1.0 - 1e-n), with "n" being the exponent with default value of 6.0.

# sampling_method: Sampling method to be used for the sampling process. The possible values are "jitter", "uniform", "sobol", "halton" or "lhs". "sobol", "halton" and "lhs" samplings are done with the routines found on scipy.stats.qmc while "jitter" follows the method described in [2].
```

When invoking the constructor, only the sampling process is realized.
The evaluation of the limit state functions are made by calling the appropriate method.
That way it's possible to use the same set of samples to reevaluate the limit state functions if necessary.
This proves crucial in the context of numerical optimization.

If it's necessary to save the coordinates of the generated samples, the following method can be used:
```python
WASM.write_samples(filename, d_lbub=None):

# filename: Name of the output file with the samples coordinates.

# d_lbub: A list of tuples defining lower and upper bounds for each one of the deterministic design variables. Those will be uniformly sampled on said interval.
```

The following method is responsible to evaluate the limit state functions on each one of the sampled points.
After the evaluation, it is internally stored a vector that indicates the failure or survival of the limit state functions and systems on each point.

```python
WASM.compute_limit_state_functions(limit_state_functions, system_definitions=None, d=None, disable_progress_bar=False):

# limit_state_functions: A list consisting of functions objects defining each one of the modes of failure of the problem.

# system_definitions: A list of system definitions.

# d: A list of design deterministic variable.

# disable_progress_bar: A boolean value to suppress the terminal progress bar for the limit state evaluations.
```

Finally, it is possible to compute the failure's probability of the defined problem.
Having a separate method is interesting in the context of optimization where it is necessary to change the atributes if the *design* random variables to find the optimum parameters for the objective function.

```python
WASM.compute_beta_Rashki(Xd=None):

# Xd: A list of design random variable objects.
```

### Result object

The result object is a namedtuple of type `result(gX_results, system_results)`.
With `gX_results` and `system_results` of types `gX_results(pfs, betas)` and `system_results(pfs, betas)` respectively.
Both `pfs` and `betas` are lists of failure probabilities and $\beta$ indexes of the limit state functions and system definitions.
The positional indexes of those values correspond to the same indexes as the input variables on the `WASM.compute_limit_state_functions(...)` method.

### Numerical example

The following example is the second numerical problem on [1].
It's the reliability analysis of a reinforced concrete beam, with failure mode given by the limit state function $g_1$.
The problem definition and resolution by the WASM library implemented is:
```python
def g1(Xi, Xd, d):
    Fy, As, Fc, Q = Xi
    width, height = d
    return As * Fy * height - 0.59 * (As * Fy) ** 2 / (Fc * width) - Q

limit_state_functions = [g1]

Xi = [
    scipy.stats.norm(44, 0.105 * 44),
    scipy.stats.norm(4.08, 0.02 * 4.08),
    scipy.stats.norm(3.12, 0.14 * 3.12),
    scipy.stats.norm(2052, 0.12 * 2052),
]

d = [12, 19]

correlation_matrix = np.eye(len(Xi))  # It's not necessary to define the correlation matrix for this problem. I have defined just to show the API.

wasm = WASM(
    Xi,
    correlation_matrix=correlation_matrix,
    n_samples=20000,
    inferior_superior_exponent=5,
    sampling_method="jitter",
)

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
```

This same problem is described in the `example.py` file, together with, possibly, other problems.

## References

[1]: Rashki M, Miri M, Moghaddam M. A new efficient simulation method to approximate the probability of failure and most probable point. Structural Safety. 2012;39:22–29.

[2]: Paulsinger F, Steinerberger S. On the discrepancy of jittered sampling. Journal of Complexity. 2016;33:199–216.
