# `folx` - Forward Laplacian for JAX

This submodule implements the forward laplacian from https://arxiv.org/abs/2307.08214. It is implemented as a [custom interpreter for Jaxprs](https://jax.readthedocs.io/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html).

## Install

Either clone repo and install locally via
```bash
poetry install
```
or
```bash
pip install .
```
or install via `pip` package manager via
```bash
pip install folx
```

## Example
### Dense example
For simple usage, one can decorate any function with `forward_laplacian`.
```python
import numpy as np
from folx import forward_laplacian

def f(x):
    return (x**2).sum()

fwd_f = forward_laplacian(f)
result = fwd_f(np.arange(3, dtype=float))
result.x # f(x) 3
result.jacobian.dense_array # J_f(x) [0, 2, 4]
result.laplacian # tr(H_f(x)) 6
```
### Sparsity example
A big feature of `folx` is to automatically work with sparse jacobians to accelerate computations. Note that the results are still **exact**. To enable this feature simply supply a maximum sparsity threshold. Compile times may increase significantly as tracing the sparsity patterns of the jacobians is a lengthy process. Here is an example with an MLP operating indepdently on individual node features.
```python
import folx
import jax
import jax.numpy as jnp
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        for _ in range(10):
            x = nn.Dense(100)(x)
            x = nn.silu(x)
        return nn.Dense(1)(x).sum()

mlp = MLP()
x = jnp.ones((20, 100, 4))
params = mlp.init(jax.random.PRNGKey(0), x)
def fwd(x):
    return mlp.apply(params, x)

# Traditional loop implementation
lapl = jax.jit(jax.vmap(folx.LoopLaplacianOperator()(fwd)))
%time jax.block_until_ready(lapl(x)) # Wall time: 1.42 s
%timeit jax.block_until_ready(lapl(x)) # 224 ms ± 54 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

# Forward laplacian without sparsity
lapl = jax.jit(jax.vmap(folx.ForwardLaplacianOperator(0)(fwd)))
%time jax.block_until_ready(lapl(x)) # Wall time: 2.66 s
%timeit jax.block_until_ready(lapl(x)) # 48.7 ms ± 42.1 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# Forward laplacian with sparsity
lapl = jax.jit(jax.vmap(folx.ForwardLaplacianOperator(4)(fwd)))
%time jax.block_until_ready(lapl(x)) # Wall time: 5.05 s
%timeit jax.block_until_ready(lapl(x)) # 2.59 ms ± 15.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```
For electronic wave function like FermiNet or PsiFormer, `sparsity_threshold=6` is a recommended value. But, tuning this hyperparameter may accelerate computations.

## Introduction
To avoid custom wrappers for all of JAX's commands, the forward laplacian is implemented as custom interpreter for Jaxpr.
This means if you have a function
```python
class Fn(Protocol):
    def __call__(self, *args: PyTree[Array]) -> PyTree[Array]:
        ...
```
the resulting function will have the signature:
```python
class LaplacianFn(Protocol):
    def __call__(self, *args: PyTree[Array]) -> PyTree[FwdLaplArray]:
        ...
```
where `FwdLaplArray` is a triplet of
```python
FwdLaplArray.x # jax.Array f(x) f(x).shape
FwdLaplArray.jacobian # FwdJacobian J_f(x)
FwdLaplArray.laplacian # jax.Array tr(H_f(x)) f(x).shape
```
The jacobian is implemented by a custom class as the forward laplacian supports automatic sparsity. To get the full jacobian:
```python
FwdLaplArray.jacobian.dense_array # jax.Array (*f(x).shape, x.size)
```

## Implementation idea
The idea is to rely on the original function and autodifferentiation to propagate `FwdLaplArray` forward instead of the regular `jax.Array`. The rules for updating `FwdLaplArray` are described by the pseudocode:
```python
x # FwdLaplArray
y = FwdLaplArray(
    x=f(x.x),
    jacobian=jvp(f, (x.x,), (x.jacobian)),
    laplacian=tr_vhv(f, x.jacobian) + jvp(f, (x.x,), (x.laplacian,))
)
# tr_vhv is tr(J_f H_f J_f^T)
```

## Implementation

When you call the function returned by `forward_laplacian(fn)`, we first use `jax.make_jaxpr` to obtain the jaxpr for `fn`.
But instead of using the [standard evaluation pipeline](https://github.com/google/jax/blob/776baba0a3fca15a909cb7d108eea830cbe3fc1d/jax/_src/core.py#L436), we use a custom interpreter that replaces all operations to propate `FwdLaplArray` forward instead of regular `jax.Array`.

### Package structure
The general structure of the package is
* `interpreter.py` contains the evaluation of jaxpr and exported function decorator.
* `wrapper.py` contains subfunction decorator that maps a function that takes `jax.Array`s to a function that accepts `FwdLaplArray`s instead.
* `wrapped_functions.py` contains a registry of predefined functions as well as utility functions to add new functions to the registry.
* `jvp.py` contains logic for jacobian vector products.
* `hessian.py` contains logic for tr(JHJ^T).
* `custom_hessian.py` contains special treatment logic for tr(JHJ^T).
* `api.py` contains general interfaces shared in the package.
* `operators.py` contains a forward laplacian operator as well as alternatives.
* `utils.py` contains several small utility functions.
* `tree_utils.py` contains several utility functions for PyTrees.
* `vmap.py` contains a batched vmap implementation to reduce memory usage by going through a batch sequentially in chunks.


### Function Annotations
There is a default interpreter that will simply apply the rules outlined above but if additional information about a function is available, e.g., that it applies elementwise like `jnp.tanh`, we can do better.
These additional annotations are available in `wrapped_functions.py`'s `_LAPLACE_FN_REGISTRY`.
Specifically, to augment a function `fn` to accept `FwdLaplArray` instead of regular `jax.Array`, we wrap it with `wrap_forward_laplacian` from `fwd_laplacian.py`:
```python
wrap_forward_laplacian(jnp.tanh, in_axes=())
```
In this case, we annotate the function to be applied elementwise, i.e., `()` indicates that none of the axes are relevant for the function.

If we know nothing about which axes might be essential, one must pass `None` (the default value) to mark all axes as imporatnt, e.g.,
```python
wrap_forward_laplacian(jnp.sum, in_axes=None, flags=FunctionFlags.LINEAR)
```
However, in this case we know that a summation is a linear operation. This information is useful for fast hessian computations.

If you want rules to a function and add it to the registry you can do the following
```python
import jax
from folx import register_function, wrap_forward_laplacian

register_function(jax.lax.cos_p, wrap_forward_laplacian(f, in_axes=()))
# Now the tracer is aware that the cosine function is applied elementwise.
```
We can do even more by defining custom rules:
```python
import jax
from folx import register_function, wrap_forward_laplacian

# the jit is important
@jax.jit
def f(x):
    return x

# define a custom jacobian hessian jacobian product rule
def custom_jac_hessian_jac(args, extra_args, merge, materialize_idx):
    return jtu.tree_map(lambda x: jnp.full_like(x, 10), args.x)

# make sure to use the same name here as above
register_function("f", wrap_forward_laplacian(f, custom_jac_hessian_jac=custom_jac_hessian_jac))

@forward_laplacian
def g(x):
    return f(x)

g(jnp.ones(())).laplacian # 10
```


### Sparsity
Sparsity is detected at compile time, this has the advantage of avoiding expensive index computations at runtime and enables efficient reductions. However, it completely prohibits dynamic indexing, i.e., if indices are data-dependent we will simply default to full jacobians.

As we know a lot about the sparsity structure apriori, e.g., that we are only sparse in one dimension, we use a custom sparsity operations that are more efficient than relying on JAX's default `BCOO` (further, at the time of writing, the support for `jax.experimental.sparse` is quite bad).
So, the sparsity data format is implemented in `FwdJacobian` in `api.py`. Instead of storing a dense array `(m, n)` for a function `f:R^n -> R^m`, we store only the non-zero data in a `(m,d)` array where `d<n` is the maximum number of non-zero inputs any output depends on.
To be able to recreate the larger `(m,n)` array from the `(m,d)` array, we additional keep track of the indices in the last dimension in a mask `(m,d)` dimensional array of integers `0<mask_ij<n`.

Masks are treated as compile time static and will be traced automatically. If the tracing is not possible, e.g., due to data dependent indexing, we will fall back to a dense implementation. These propagation rules are implemented in `jvp.py`.


### Memory efficiency
The forward laplacian uses more GPU memory due to the full materialization of the Jacobian matrix.
To compensate for this, it is recommended to loop over the batch size (while other implementations typically loop over the Hessian).
We provide an easy to use utility for this via `folx.batched_vmap` which functions like `jax.vmap` but chunks the input into batches and operates on these sequentially.
```python
from folx import batched_vmap

def f(x):
    return x**2

batched_f = batched_vmap(f, max_batch_size=64)
```

As an experimental tool, one can use `folx.experimental.auto_batched_vmap` which will automatically determine the optimal batch size based on the available memory. Though, this is highly experimental and may be a bad estimate.

## Citation
If you find work helpful, please consider citing it as
```
@software{gao2023folx,
  author = {Nicholas Gao and Jonas Köhler and Adam Foster},
  title = {folx - Forward Laplacian for JAX},
  url = {http://github.com/microsoft/folx},
  version = {0.2.5},
  year = {2023},
}
```
as well as the original forward laplacian:
```
@article{li2023forward,
  title={Forward Laplacian: A New Computational Framework for Neural Network-based Variational Monte Carlo},
  author={Li, Ruichen and Ye, Haotian and Jiang, Du and Wen, Xuelan and Wang, Chuwei and Li, Zhe and Li, Xiang and He, Di and Chen, Ji and Ren, Weiluo and Wang, Liwei},
  journal={arXiv preprint arXiv:2307.08214},
  year={2023}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
