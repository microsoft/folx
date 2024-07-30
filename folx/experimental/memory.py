import functools
import sys
from collections import defaultdict
from io import StringIO
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import Array, core
from jax._src import source_info_util
from jax.util import safe_map


def _bytes_from_dtype(dtype):
    import re

    if str(dtype) == 'bool':
        return 1
    return int(re.findall(r'\d+', str(dtype))[0]) // 8


class MemoryExprEnvironment:
    # A simple environment that keeps track of the variables
    # and frees them once they are no longer needed.
    env: dict[core.Var, jax.Array]
    reference_counter: dict[core.Var, int]
    memory: int = 0

    def __init__(self, jaxpr: core.Jaxpr, consts: Sequence[Array], *args: Array):
        self.env = {}
        self.reference_counter = defaultdict(int)
        for v in jaxpr.invars + jaxpr.constvars:
            if isinstance(v, core.Literal):
                continue
            self.reference_counter[v] += 1
        eqn: core.JaxprEqn
        for eqn in jaxpr.eqns:
            for vi in eqn.invars:
                if isinstance(vi, core.Literal):
                    continue
                self.reference_counter[vi] += 1
        for vo in jaxpr.outvars:
            if isinstance(vo, core.Literal):
                continue
            self.reference_counter[vo] = np.iinfo(np.int32).max
        self.write_many(jaxpr.constvars, consts)
        self.write_many(jaxpr.invars, args)

    def read(self, var: core.Atom) -> Array:
        if isinstance(var, core.Literal):
            return var.val
        self.reference_counter[var] -= 1
        result = self.env[var]
        if self.reference_counter[var] == 0:
            del self.env[var]
            del self.reference_counter[var]
        return result

    def write(self, var: core.Var, val: Array):
        if self.reference_counter[var] > 0:
            self.env[var] = val
            v = jnp.asarray(val)
            self.memory += v.size * _bytes_from_dtype(v.dtype)

    def read_many(self, vars: Sequence[core.Atom]) -> list[Array]:
        return safe_map(self.read, vars)

    def write_many(self, vars: Sequence[core.Var], vals: Sequence[Array]):
        return safe_map(self.write, vars, vals)

    def _free_memory(self, var: core.Atom, val: Array):
        if isinstance(var, core.Literal):
            return
        if self.reference_counter[var] == 0:
            v = jnp.asarray(val)
            self.memory -= v.size * _bytes_from_dtype(v.dtype)

    def free_memory(self, vars: Sequence[core.Atom], vals: Sequence[Array]):
        safe_map(self._free_memory, vars, vals)


def _compute_jaxpr_memory(
    jaxpr: core.Jaxpr,
    consts,
    *args,
    return_result=False,
    print_top_tensors: bool = False,
):
    env = MemoryExprEnvironment(jaxpr, consts, *args)

    peak_memory = 0
    all_tensors = []
    for eqn in jaxpr.eqns:
        invals = env.read_many(eqn.invars)

        sub_peak_mem = 0
        if eqn.primitive.name == 'pjit':
            sub_expr = eqn.params['jaxpr']
            ans, sub_peak_mem, sub_tensors = _compute_jaxpr_memory(
                sub_expr.jaxpr, sub_expr.literals, *invals, return_result=True
            )
            all_tensors += sub_tensors
        elif eqn.primitive.name == 'scan':
            n_carry, n_const = eqn.params['num_carry'], eqn.params['num_consts']
            in_const, in_carry, in_inp = (
                invals[:n_const],
                invals[n_const : n_carry + n_const],
                invals[n_const + n_carry :],
            )
            ans, sub_peak_mem, sub_tensors = _compute_jaxpr_memory(
                eqn.params['jaxpr'].jaxpr,
                (),
                *in_const,
                *in_carry,
                *jtu.tree_map(lambda x: x[0], in_inp),
                return_result=True,
            )
            # populate the results by the loop size
            ans[n_carry:] = jtu.tree_map(
                lambda x: jnp.repeat(x[None], in_inp[0].shape[0], axis=0),
                ans[n_carry:],
            )
            all_tensors += sub_tensors
        else:
            subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
            ans = eqn.primitive.bind(*subfuns, *invals, **bind_params)

        source_info = source_info_util.summarize(eqn.source_info)
        if eqn.primitive.multiple_results:
            env.write_many(eqn.outvars, ans)
            for v, t in zip(eqn.outvars, ans):
                all_tensors.append((v, t, source_info))
        else:
            env.write(eqn.outvars[0], ans)
            all_tensors.append((eqn.outvars[0], ans, source_info))

        if env.memory + sub_peak_mem > peak_memory:
            peak_memory = env.memory + sub_peak_mem

        all_tensors = sorted(
            all_tensors,
            key=lambda x: x[1].size * _bytes_from_dtype(x[1].dtype),
            reverse=True,
        )
        if not return_result and print_top_tensors:
            print('Largest Tensors')
            print('=' * 80)
            for t in all_tensors[:10]:
                size = t[1].size * _bytes_from_dtype(t[1].dtype)
                size_in_gb = size / 1024**3
                print(t[2], f'{size_in_gb:.2f}GB', t[1].shape, t[1].dtype)

        env.free_memory(eqn.invars, invals)

    if return_result:
        return env.read_many(jaxpr.outvars), peak_memory, all_tensors
    print(peak_memory)
    return None


class Capturing(list):  # Utility that captured the stdout of a function
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def compute_memory(
    f,
    *args,
    static_argnums: int | Sequence[int] = (),
    print_top_tensors: bool = False,
    **kwargs,
):
    """
    Computes the maximum memory consumption of a function in bytes. This may be highly inaccurate since it is
    only estimated based on the Jaxpr which is executed sequentially. The compiler may optimize the memory usage
    significantly.

    Args:
        f: The function to compute the memory usage of.
        *args: The arguments to the function.
        static_argnums: The arguments to the function that are static.
        print_top_tensors: Whether to print the top tensors.
        **kwargs: The keyword arguments to the function.

    Returns:
        The estimated maximum memory consumption in bytes
    """
    if isinstance(static_argnums, int):
        static_argnums = (static_argnums,)
    closed_jaxpr = jax.make_jaxpr(f, static_argnums=static_argnums)(*args, **kwargs)
    # enclose function
    fn = functools.partial(
        _compute_jaxpr_memory,
        closed_jaxpr.jaxpr,
        closed_jaxpr.literals,
        print_top_tensors=print_top_tensors,
    )
    # remove statics
    args = tuple(args[i] for i in range(len(args)) if i not in static_argnums)
    # Here we abuse the eval_shape function to avoid actually computing something.
    # Instead the function prints to correct usage
    with Capturing() as output:
        jax.eval_shape(fn, *jtu.tree_leaves(args))

    # Print everything except the last line
    if len(output) > 1:
        print('\n'.join(output[:-1]))

    # Return the last line
    return int(output[-1])
