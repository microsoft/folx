import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu


def is_tree_complex(tree):
    leaves = jtu.tree_leaves(tree)
    return any(jnp.iscomplexobj(leaf) for leaf in leaves)


def _varying_axes(x: jax.Array) -> tuple:
    if not hasattr(jax, 'typeof'):
        return ()

    typ = jax.typeof(x)
    manual_axis_type = getattr(typ, 'manual_axis_type', None)
    if manual_axis_type is not None:
        return tuple(getattr(manual_axis_type, 'varying', ()))

    return tuple(getattr(typ, 'vma', ()))


def _mark_varying_like(x: jax.Array, like: jax.Array) -> jax.Array:
    axes = _varying_axes(like)
    if not axes:
        return x

    if hasattr(jax.lax, 'pcast'):
        return jax.lax.pcast(x, axes, to='varying')
    if hasattr(jax.lax, 'pvary'):
        return jax.lax.pvary(x, axes)
    return x


def vjp_rc(fun, *primals: jax.Array):
    def real_fun(*primals):
        return jnp.real(fun(*primals))

    def imag_fun(*primals):
        return jnp.imag(fun(*primals))

    _, vjp_r = jax.vjp(real_fun, *primals)
    _, vjp_i = jax.vjp(imag_fun, *primals)

    def vjp(*tangents: jax.Array):
        real_tangents = jtu.tree_map(jnp.real, tangents)
        imag_tangents = jtu.tree_map(jnp.imag, tangents)

        # letters: v=vector, j=jacobian, r=real, i=imag
        vr_jr = vjp_r(*real_tangents)
        vi_jr = vjp_r(*imag_tangents)
        vr_ji = vjp_i(*real_tangents)
        vi_ji = vjp_i(*imag_tangents)

        result = jtu.tree_map(
            lambda vr_jr, vi_jr, vr_ji, vi_ji: vr_jr - vi_ji + 1j * (vr_ji + vi_jr),
            vr_jr,
            vi_jr,
            vr_ji,
            vi_ji,
        )
        return result

    return vjp


def vjp(fun, *primals: jax.Array):
    out, vjp = jax.vjp(fun, *primals)
    if is_tree_complex(primals):
        if is_tree_complex(out):
            # C -> C
            return vjp
        else:
            # C -> R
            return vjp
    else:
        if is_tree_complex(out):
            # R -> C
            return vjp_rc(fun, *primals)
        else:
            # R -> R
            return vjp


def jacrev(f):
    # Similar to jax.jacrev but works with complex inputs and outputs.
    # A crucial difference is that we do not preserve the structure of the output but
    # always flatten it to a 1D array.
    def jacfun(*primals):
        flat_primals, unravel = jfu.ravel_pytree(primals)

        def flat_f(x):
            return jfu.ravel_pytree(f(*unravel(x)))[0]

        out = flat_f(flat_primals)

        eye = jnp.eye(out.size, dtype=out.dtype)
        eye = _mark_varying_like(eye, out)
        result = jax.vmap(vjp(flat_f, flat_primals))(eye)[0]
        result = jax.vmap(unravel, out_axes=0)(result)
        if len(primals) == 1:
            return result[0]
        return result

    return jacfun


def jacfwd(f):
    # Similar to jax.jacfwd but works with complex inputs and outputs.
    # A crucial difference is that we do not preserve the structure of the input but
    # always flatten it to a 1D array.
    def jacfun(*primals):
        flat_primals, unravel = jfu.ravel_pytree(primals)

        def jvp_fun(s):
            return jax.jvp(f, primals, unravel(s))[1]

        eye = jnp.eye(flat_primals.size, dtype=flat_primals.dtype)
        eye = _mark_varying_like(eye, flat_primals)
        J = jax.vmap(jvp_fun, out_axes=-1)(eye)
        return J

    return jacfun


def hessian(f):
    # Similar to jax.hessian but works with complex inputs and outputs.
    return jacfwd(jacrev(f))
