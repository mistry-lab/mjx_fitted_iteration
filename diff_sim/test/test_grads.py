import jax
from jax import custom_vjp
import jax.numpy as jnp

# dx3/dx = dx3/dx2(16) * dx2/dx1(8) * dx1/dx (4)
@jax.custom_vjp
def step(x):
    return x**2  # Updated carry and output for the sequence

def step_fwd(x):
    return x**2, 2*x  # Updated carry, None (for sequence), and residual for bwd pass

def step_bwd(res, g):
    df_dx = res  # Residual is 2
    vjp = df_dx * g
    jax.debug.print("{}", g)
    jax.debug.print("{}", df_dx)
    return (vjp,)

step.defvjp(step_fwd, step_bwd)


def f(xs):
    def f_scan(carry, _):
        return step(carry), None

    final_carry, _ = jax.lax.scan(f_scan, xs=xs, init=xs[0])
    cost = final_carry * 1
    return cost


print(jax.grad(f)(jnp.array([2.0, 2.0, 2.0])))
