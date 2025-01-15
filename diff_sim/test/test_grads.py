import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

# System :
#   x_next = 0.5 * x + 1. * u
#   x_next = 0.5 * (0.5 * x_n-1 + 1. * u_n-1) + 1. * u
# 3 steps : 
#   U = [u_0, u_1, u_2]
#   x_3 = 0.5*x_2 + u_2
# Loss :
#   L = x_3

# dL/u2 = 1.
# dL/u1 = dL/dx2 dx2/du1 = 0.5 * 1
# dL/u0 = dL/dx2 dx2/dx1 dx1/du0 = 0.5 * 0.5 * 1 

# Results : [0.25 0.5  1.  ]  --> Working
# Get [0., 0., 1.] if grad_x = 0.

def step_oem(x, u):
    return 0.5*x + u

@jax.custom_vjp
def step(x, u):
    return 0.5*x + u  # Updated carry and output for the sequence

def step_fwd(x, u):
    result = 0.5*x + u
    residual = (x, u, result)  # Save carry and x for backward pass
    return result, residual

def step_bwd(residual, g):
    x, u, _ = residual
    df_dx = 0.5 # Derivative w.r.t. x
    df_du = 1.  # Derivative w.r.t. u
    grad_x = g * df_dx
    grad_u = g * df_du
    # jax.debug.print("\n\n Cotangent g : {}", g)
    return grad_x, grad_u

# Corrected version:
def step_bwd_fd(residual, g):
    x, u, result = residual
    eps = 1e-5
    x_eps = step(x + eps, u)
    u_eps = step(x, u + eps)
    df_dx = (x_eps - result) / eps
    df_du = (u_eps - result) / eps
    grad_x = g * df_dx
    grad_u = g * df_du
    jax.debug.print("\n\n x : {}", x)
    jax.debug.print("\n\n Cotangent g : {}", g)
    return grad_x, grad_u

def f(xs):
    def f_scan(carry, x):
        new_carry = step(carry, x)  # Pass both carry and sequence element
        return new_carry, x

    final_carry, x_out = jax.lax.scan(f_scan, init=xs[0], xs=xs[1:])
    cost = final_carry * 1  # Final cost depends on final carry
    return cost

def f_oem(xs):
    def f_scan(carry, x):
        new_carry = step_oem(carry, x)  # Pass both carry and sequence element
        return new_carry, x

    final_carry, x_out = jax.lax.scan(f_scan, init=xs[0], xs=xs[1:])
    cost = final_carry * 1  # Final cost depends on final carry
    return cost

print("First test, L = x3, x_n = 0.5*x_n-1 + u")
print("Gradient wrt U = [u0, u1, u2] : ", jax.grad(f_oem)(jnp.array([10.,10.,10.])))
step.defvjp(step_fwd, step_bwd)
print("Gradient wrt U analytical = [u0, u1, u2] : ", jax.grad(f)(jnp.array([10.,10.,10.])))
step.defvjp(step_fwd, step_bwd_fd)
print("Gradient wrt U finite diff = [u0, u1, u2] : ", jax.grad(f)(jnp.array([10.,10.,10.])))


def running_costs(x,u):
    return x + u 

def terminal_cost(x):
    return x

def f(us,x0):
    def f_scan(carry, x):
        step_costs = running_costs(carry,x)
        new_carry = step(carry, x)  # Pass both carry and sequence element
        return new_carry, step_costs

    final_carry, step_costs = jax.lax.scan(f_scan, init=x0[0], xs=us)
    tcost = terminal_cost(final_carry)
    cost = jnp.sum(step_costs) + tcost
    return cost

print("\n\n")
print("Second test, L = x0 + u0 + ... + u2 + x3, x_n = 0.5*x_n-1 + u")
print("Gradient wrt U finite diff = [u0, u1, u2] : ", jax.grad(f)(jnp.array([0.,2.,2.]) , jnp.array([1.])))
step.defvjp(step_fwd, step_bwd)
print("Gradient wrt U analytical = [u0, u1, u2] : ", jax.grad(f)(jnp.array([0.,2.,2.]) , jnp.array([1.])))
