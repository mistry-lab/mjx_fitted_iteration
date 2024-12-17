import jax
import jax.numpy as jnp

# System :
#   x_next = 0.5 * x + 1. * u
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

@jax.custom_vjp
def step(x, u):
    return 0.5*x + u  # Updated carry and output for the sequence

def step_fwd(x, u):
    result = 0.5*x + u
    residual = (x, u)  # Save carry and x for backward pass
    return result, residual

def step_bwd(residual, g):
    x, u = residual
    df_dx = 0.5 # Derivative w.r.t. x
    df_du = 1.            # Derivative w.r.t. u
    grad_x = g * df_dx
    grad_u = g * df_du
    return grad_x, grad_u

step.defvjp(step_fwd, step_bwd)

def f(xs):
    def f_scan(carry, x):
        new_carry = step(carry, x)  # Pass both carry and sequence element
        return new_carry, None

    final_carry, _ = jax.lax.scan(f_scan, init=xs[0], xs=xs[1:])
    cost = final_carry * 1  # Final cost depends on final carry
    return cost

print("First test, L = x3, x_n = 0.5*x_n-1 + u")
print("Gradient wrt U = [u0, u1, u2] : ", jax.grad(f)(jnp.array([10.,10.,10.])))


############################
# Test 2

# def cost(x,u):
#     return x + u

# @jax.custom_vjp
# def step2(carry, u):
#     x = carry
#     return 0.5*x + u # Updated carry and output for the sequence

# # Forward pass for the custom VJP
# def step2_fwd(x, u):
#     result = 0.5*x + u
#     running_cost = cost(x, u)  # Calculate the cost for this step
#     state = (x, u)  # Save x and u for the backward pass
#     return result, (running_cost, state)

# # # Backward pass for the custom VJP
# def step2_bwd(residual, g):
#     running_cost, state = residual
#     df_dx = 0.5  # Derivative w.r.t. x
#     df_du = 1.0  # Derivative w.r.t. u
#     grad_x = g * df_dx
#     grad_u = g * df_du
#     # grad_cost = running_cost  # The gradient of the cost, passed from the running cost

#     # Return gradients for both inputs (x and u), and also return the gradient of the cost
#     return grad_x, grad_u

# # Attach the forward and backward passes to the custom VJP
# step2.defvjp(step2_fwd, step2_bwd)


# def f2(x, u):
#     def f_scan(carry, u):
#         # Run the step with both the current state (carry) and the input (u), and get the cost
#         new_carry, step_cost = step2(carry, u)
#         running_co cost(x,u) 
#         return new_carry, step_cost

#     # Use scan to iterate over the sequence (u's), starting with initial state x
#     _, step_costs = jax.lax.scan(f_scan, init=x, xs=u)  # Each u corresponds to a time step
    
#     # Aggregate the running cost (sum of costs across all time steps)
#     total_cost = jnp.sum(step_costs)
#     print(step_costs)
#     return total_cost

# print("\n\n\nSecond test, L = x3, x_n = 0.5*x_n-1 + u")
# grad_x, grad_u = jax.grad(f2, (0, 1))(jnp.array([1.]), jnp.array([10.,10.,10.]))
# # print("Gradient wrt U = [u0, u1, u2] : ", grad_u)
# print(f2(jnp.array([1.]), jnp.array([2.,2.,2.])))