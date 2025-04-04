import jax
import jax.numpy as jnp
# import mujoco
from mujoco import mjx
import equinox
from typing import Callable
import time

from jax import config
from pydantic.dataclasses import dataclass

config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)
 
# ------------------------------------------------------------------------
# simulate_trajectory_ilqr remains unchanged
# ------------------------------------------------------------------------
 
@equinox.filter_jit
def simulate_trajectory_ilqr(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn, U):
    """
    Simulate a trajectory given a control sequence U for iLQR.
    """
    def step_fn(x, u):
        nq = mx.nq
        dx = mjx.make_data(mx)
        dx = dx.replace(
            qpos=dx.qpos.at[:].set(x[:nq]),
            qvel=dx.qvel.at[:].set(x[nq:])
        )
        dx = set_control_fn(dx, u)
        dx = mjx.step(mx, dx)
        x_next = jnp.concatenate([dx.qpos, dx.qvel])
        c = running_cost_fn(dx)
        return x_next, (x, u, c)
    
    dx_init = mjx.make_data(mx)
    nq = mx.nq
    qvel_init = jnp.zeros_like(dx_init.qvel)
    x0 = jnp.concatenate([qpos_init, qvel_init])
    x_final, (X_partial, U_out, C_partial) = jax.lax.scan(step_fn, x0, U)

    # Add final state and terminal cost
    X = jnp.vstack((X_partial, x_final))
    dx_final = mjx.make_data(mx)
    dx_final = dx_final.replace(
        qpos=dx_final.qpos.at[:].set(x_final[:nq]),
        qvel=dx_final.qvel.at[:].set(x_final[nq:])
    )
    term_c = terminal_cost_fn(dx_final)
    C = jnp.hstack((C_partial, term_c))
    return X, U_out, C
 
 
# ------------------------------------------------------------------------
# iLQR Step and Linearization functions (unchanged)
# ------------------------------------------------------------------------
 
def make_ilqr_step(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn, reg=1e-6, alpha=1.0):
    @equinox.filter_jit
    def terminal_expansion(x):
        nq = mx.nq
        # x is [qpos; qvel]
        def tc(x_):
            dx_local = mjx.make_data(mx).replace(
                qpos=x_[:nq],
                qvel=x_[nq:]
            )
            return terminal_cost_fn(dx_local)
        t_c = tc(x)
        t_c_x = jax.grad(tc)(x)
        t_c_xx = jax.jacfwd(lambda xx: jax.grad(tc)(xx))(x)
        return t_c, t_c_x, t_c_xx
 
    @equinox.filter_jit
    def f_state_input(x, u):
        nq = mx.nq
        dx = mjx.make_data(mx)
        dx = dx.replace(
            qpos=dx.qpos.at[:].set(x[:nq]),
            qvel=dx.qvel.at[:].set(x[nq:])
        )
        dx = set_control_fn(dx, u)
        dx = mjx.step(mx, dx)
        x_next = jnp.concatenate([dx.qpos, dx.qvel])
        c = running_cost_fn(dx)
        return x_next, c
 
    @equinox.filter_jit
    def linearize_dynamics_and_cost(X, U):
        """
        Linearize about (X,U). We have N steps.
        """
        def single_lin(x, u):
            x_next, c = f_state_input(x, u)
            f_x = jax.jacfwd(lambda xx: f_state_input(xx, u)[0])(x)
            f_u = jax.jacfwd(lambda uu: f_state_input(x, uu)[0])(u)
            c_x = jax.grad(lambda xx: f_state_input(xx, u)[1])(x)
            c_u = jax.grad(lambda uu: f_state_input(x, uu)[1])(u)
            c_xx = jax.jacfwd(lambda xx: jax.grad(lambda xxx: f_state_input(xxx, u)[1])(xx))(x)
            c_uu = jax.jacfwd(lambda uu: jax.grad(lambda uuu: f_state_input(x, uuu)[1])(uu))(u)
            c_ux = jax.jacfwd(lambda xx: jax.grad(lambda uu: f_state_input(xx, uu)[1])(u))(x)
            return f_x, f_u, c_x, c_u, c_xx, c_ux, c_uu
        
        f_x_all, f_u_all, c_x_all, c_u_all, c_xx_all, c_ux_all, c_uu_all = jax.vmap(single_lin)(X[:-1], U)
        return f_x_all, f_u_all, c_x_all, c_u_all, c_xx_all, c_ux_all, c_uu_all
    
    @equinox.filter_jit
    def backward_pass(f_x, f_u, c_x, c_u, c_xx, c_ux, c_uu, x_final):
        t_c, t_c_x, t_c_xx = terminal_expansion(x_final)
        V_x = t_c_x
        V_xx = t_c_xx

        f_x_rev = jnp.flip(f_x, axis=0)
        f_u_rev = jnp.flip(f_u, axis=0)
        c_x_rev = jnp.flip(c_x, axis=0)
        c_u_rev = jnp.flip(c_u, axis=0)
        c_xx_rev = jnp.flip(c_xx, axis=0)
        c_ux_rev = jnp.flip(c_ux, axis=0)
        c_uu_rev = jnp.flip(c_uu, axis=0)


        def bp_fn(carry, inp):
            V_x, V_xx = carry
            f_x_t, f_u_t, c_x_t, c_u_t, c_xx_t, c_ux_t, c_uu_t = inp

            Q_x = c_x_t + f_x_t.T @ V_x
            Q_u = c_u_t + f_u_t.T @ V_x
            Q_xx = c_xx_t + f_x_t.T @ V_xx @ f_x_t
            Q_ux = c_ux_t + f_u_t.T @ V_xx @ f_x_t
            Q_uu = c_uu_t + f_u_t.T @ V_xx @ f_u_t

            Q_uu_reg = Q_uu + reg * jnp.eye(Q_uu.shape[0])
            K_t = -jnp.linalg.solve(Q_uu_reg, Q_ux)
            k_t = -jnp.linalg.solve(Q_uu_reg, Q_u)

            V_x_new = Q_x + K_t.T @ Q_uu @ k_t + K_t.T @ Q_u + Q_ux.T @ k_t
            V_xx_new = Q_xx + K_t.T @ Q_uu @ K_t + K_t.T @ Q_ux + Q_ux.T @ K_t

            return (V_x_new, V_xx_new), (K_t, k_t)
        
        (V_x_final, V_xx_final), (K_rev, k_rev) = jax.lax.scan(
            bp_fn, (V_x, V_xx),(f_x_rev, f_u_rev, c_x_rev, c_u_rev, c_xx_rev, c_ux_rev, c_uu_rev)
        )
        # Reverse K and k back to original order
        K = jnp.flip(K_rev, axis=0)
        k = jnp.flip(k_rev, axis=0)
        return K, k
 
    # --------------------------------------------------------------------
    # NEW: Forward Pass with Discrete Line Search (vectorized over alpha)
    # --------------------------------------------------------------------
    @equinox.filter_jit
    def forward_pass_ls(X, U, K, k, alpha_candidates):
        """
        For a candidate array of alphas, rollout the new controls and compute
        the trajectory cost for each. Then select the best alpha.
        """
        def rollout_for_alpha(alpha):
            # Define the step function that uses the given alpha.
            def step_fn(carry, inp):
                x = carry
                K_t, k_t, U_nom_t, X_nom_t = inp
                u_new = U_nom_t + alpha * (k_t + K_t @ (x - X_nom_t))
                x_next, _ = f_state_input(x, u_new)
                return x_next, u_new
            x_final, U_new = jax.lax.scan(step_fn, X[0], (K, k, U, X[:-1]))
            # Compute terminal cost using the final state rollout.
            nq = mx.nq
            dx_final = mjx.make_data(mx)
            dx_final = dx_final.replace(
                qpos=dx_final.qpos.at[:].set(x_final[:nq]),
                qvel=dx_final.qvel.at[:].set(x_final[nq:])
            )
            term_cost = terminal_cost_fn(dx_final)
            # Also, accumulate running cost from simulation.
            # For simplicity, re-run the simulation using U_new.
            _, _, C = simulate_trajectory_ilqr(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn, U_new)
            total_cost = jnp.sum(C)
            return U_new, total_cost
 
        # Vectorize the rollout for each candidate alpha in parallel.
        U_candidates, cost_candidates = jax.vmap(rollout_for_alpha)(alpha_candidates)
        i_best = jnp.argmin(cost_candidates)
        best_U = jax.tree_util.tree_map(lambda arr: arr[i_best], U_candidates)
        best_cost = cost_candidates[i_best]
        jax.debug.print("i_best : {}", i_best)
        return best_U, best_cost
 
    @equinox.filter_jit
    def forward_pass(X, U, K, k):
        # Define candidate alphas (for example, 1.0, 0.5, 0.25, 0.125).
        alpha_candidates = jnp.array([1.00000000e+00, 9.09090909e-01,
                        6.83013455e-01, 4.24097618e-01,
                        2.17629136e-01, 9.22959982e-02,
                        3.23491843e-02, 9.37040641e-03,
                        2.24320079e-03, 4.43805318e-04, 0.00000001], dtype=jnp.float64)
        return forward_pass_ls(X, U, K, k, alpha_candidates)

    # @equinox.filter_jit
    # def forward_pass(X, U, K, k):
    #     def step_fn(carry, inp):
    #         x, = carry
    #         K_t, k_t, U_nom_t, X_nom_t = inp
    #         u_new = U_nom_t + alpha * (k_t + K_t @ (x - X_nom_t))
    #         x_next, _ = f_state_input(x, u_new)
    #         return (x_next,), u_new

    #     inp = (K, k, U, X[:-1])
    #     (_, U_new) = jax.lax.scan(step_fn, (X[0],), inp)
    #     return U_new    
    
    def ilqr_step(U0):
        U = U0
        X, U_out, C = simulate_trajectory_ilqr(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn, U)
        f_x, f_u, c_x, c_u, c_xx, c_ux, c_uu = linearize_dynamics_and_cost(X, U)
        K, k = backward_pass(f_x, f_u, c_x, c_u, c_xx, c_ux, c_uu, X[-1])
        U_new, C_new = forward_pass(X, U, K, k)
        return U_new, C_new
    
    return ilqr_step


# ------------------------------------------------------------------------
# The main iLQR trajectory simulation function remains unchanged.
# ------------------------------------------------------------------------
 
# @equinox.filter_jit
# def simulate_trajectory_ilqr(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn, U):
#     def step_fn(x, u):
#         nq = mx.nq
#         dx = mjx.make_data(mx)
#         dx = dx.replace(
#             qpos=dx.qpos.at[:].set(x[:nq]),
#             qvel=dx.qvel.at[:].set(x[nq:])
#         )
#         dx = set_control_fn(dx, u)
#         dx = mjx.step(mx, dx)
#         x_next = jnp.concatenate([dx.qpos, dx.qvel])
#         c = running_cost_fn(dx)
#         return x_next, (x, u, c)
#     dx_init = mjx.make_data(mx)
#     nq = mx.nq
#     qvel_init = jnp.zeros_like(dx_init.qvel)
#     x0 = jnp.concatenate([qpos_init, qvel_init])
#     x_final, (X_partial, U_out, C_partial) = jax.lax.scan(step_fn, x0, U)
#     X = jnp.vstack((X_partial, x_final))
#     dx_final = mjx.make_data(mx)
#     dx_final = dx_final.replace(
#         qpos=dx_final.qpos.at[:].set(x_final[:nq]),
#         qvel=dx_final.qvel.at[:].set(x_final[nq:])
#     )
#     term_c = terminal_cost_fn(dx_final)
#     C = jnp.hstack((C_partial, term_c))
#     return X, U_out, C
 
 
# ------------------------------------------------------------------------
# Define a simple ILQR context data structure for the outer while_loop.
# ------------------------------------------------------------------------
 
# class _ILQRContext(PyTreeNode):
#     """
#     Holds iteration state for the while_loop-based iLQR solve.
#     """
#     U: jnp.ndarray             # Current control sequence
#     prev_cost: jnp.ndarray           # Cost from previous iteration
#     cost: jnp.ndarray                # Current cost
#     improvement: jnp.ndarray          # prev_cost - cost
#     iteration: jnp.ndarray             # Iteration counter
#     done: jnp.ndarray                 # Whether we've converged
#     max_iter: jnp.ndarray            # Maximum iterations
#     tol: jnp.ndarray                  # Tolerance for improvement

# def _ilqr_cond(ctx: _ILQRContext) -> bool:
#     return jnp.array((
#         (ctx.iteration < ctx.max_iter) &
#         (jnp.abs(ctx.improvement) >= ctx.tol) &
#         (~ctx.done)
#     ))

# def _ilqr_body(ilqr_step_fn):
#     """
#     Returns a function that performs one iLQR iteration in a jax.lax.while_loop.
#     This version embeds the discrete line search within the forward pass via vmap.
#     """
#     def body(ctx: _ILQRContext) -> _ILQRContext:
#         now = time.time()
#         U_candidate, C_candidate = ilqr_step_fn(ctx.U)  # unconstrained candidate update from backward pass
       
       
#         improved = (C_candidate < ctx.cost)

#         new_cost = jnp.where(improved, C_candidate, ctx.cost)
#         new_U = jnp.where(improved, U_candidate, ctx.U)
#         improvement = ctx.cost - new_cost
#         # Update iteration count and store new values in context.

#         # print(f"\n--- Iteration {ctx.iteration + 1} ---")
#         jax.debug.print("Iteration : {}", ctx.iteration+1)
#         # print(f"Time: {time.time() - now}")
#         # print(f"improvement={improvement:0.6f}")


#         return _ILQRContext(
#             U=new_U,
#             prev_cost=ctx.cost,
#             cost=new_cost,
#             improvement=improvement,
#             iteration=ctx.iteration + 1,
#             done=jnp.array(False),  # you can set done based on additional criteria
#             max_iter=ctx.max_iter,
#             tol=ctx.tol
#         )
#     return body

# @dataclass
# class ILQR:
#     ilqr_step: Callable[[jnp.ndarray], jnp.ndarray]
#     def solve(self, U0: jnp.ndarray,
#               tol=1e-6,
#               max_iter=50):
#         """
#         iLQR outer loop with:
#           - jax.lax.while_loop for iteration
#           - a discrete line search (implemented via vmap) within the forward pass.
#         """
#         ctx0 = _ILQRContext(
#             U=U0,
#             prev_cost=jnp.array(jnp.inf),
#             cost=jnp.array(jnp.inf),
#             improvement=jnp.array(jnp.inf),
#             iteration=jnp.array(0),
#             done=jnp.array(False),
#             max_iter=jnp.array(max_iter),
#             tol=jnp.array(tol)
#         )
#         cond_fn = _ilqr_cond
#         # with jax.checking_leaks():
#         body_fn = _ilqr_body(self.ilqr_step)
#         ctx_final = jax.lax.while_loop(cond_fn, body_fn, ctx0)
#         return ctx_final.U, ctx_final.cost


@dataclass
class ILQR:
    ilqr_step: Callable[[jnp.ndarray], jnp.ndarray]

    def solve(self, U0: jnp.ndarray, tol=1e-6, max_iter=50):
        U = U0
        prev_cost = jnp.inf
        total_cost = jnp.inf
        for i in range(max_iter):
            now = time.time()
            U_new, C = self.ilqr_step(U)
            total_cost = jnp.sum(C)
            print(f"\nIteration {i}: cost={total_cost}")
            print(f"Time: {time.time() - now}")

            # Check for cost improvement
            improvement = prev_cost - total_cost
            if improvement < 0:
                # If cost got worse, you might consider adjustments or break
                # For now, we just proceed; you could implement line-search here
                pass

            # Check convergence by improvement and/or by norm of control changes
            if jnp.abs(improvement) < tol:
                print(f"Converged at iteration {i} with cost={total_cost}")
                break

            # Check norm of update to controls
            control_diff_norm = jnp.linalg.norm(U_new - U)
            if control_diff_norm < tol:
                print(f"Control update norm below tolerance at iteration {i}")
                U = U_new
                break

            U = U_new
            prev_cost = total_cost

        return U, total_cost
