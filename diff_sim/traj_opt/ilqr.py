import jax
import jax.numpy as jnp
from mujoco import mjx
import equinox
from typing import Callable

from jax import config
from pydantic.dataclasses import dataclass

config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)

def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x


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


def make_ilqr_step(
        mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn, reg=1e-6, alpha=1.0
):
    @equinox.filter_jit
    def terminal_expansion(x):
        nq = mx.nq

        # Extract final state x
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
        Linearize about (X,U). We have N steps, so arrays have length N.
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
        """
        Backward pass of iLQR:
        Use terminal expansions as initial conditions for V_x and V_xx.
        No concatenation of terminal arrays. Just initialize V_x, V_xx from terminal_expansion.
        Then process steps from N-1 to 0.
        """
        t_c, t_c_x, t_c_xx = terminal_expansion(x_final)
        # Initial conditions for value function at final time
        V_x = t_c_x
        V_xx = t_c_xx

        state_dim = f_x.shape[1]  # dimension of state
        nu = f_u.shape[2]  # dimension of control

        # We'll run from step N-1 down to 0
        # Reverse arrays for scanning backward
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

    @equinox.filter_jit
    def forward_pass(X, U, K, k):
        def step_fn(carry, inp):
            x, = carry
            K_t, k_t, U_nom_t, X_nom_t = inp
            u_new = U_nom_t + alpha * (k_t + K_t @ (x - X_nom_t))
            x_next, _ = f_state_input(x, u_new)
            return (x_next,), u_new

        inp = (K, k, U, X[:-1])
        (_, U_new) = jax.lax.scan(step_fn, (X[0],), inp)
        return U_new

    def ilqr_step(U0):
        U = U0
        X, U_out, C = simulate_trajectory_ilqr(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn, U)
        f_x, f_u, c_x, c_u, c_xx, c_ux, c_uu = linearize_dynamics_and_cost(X, U)
        K, k = backward_pass(f_x, f_u, c_x, c_u, c_xx, c_ux, c_uu, X[-1])
        U_new = forward_pass(X, U, K, k)
        return U_new, C

    return ilqr_step


@dataclass
class ILQR:
    ilqr_step: Callable[[jnp.ndarray], jnp.ndarray]

    def solve(self, U0: jnp.ndarray, tol=1e-6, max_iter=50):
        U = U0
        prev_cost = jnp.inf
        total_cost = jnp.inf
        for i in range(max_iter):
            U_new, C = self.ilqr_step(U)
            total_cost = jnp.sum(C)
            print(f"Iteration {i}: cost={total_cost}")

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
