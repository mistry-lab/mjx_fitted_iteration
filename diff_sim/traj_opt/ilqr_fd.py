import equinox
import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass
from typing import Callable, Tuple
from mujoco import mjx


def fd_next_state(
    dx_template: mjx.Data,
    mx: mjx.Model,
    set_control_fn: Callable[[mjx.Data, jnp.ndarray], mjx.Data],
    x: jnp.ndarray,
    u: jnp.ndarray
) -> jnp.ndarray:
    """
    Forward function (no derivative).
    x: [qpos; qvel]
    u: control
    Returns x_next: [qpos_next; qvel_next].
    """
    nq = mx.nq
    # Reuse pre-made dx_template
    dx_local = dx_template.replace(
        qpos=dx_template.qpos.at[:].set(x[:nq]),
        qvel=dx_template.qvel.at[:].set(x[nq:])
    )
    dx_local = set_control_fn(dx_local, u)
    dx_local = mjx.step(mx, dx_local)  # calls the while_loop inside
    x_next = jnp.concatenate([dx_local.qpos, dx_local.qvel])
    return x_next


def fd_jacobian_xu(
    dx_template: mjx.Data,
    mx: mjx.Model,
    set_control_fn: Callable[[mjx.Data, jnp.ndarray], mjx.Data],
    x: jnp.ndarray,
    u: jnp.ndarray,
    eps: float = 1e-6
):
    """
    Returns (f_x, f_u) by finite differences at (x, u).
    f_x has shape (state_dim, state_dim).
    f_u has shape (state_dim, control_dim).
    """
    f0 = fd_next_state(dx_template, mx, set_control_fn, x, u)
    xdim = x.size
    udim = u.size

    # FD wrt x
    def fd_x(i):
        dx_ = jnp.zeros_like(x).at[i].set(eps)
        f_plus = fd_next_state(dx_template, mx, set_control_fn, x + dx_, u)
        return (f_plus - f0) / eps

    Jx = jax.vmap(fd_x)(jnp.arange(xdim))   # shape: (xdim, state_dim)
    f_x = Jx.T  # shape: (state_dim, xdim)

    # FD wrt u
    def fd_u(i):
        du_ = jnp.zeros_like(u).at[i].set(eps)
        f_plus = fd_next_state(dx_template, mx, set_control_fn, x, u + du_)
        return (f_plus - f0) / eps

    Ju = jax.vmap(fd_u)(jnp.arange(udim))  # shape: (udim, state_dim)
    f_u = Ju.T  # shape: (state_dim, udim)

    return f_x, f_u


def cost_prestate(
    dx_template: mjx.Data,
    mx: mjx.Model,
    running_cost_fn: Callable[[mjx.Data], float],
    x: jnp.ndarray,  # [qpos; qvel]
    u: jnp.ndarray
) -> float:
    """
    Build a dx_local with qpos=x[:nq], qvel=x[nq:], ctrl=u,
    then call running_cost_fn(dx_local). No stepping!
    """
    nq = mx.nq
    dx_local = dx_template.replace(
        qpos=dx_template.qpos.at[:].set(x[:nq]),
        qvel=dx_template.qvel.at[:].set(x[nq:]),
        ctrl=dx_template.ctrl.at[:].set(u)
    )
    return running_cost_fn(dx_local)


def linearize_dynamics_and_cost_fd(
    dx_template: mjx.Data,
    mx: mjx.Model,
    set_control_fn: Callable[[mjx.Data, jnp.ndarray], mjx.Data],
    running_cost_fn: Callable[[mjx.Data], float],
    X: jnp.ndarray,  # shape: (N+1, state_dim)
    U: jnp.ndarray,  # shape: (N, control_dim)
):
    """
    For t=0..N-1:
      1) f_x, f_u = FD derivatives of x_{t+1} = f(x_t, u_t).
      2) c_t = cost_prestate(x_t, u_t);
              compute c_x, c_u, c_xx, c_ux, c_uu by AD.
    """
    def single_lin(x, u):
        # FD for dynamics
        f_x, f_u = fd_jacobian_xu(dx_template, mx, set_control_fn, x, u)

        # AD for cost
        def c_fn_xy(xx, uu):
            return cost_prestate(dx_template, mx, running_cost_fn, xx, uu)

        c_val = c_fn_xy(x, u)

        # Grad wrt x
        c_x_ = jax.grad(lambda xx: c_fn_xy(xx, u))(x)   # shape (state_dim,)
        # Grad wrt u
        c_u_ = jax.grad(lambda uu: c_fn_xy(x, uu))(u)   # shape (control_dim,)

        # Hessians
        c_xx_ = jax.jacrev(lambda xx: jax.grad(c_fn_xy, argnums=0)(xx, u))(x)
        c_uu_ = jax.jacrev(lambda uu: jax.grad(c_fn_xy, argnums=1)(x, uu))(u)

        # Cross partial: d/dx of c_u => shape (control_dim, state_dim).
        # For iLQR, we want c_ux = partial^2 c / partial u partial x => also (control_dim, state_dim).
        # So just do:
        def cross_fn_xu(xx):
            return jax.grad(c_fn_xy, argnums=1)(xx, u)  # c_u wrt x
        c_ux_ = jax.jacrev(cross_fn_xu)(x)  # shape (control_dim, state_dim)

        return (f_x, f_u, c_val, c_x_, c_u_, c_xx_, c_ux_, c_uu_)

    # vmap over the first N time steps
    (
        f_x_all,
        f_u_all,
        c_stage_all,
        c_x_all,
        c_u_all,
        c_xx_all,
        c_ux_all,
        c_uu_all
    ) = jax.vmap(single_lin)(X[:-1], U)

    return f_x_all, f_u_all, c_stage_all, c_x_all, c_u_all, c_xx_all, c_ux_all, c_uu_all


@equinox.filter_jit
def simulate_trajectory_ilqr_fd(
    dx_template: mjx.Data,
    mx: mjx.Model,
    qpos_init: jnp.ndarray,
    set_control_fn: Callable[[mjx.Data, jnp.ndarray], mjx.Data],
    running_cost_fn: Callable[[mjx.Data], float],
    terminal_cost_fn: Callable[[mjx.Data], float],
    U: jnp.ndarray   # shape (N, nu)
):
    """
    Forward pass for iLQR, collecting:
      X: shape (N+1, state_dim)
      C: shape (N+1,)
    c_0..c_{N-1} are the running costs, c_N is the terminal cost.
    """
    nq = mx.nq
    dx0 = dx_template
    x0 = jnp.concatenate([qpos_init, jnp.zeros_like(dx0.qvel)])

    def scan_fn(x, u):
        # compute cost pre-step
        c_t = cost_prestate(dx0, mx, running_cost_fn, x, u)
        # next state
        x_next = fd_next_state(dx0, mx, set_control_fn, x, u)
        return x_next, (x, c_t)

    x_final, (X_partial, Costs_partial) = lax.scan(scan_fn, x0, U)
    X = jnp.vstack([X_partial, x_final])

    # Terminal cost on x_final
    dx_final = dx0.replace(
        qpos=dx0.qpos.at[:].set(x_final[:nq]),
        qvel=dx0.qvel.at[:].set(x_final[nq:])
    )
    c_terminal = terminal_cost_fn(dx_final)
    Costs = jnp.hstack([Costs_partial, c_terminal])

    return X, Costs


@equinox.filter_jit
def backward_pass_fd(
    f_x, f_u,
    c_x, c_u, c_xx, c_ux, c_uu,
    x_final: jnp.ndarray,
    terminal_cost_fn: Callable[[mjx.Data], float],
    dx_template: mjx.Data,
    mx: mjx.Model,
    reg: float = 1e-6
):
    """
    Standard iLQR backward pass.
    We'll get the terminal expansions via AD on terminal_cost_fn,
    but that does not call mjx.step.
    """
    nq = mx.nq
    dx0 = dx_template

    # Terminal expansions
    def t_cost_fn(xx):
        dx_local = dx0.replace(
            qpos=dx0.qpos.at[:].set(xx[:nq]),
            qvel=dx0.qvel.at[:].set(xx[nq:])
        )
        return terminal_cost_fn(dx_local)

    # V_x, V_xx at final
    V_x_init = jax.grad(t_cost_fn)(x_final)
    V_xx_init = jax.jacrev(lambda xx: jax.grad(t_cost_fn)(xx))(x_final)

    f_x_rev = jnp.flip(f_x, axis=0)
    f_u_rev = jnp.flip(f_u, axis=0)
    c_x_rev = jnp.flip(c_x, axis=0)
    c_u_rev = jnp.flip(c_u, axis=0)
    c_xx_rev = jnp.flip(c_xx, axis=0)
    c_ux_rev = jnp.flip(c_ux, axis=0)
    c_uu_rev = jnp.flip(c_uu, axis=0)

    def bp_step(carry, inp):
        V_x, V_xx = carry
        f_x_t, f_u_t, c_x_t, c_u_t, c_xx_t, c_ux_t, c_uu_t = inp

        # The shapes we want:
        # f_x_t: (nx, nx),    f_u_t: (nx, nu)
        # c_x_t: (nx,),       c_u_t: (nu,)
        # c_xx_t:(nx, nx),    c_ux_t:(nu, nx),  c_uu_t:(nu, nu)
        # => Q_x, Q_u => (nx,), (nu,)
        # => Q_xx => (nx, nx)
        # => Q_ux => (nu, nx)
        # => Q_uu => (nu, nu)

        Q_x = c_x_t + f_x_t.T @ V_x
        Q_u = c_u_t + f_u_t.T @ V_x
        Q_xx = c_xx_t + f_x_t.T @ V_xx @ f_x_t
        Q_ux = c_ux_t + f_u_t.T @ V_xx @ f_x_t
        Q_uu = c_uu_t + f_u_t.T @ V_xx @ f_u_t

        # regularize Q_uu
        Q_uu_reg = Q_uu + reg * jnp.eye(Q_uu.shape[0])

        K_t = -jnp.linalg.solve(Q_uu_reg, Q_ux)  # shape (nu, nx)
        k_t = -jnp.linalg.solve(Q_uu_reg, Q_u)   # shape (nu,)

        # update V_x, V_xx
        V_x_new = Q_x + K_t.T @ Q_uu @ k_t + K_t.T @ Q_u + Q_ux.T @ k_t
        V_xx_new = Q_xx + K_t.T @ Q_uu @ K_t + K_t.T @ Q_ux + Q_ux.T @ K_t

        return (V_x_new, V_xx_new), (K_t, k_t)

    (V_x_final, V_xx_final), (K_rev, k_rev) = lax.scan(
        bp_step, (V_x_init, V_xx_init),
        (f_x_rev, f_u_rev, c_x_rev, c_u_rev, c_xx_rev, c_ux_rev, c_uu_rev)
    )

    K = jnp.flip(K_rev, axis=0)
    k = jnp.flip(k_rev, axis=0)
    return K, k


@equinox.filter_jit
def forward_pass_fd(
    dx_template: mjx.Data,
    mx: mjx.Model,
    set_control_fn: Callable[[mjx.Data, jnp.ndarray], mjx.Data],
    X_nom: jnp.ndarray,
    U_nom: jnp.ndarray,
    K: jnp.ndarray,
    k: jnp.ndarray,
    alpha: float
):
    """
    Standard iLQR forward pass for the new controls:
      u_new_t = U_nom_t + alpha*(k_t + K_t (x - X_nom_t)).
    We do not re-AD through mjx.step. We just do FD steps for the next state.
    """

    def scan_fn(carry, inp):
        x_cur = carry
        K_t, k_t, U_nom_t, X_nom_t = inp
        u_new = U_nom_t + alpha*(k_t + K_t @ (x_cur - X_nom_t))
        x_next = fd_next_state(dx_template, mx, set_control_fn, x_cur, u_new)
        return x_next, u_new

    inps = (K, k, U_nom, X_nom[:-1])
    x0 = X_nom[0]
    _, U_new = lax.scan(scan_fn, x0, inps)
    return U_new


def make_ilqr_step_fd(
    dx_template: mjx.Data,
    mx: mjx.Model,
    qpos_init: jnp.ndarray,
    set_control_fn: Callable[[mjx.Data, jnp.ndarray], mjx.Data],
    running_cost_fn: Callable[[mjx.Data], float],
    terminal_cost_fn: Callable[[mjx.Data], float],
    reg: float = 1e-6,
    alpha: float = 1.0
):
    """
    Build a single iLQR iteration function that:
      1) Rolls out the current (X, U) via FD
      2) Linearizes f, cost => (f_x, f_u), (c_x, c_u, c_xx, c_ux, c_uu)
      3) Backward pass => (K, k)
      4) Forward pass => new U
    """

    @equinox.filter_jit
    def ilqr_step(U0: jnp.ndarray):
        # 1) Forward rollout
        X_nom, C_nom = simulate_trajectory_ilqr_fd(
            dx_template, mx, qpos_init,
            set_control_fn, running_cost_fn, terminal_cost_fn,
            U0
        )
        # 2) Linearize
        f_x, f_u, _, c_x, c_u, c_xx, c_ux, c_uu = linearize_dynamics_and_cost_fd(
            dx_template, mx, set_control_fn, running_cost_fn,
            X_nom, U0
        )
        # 3) Backward pass
        K, k = backward_pass_fd(
            f_x, f_u, c_x, c_u, c_xx, c_ux, c_uu,
            X_nom[-1],
            terminal_cost_fn,
            dx_template,
            mx,
            reg
        )
        # 4) Forward pass
        U_new = forward_pass_fd(
            dx_template, mx, set_control_fn,
            X_nom, U0,
            K, k,
            alpha
        )
        return U_new, C_nom

    return ilqr_step


@dataclass
class ILQR:
    ilqr_step: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]

    def solve(self, U0: jnp.ndarray, tol=1e-6, max_iter=50):
        """
        Repeatedly call self.ilqr_step.
        """
        U = U0
        prev_cost = jnp.inf
        for i in range(max_iter):
            U_new, C = self.ilqr_step(U)
            total_cost = jnp.sum(C)
            print(f"Iteration {i}: cost = {total_cost}")

            improvement = prev_cost - total_cost
            if improvement < 0:
                pass  # cost got worse => maybe do line search

            if jnp.abs(improvement) < tol:
                print(f"Converged at iteration {i} with cost={total_cost}")
                break

            dU = jnp.linalg.norm(U_new - U)
            if dU < tol:
                print(f"Control update norm below tol at iteration {i}")
                break

            U = U_new
            prev_cost = total_cost

        return U, total_cost
