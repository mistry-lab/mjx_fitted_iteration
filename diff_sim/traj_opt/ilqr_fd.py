import jax
import jax.numpy as jnp
import equinox as eqx
from mujoco import mjx
from jax import lax

# ---------------------------------------------------------------------
#  Part A: FD-based iLQR subroutines for SINGLE shape
# ---------------------------------------------------------------------

def fd_next_state(dx_template, mx, set_control_fn, x, u):
    """
    x, u are single shapes: x: (nq+nv,), u: (nu,).
    """
    nq = mx.nq
    dx_local = dx_template.replace(
        qpos=dx_template.qpos.at[:].set(x[:nq]),
        qvel=dx_template.qvel.at[:].set(x[nq:])
    )
    dx_local = set_control_fn(dx_local, u)
    dx_local = mjx.step(mx, dx_local)
    return jnp.concatenate([dx_local.qpos, dx_local.qvel])


def fd_jacobian_xu(dx_template, mx, set_control_fn, x, u, eps=1e-6):
    """
    Single shape, returns f_x, f_u each shape (state_dim,).
    """
    f0 = fd_next_state(dx_template, mx, set_control_fn, x, u)
    xdim = x.shape[0]
    udim = u.shape[0]

    def fd_x(i):
        dx_ = jnp.zeros_like(x).at[i].set(eps)
        f_plus = fd_next_state(dx_template, mx, set_control_fn, x + dx_, u)
        return (f_plus - f0) / eps

    Jx = jax.vmap(fd_x)(jnp.arange(xdim))  # shape (xdim, state_dim)
    f_x = Jx.T                              # shape (state_dim, xdim)

    def fd_u(i):
        du_ = jnp.zeros_like(u).at[i].set(eps)
        f_plus = fd_next_state(dx_template, mx, set_control_fn, x, u + du_)
        return (f_plus - f0) / eps

    Ju = jax.vmap(fd_u)(jnp.arange(udim))  # shape (udim, state_dim)
    f_u = Ju.T                             # shape (state_dim, udim)
    return f_x, f_u


def cost_prestate(dx_template, mx, running_cost_fn, x, u):
    """
    Single shapes: x: (nq+nv,), u: (nu,).
    """
    nq = mx.nq
    dx_local = dx_template.replace(
        qpos=dx_template.qpos.at[:].set(x[:nq]),
        qvel=dx_template.qvel.at[:].set(x[nq:]),
        ctrl=dx_template.ctrl.at[:].set(u)
    )
    return running_cost_fn(dx_local)


def simulate_fd_trajectory_single(dx_template, mx, qpos_init, qvel_init,
                                  set_control_fn, running_cost_fn, terminal_cost_fn,
                                  U):
    """
    Single shapes:
      qpos_init: (nq,), qvel_init: (nv,), U: (N, nu).
    Returns X: (N+1, nq+nv), C: (N+1,).
    """
    nq = mx.nq
    x0 = jnp.concatenate([qpos_init, qvel_init])  # shape (nq+nv,)

    def step_fn(x, u):
        c_t = cost_prestate(dx_template, mx, running_cost_fn, x, u)
        x_next = fd_next_state(dx_template, mx, set_control_fn, x, u)
        return x_next, (x, c_t)

    x_final, (X_partial, C_partial) = lax.scan(step_fn, x0, U)
    X = jnp.vstack([X_partial, x_final])

    # terminal cost
    dx_local = dx_template.replace(
        qpos=dx_template.qpos.at[:].set(x_final[:nq]),
        qvel=dx_template.qvel.at[:].set(x_final[nq:])
    )
    c_term = terminal_cost_fn(dx_local)
    C = jnp.hstack([C_partial, c_term])
    return X, C


def linearize_fd_single(dx_template, mx, set_control_fn, running_cost_fn, X, U):
    """
    Single shapes: X: (N+1, state_dim), U: (N, nu).
    """
    def single_lin(x, u):
        f_x, f_u = fd_jacobian_xu(dx_template, mx, set_control_fn, x, u)
        def c_fn_xy(xx, uu):
            return cost_prestate(dx_template, mx, running_cost_fn, xx, uu)
        c_x_ = jax.grad(lambda xx: c_fn_xy(xx, u))(x)
        c_u_ = jax.grad(lambda uu: c_fn_xy(x, uu))(u)
        c_xx_ = jax.jacrev(lambda xx: jax.grad(c_fn_xy, argnums=0)(xx, u))(x)
        c_uu_ = jax.jacrev(lambda uu: jax.grad(c_fn_xy, argnums=1)(x, uu))(u)

        def cross_fn_x(xx):
            return jax.grad(c_fn_xy, argnums=1)(xx, u)
        c_ux_ = jax.jacrev(cross_fn_x)(x)  # shape (nu, nx)
        return f_x, f_u, c_x_, c_u_, c_xx_, c_ux_, c_uu_

    out = jax.vmap(single_lin)(X[:-1], U)
    f_x_all, f_u_all, c_x_all, c_u_all, c_xx_all, c_ux_all, c_uu_all = out
    return f_x_all, f_u_all, c_x_all, c_u_all, c_xx_all, c_ux_all, c_uu_all


def backward_pass_fd_single(
    f_x, f_u, c_x, c_u, c_xx, c_ux, c_uu,
    x_final,
    terminal_cost_fn,
    dx_template,
    mx,
    reg
):
    nq = mx.nq
    def t_cost_fn(xx):
        dx_local = dx_template.replace(
            qpos=dx_template.qpos.at[:].set(xx[:nq]),
            qvel=dx_template.qvel.at[:].set(xx[nq:])
        )
        return terminal_cost_fn(dx_local)

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
        fx_t, fu_t, cx_t, cu_t, cxx_t, cux_t, cuu_t = inp

        Q_x = cx_t + fx_t.T @ V_x
        Q_u = cu_t + fu_t.T @ V_x
        Q_xx = cxx_t + fx_t.T @ V_xx @ fx_t
        Q_ux = cux_t + fu_t.T @ V_xx @ fx_t
        Q_uu = cuu_t + fu_t.T @ V_xx @ fu_t
        Q_uu_reg = Q_uu + reg*jnp.eye(Q_uu.shape[0])

        K_t = -jnp.linalg.solve(Q_uu_reg, Q_ux)
        k_t = -jnp.linalg.solve(Q_uu_reg, Q_u)

        V_x_new = Q_x + K_t.T @ Q_uu @ k_t + K_t.T @ Q_u + Q_ux.T @ k_t
        V_xx_new = (Q_xx
                    + K_t.T @ Q_uu @ K_t
                    + K_t.T @ Q_ux
                    + Q_ux.T @ K_t)
        return (V_x_new, V_xx_new), (K_t, k_t)

    (V_x_final, V_xx_final), (K_rev, k_rev) = lax.scan(
        bp_step,
        (V_x_init, V_xx_init),
        (f_x_rev, f_u_rev, c_x_rev, c_u_rev, c_xx_rev, c_ux_rev, c_uu_rev)
    )
    K = jnp.flip(K_rev, axis=0)
    k = jnp.flip(k_rev, axis=0)
    return K, k


def forward_pass_fd_single(
    dx_template, mx,
    set_control_fn,
    X_nom,
    U_nom,
    K, k,
    alpha
):
    def scan_fn(x, inp):
        K_t, k_t, U_nom_t, X_nom_t = inp
        u_new = U_nom_t + alpha*(k_t + K_t @ (x - X_nom_t))
        x_next = fd_next_state(dx_template, mx, set_control_fn, x, u_new)
        return x_next, u_new

    inps = (K, k, U_nom, X_nom[:-1])
    x0 = X_nom[0]
    _, U_new = lax.scan(scan_fn, x0, inps)
    return U_new

# ----------------------------------------------------------------
#  Part B: One iteration for SINGLE shapes, JIT-compiled
# ----------------------------------------------------------------

@eqx.filter_jit
def one_ilqr_iteration_single(
    dx_template,
    mx,
    set_control_fn,
    running_cost_fn,
    terminal_cost_fn,
    alpha,
    reg,
    qpos_init,  # shape (n_q,)
    qvel_init,  # shape (n_v,)
    U,          # shape (N, nu)
):
    # 1) Forward simulate
    X_nom, C_nom = simulate_fd_trajectory_single(
        dx_template, mx,
        qpos_init, qvel_init,
        set_control_fn, running_cost_fn, terminal_cost_fn,
        U
    )
    total_cost = jnp.sum(C_nom)  # scalar

    # 2) Linearize
    f_x, f_u, c_x, c_u, c_xx, c_ux, c_uu = linearize_fd_single(
        dx_template, mx, set_control_fn, running_cost_fn,
        X_nom, U
    )
    # 3) Backward pass
    K, k = backward_pass_fd_single(
        f_x, f_u, c_x, c_u,
        c_xx, c_ux, c_uu,
        X_nom[-1],
        terminal_cost_fn,
        dx_template,
        mx,
        reg
    )
    # 4) Forward pass => new controls
    U_new = forward_pass_fd_single(
        dx_template, mx,
        set_control_fn,
        X_nom, U,
        K, k,
        alpha
    )
    return U_new, total_cost


# ----------------------------------------------------------------
#  Part C: The ILQR class with a python loop over iterations
#           and separate single/batch solutions
# ----------------------------------------------------------------

class ILQR(eqx.Module):
    dx_template: eqx.static_field()
    mx: eqx.static_field()
    set_control_fn: eqx.static_field()
    running_cost_fn: eqx.static_field()
    terminal_cost_fn: eqx.static_field()
    alpha: float = 0.05
    reg: float = 1e-6

    def _solve_single(
        self,
        qpos_init: jnp.ndarray,  # (n_q,)
        qvel_init: jnp.ndarray,  # (n_v,)
        U0: jnp.ndarray,         # (N, nu)
        tol=1e-6,
        max_iter=50
    ):
        """
        A python loop. Each iteration calls one_ilqr_iteration_single (which is JIT).
        We can see cost each iteration.
        """
        U = U0
        prev_cost = jnp.inf

        for i in range(max_iter):
            U_new, cost_scalar = one_ilqr_iteration_single(
                self.dx_template,
                self.mx,
                self.set_control_fn,
                self.running_cost_fn,
                self.terminal_cost_fn,
                self.alpha,
                self.reg,
                qpos_init,
                qvel_init,
                U
            )
            improvement = prev_cost - cost_scalar
            print(f"Iteration {i}, cost={cost_scalar}")

            if improvement < 0:
                pass
            if jnp.abs(improvement) < tol:
                print(f"Converged at iteration {i}, cost={cost_scalar}")
                return U_new, cost_scalar

            U = U_new
            prev_cost = cost_scalar

        return U, cost_scalar

    def _solve_batch(
        self,
        qpos_init_b: jnp.ndarray,  # (B,n_q)
        qvel_init_b: jnp.ndarray,  # (B,n_v)
        U0_b: jnp.ndarray,         # (B,N,nu)
        tol=1e-6,
        max_iter=50
    ):
        """
        We do a python loop, but each iteration we vmap 'one_ilqr_iteration_single'
        so each problem is updated in parallel.
        We'll print the mean cost across the batch.
        This ensures each single iLQR sees shapes (N, nu) for U, etc.
        """
        U_b = U0_b
        prev_mean = jnp.inf

        def do_one(qpos_i, qvel_i, U_i):
            return one_ilqr_iteration_single(
                self.dx_template,
                self.mx,
                self.set_control_fn,
                self.running_cost_fn,
                self.terminal_cost_fn,
                self.alpha,
                self.reg,
                qpos_i,  # shape (n_q,)
                qvel_i,  # shape (n_v,)
                U_i      # shape (N, nu)
            )

        for i in range(max_iter):
            # vmap over B
            U_new_b, cost_b = jax.vmap(do_one)(qpos_init_b, qvel_init_b, U_b)
            # cost_b shape: (B,)
            cost_mean = jnp.mean(cost_b)
            improvement = prev_mean - cost_mean
            print(f"[Batch] Iteration {i}, mean cost={cost_mean}")

            if improvement < 0:
                pass
            if jnp.abs(improvement) < tol:
                print(f"Converged at iteration {i}, cost={cost_mean}")
                return U_new_b, cost_b

            U_b = U_new_b
            prev_mean = cost_mean

        return U_b, cost_b

    def solve(
        self,
        qpos_init: jnp.ndarray,
        qvel_init: jnp.ndarray,
        U0: jnp.ndarray,
        tol=1e-6,
        max_iter=50
    ):
        """
        Decide single vs. batch from shape.
        If qpos_init.shape == (n_q,), => single
        If qpos_init.shape == (B,n_q), => batch
        """
        if qpos_init.ndim == 1:
            return self._solve_single(qpos_init, qvel_init, U0, tol, max_iter)
        elif qpos_init.ndim == 2:
            return self._solve_batch(qpos_init, qvel_init, U0, tol, max_iter)
        else:
            raise ValueError("qpos_init must be shape (n_q,) or (B, n_q).")
