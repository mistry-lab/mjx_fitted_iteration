import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")

import mujoco
from mujoco import mjx

from diff_sim.traj_opt.ilqr_fd import ILQR  # or wherever you place the code above

def upscale(x):
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return x.astype(jnp.int64)
        elif x.dtype == jnp.float32:
            return x.astype(jnp.float64)
    return x

if __name__ == "__main__":
    # 1) Load MuJoCo model
    model = mujoco.MjModel.from_xml_path("../../xmls/finger_mjx.xml")
    mx = mjx.put_model(model)
    dx_template = mjx.make_data(mx)
    dx_template = jax.tree_map(upscale, dx_template)

    # 2) Define cost & control setter
    def running_cost_fn(dx: mjx.Data) -> float:
        pos_finger = dx.qpos[2]
        ctrl = dx.ctrl
        return 0.002 * jnp.sum(ctrl**2) + 0.001 * (pos_finger**2)

    def terminal_cost_fn(dx: mjx.Data) -> float:
        pos_finger = dx.qpos[2]
        return 4.0 * (pos_finger**2)

    def set_control_fn(dx: mjx.Data, u: jnp.ndarray) -> mjx.Data:
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    # 3) Build ILQR
    solver = ILQR(
        dx_template=dx_template,
        mx=mx,
        set_control_fn=set_control_fn,
        running_cost_fn=running_cost_fn,
        terminal_cost_fn=terminal_cost_fn,
        alpha=0.05,
        reg=1e-6
    )

    # 4) Single example
    nq, nv = mx.nq, mx.nv
    Nsteps, nu = 300, 2

    qpos_init_single = jnp.array([-0.5, 0.0, -1.0])
    qvel_init_single = jnp.zeros_like(qpos_init_single)
    U0_single = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu))

    U_opt_single, cost_single = solver.solve(
        qpos_init_single,
        qvel_init_single,
        U0_single,
        tol=1e-5,
        max_iter=50
    )
    print("Single final cost:", cost_single)

    # 5) Batch example of size B=3
    qpos_init_batch = jnp.array([
        [-0.5,  0.0, -1.0],
        [-0.8,  0.2, -0.2],
        [ 0.2, -0.1,  0.6]
    ])
    qvel_init_batch = jnp.zeros_like(qpos_init_batch)  # shape (3,3)
    U0_batch = 0.1 * jax.random.normal(jax.random.PRNGKey(1), (3, Nsteps, nu))

    U_opt_batch, cost_batch = solver.solve(
        qpos_init_batch,
        qvel_init_batch,
        U0_batch,
        tol=1e-5,
        max_iter=50
    )
    print("Batch final cost:", cost_batch)  # shape (3,)

    # Each iteration in the batch loop prints:
    # [Batch] Iteration i, mean cost=0.1234
