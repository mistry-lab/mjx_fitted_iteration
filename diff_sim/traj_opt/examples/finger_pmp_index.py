import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import mujoco
from mujoco import mjx
from diff_sim.traj_opt.pmp_fd_indexes import PMP, make_loss_fn, build_fd_cache
from diff_sim.utils.mj_viewers import visualise_traj_generic
from mujoco.mjx._src.math import quat_to_mat, axis_angle_to_quat, quat_to_axis_angle

def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("../../xmls/finger_mjx.xml")
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    dx = jax.tree.map(upscale, dx)
    d = mujoco.MjData(model)
    # qpos_init = jnp.array([-.8, 0, -.8])


    qpos_init = jnp.concatenate([jnp.array([-0.8, 0.0]), axis_angle_to_quat(jnp.array([0., 1., 0.]), jnp.array([0.8]))])
    d.qpos = np.array(qpos_init)
    Nsteps, nu = 150, 1
    #make random control sequence
    U0 = jnp.ones((Nsteps, nu)) * -.02
    def running_cost(dx):
        # pos_finger = dx.qpos[2]
        # jax.debug.print("angle {p}", p = pos_finger)
        # jax.debug.print(f"quat of angle")
        # quat_ref = axis_angle_to_quat(jnp.array([0., 1., 0.]), jnp.array([0.]))
        # costR = jnp.sum((quat_to_mat(dx.qpos[2:6]) - quat_to_mat(quat_ref)) ** 2)
        angle = quat_to_axis_angle(dx.qpos[2:6])[1]
        # jax.debug.print("angle {p}", p=angle)
        u = dx.ctrl
        return 0.0 * jnp.sum(u ** 2) + 0.001 * (angle ** 2)

    def terminal_cost(dx):
        # pos_finger = dx.qpos[2]
        angle = quat_to_axis_angle(dx.qpos[2:6])[1]
        return 4 * (angle ** 2)

    def set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    fd_cache = build_fd_cache(
        mx, dx, ('qpos', 'qvel'), 1
    )

    # 3) Build the loss function with the new step fn
    loss_fn = make_loss_fn(
        mx=mx,
        qpos_init=qpos_init,
        set_ctrl_fn=set_control,
        running_cost_fn=running_cost,
        terminal_cost_fn=terminal_cost,
        fd_cache=fd_cache,
    )

    l, x = loss_fn(U0)
    visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)
    print("Initial loss: ", l)

    # 4) Optimize
    pmp = PMP(loss=lambda u: loss_fn(u)[0])
    optimal_U = pmp.solve(U0=U0, learning_rate=0.03, max_iter=50)

    l, x = loss_fn(optimal_U)

    visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)
