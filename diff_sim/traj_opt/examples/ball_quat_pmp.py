import os
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from diff_sim.traj_opt.pmp_fd_indexes import build_fd_cache, make_loss_fn, make_step_fn
from diff_sim.utils.math_helper import sub_quat


if __name__ == "__main__":
    # Load mj and mjx model
    model = mujoco.MjModel.from_xml_path("../../xmls/two_body.xml")
    mx = mjx.put_model(model)
    dx_ref = mjx.make_data(mx)

    # fd cache needs to return indicies for free and ball joints separately if they exist fintie difference will then
    # perturb these indicies specifically in a separate function to perturb differently based on theirs lists. diff is
    # also done in different manner (same as previous) for free and ball joints

    # Build an FD cache once, as usual
    fd_cache = build_fd_cache(
        mx,
        dx_ref,
        target_fields={"qpos", "qvel", "ctrl", "sensordata", "qfrc_applied", "qacc"},
        ctrl_dim=1,
        eps=1e-6
    )
    # break out of the main
    from IPython import embed
    embed()
    print(f"quat index: {fd_cache.quat_idx}")

    def running_cost(dx: mjx.Data):
        # return sub_quat()
        quat_ref = jnp.array([1., 0., 0., 0.])
        quat_diff = sub_quat(quat_ref, dx.qpos[3:7])

        # return 0.01*jnp.array([jnp.sum(quat_diff**2)]) + 0.00001*dx.qfrc_applied[5]**2
        return 0.00001 * dx.qfrc_applied[5] ** 2

    def terminal_cost(dx: mjx.Data):
        return running_cost(dx)

    def set_control(dx, u):
        dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[5].set(u[0]))
        return dx

    @jax.jit
    def compute_loss_grad(u):
        jac_fun = jax.jacrev(lambda x: loss(x)[0])
        ad_grad = jac_fun(u)
        return ad_grad

    idata = mujoco.MjData(model)
    qx0, qz0, qx1 = -0., 0.25, -0.2  # Inititial positions
    idata.qpos[0], idata.qpos[2], idata.qpos[7] = qx0, qz0, qx1
    Nlength = 40

    u0 = jnp.ones(Nlength) * 0.002
    u0 = jnp.expand_dims(u0, 1)
    qpos = jnp.array(idata.qpos)
    loss = make_loss_fn(mx, qpos, set_control, running_cost, terminal_cost, fd_cache)
    l, x = loss(u0)
    dldu = compute_loss_grad(u0)

    print("Loss: ", l)
    print("DlDu: ", dldu)

    from diff_sim.utils.mj import visualise_traj_generic
    visualise_traj_generic(jnp.expand_dims(x, axis=0), idata, model, sleep=0.1)
