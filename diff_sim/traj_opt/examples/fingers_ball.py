import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from diff_sim.utils.mj_viewers import visualise_traj_generic
from diff_sim.traj_opt.pmp_fd_indexes import build_fd_cache, make_loss_fn, make_step_fn, PMP
from diff_sim.utils.math_helper import sub_quat,quaternion_conjugate, quaternion_multiply
from mujoco.mjx._src.math import quat_to_mat, axis_angle_to_quat, quat_to_axis_angle
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')

if __name__ == "__main__":
    # Load mj and mjx model
    model = mujoco.MjModel.from_xml_path("../../xmls/fingers_ball.xml")
    mx = mjx.put_model(model)
    dx_ref = mjx.make_data(mx)

    # fd cache needs to return indicies for free and ball joints separately if they exist fintie difference will then
    # perturb these indicies specifically in a separate function to perturb differently based on theirs lists. diff is
    # also done in different manner (same as previous) for free and ball joints

    # Build an FD cache once, as usual
    fd_cache = build_fd_cache(
        mx,
        dx_ref,
        target_fields={"qpos", "qvel"},
        ctrl_dim=mx.nu,
        eps=1e-6
    )

    # from IPython import embed
    # embed()

    def skew_to_vector(skew_matrix):
        """Convert a 3x3 skew-symmetric matrix to a 3D vector (vee operator)."""
        return jnp.array([skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]])

    def matrix_log(R):
        def compute_log(R):
            # Compute the angle of rotation
            theta = jnp.arccos((jnp.trace(R) - 1) / 2)
            return (theta / (2 * jnp.sin(theta))) * (R - R.T)  # Normal log formula otherwise

        # Handle the special case where theta is very small (close to zero)
        # Instead of a conditional, use jnp.where to return a zero matrix when theta is very small
        return jnp.where(
            jnp.isclose((jnp.trace(R) - 1) / 2, 1.),
            jnp.zeros((3, 3)),  # Return zero matrix if theta is close to 0
            compute_log(R)
        )

    def rotation_distance(RA, RB):
        """Compute the geodesic distance between two SO(3) rotation matrices."""
        relative_rotation = jnp.dot(RA.T, RB)  # Compute RA^T * RB
        log_relative = matrix_log(relative_rotation)  # Compute matrix logarithm
        omega = skew_to_vector(log_relative)    # Extract the rotation vector (vee operator)
        return jnp.linalg.norm(omega)           # Compute the rotation distance

    def running_cost0(dx: mjx.Data):
        quat_ref =   axis_angle_to_quat(jnp.array([0.,0.,1.]), jnp.array([2.35]))
        # Chordal distance :
        # it complies with the four metric requirements while being more numerically stable
        # and simpler than the geodesic distance
        # https://arxiv.org/pdf/2401.05396
        costR = jnp.sum((quat_to_mat(dx.qpos[4:8])  - quat_to_mat(quat_ref))**2)
        return 0.01*costR + 0.0001*jnp.sum(dx.ctrl**2)

    def running_cost(dx: mjx.Data):
        cost = running_cost0(dx)
        return cost

    def terminal_cost(dx: mjx.Data):
        return 10*running_cost(dx)

    def set_control(dx, u):
        dx = dx.replace(ctrl=dx.ctrl.at[:].set(u))
        return dx

    @jax.jit
    def compute_loss_grad(u):
        jac_fun = jax.jacrev(lambda x: loss(x)[0])
        ad_grad = jac_fun(u)
        return ad_grad

    idata = mujoco.MjData(model)
    q0, q1, q2, q3 = -0.4, 0.44, 0.44, -0.4
    idata.qpos[0], idata.qpos[1], idata.qpos[2], idata.qpos[3] = q0, q1, q2, q3
    Nlength = 200

    u0 = 0.5 * jax.random.normal(jax.random.PRNGKey(0), (Nlength, 4))
    u0 = u0.at[:, 0].set(0.1)
    u0 = u0.at[:, 2].set(-0.1)
    u0 = u0.at[:, 3].set(0.)
    u0 = u0.at[:, 1].set(0.)
    qpos = jnp.array(idata.qpos)
    loss = make_loss_fn(mx, qpos, set_control, running_cost, terminal_cost, fd_cache)

    l, x = loss(u0)
    visualise_traj_generic(jnp.expand_dims(x, axis=0), idata, model, sleep=0.1)

    pmp = PMP(loss=lambda x: loss(x)[0])
    optimal_U = pmp.solve(U0=u0, learning_rate=0.001, max_iter=100)

    l, x = loss(optimal_U)
    visualise_traj_generic(jnp.expand_dims(x, axis=0), idata, model, sleep=0.1)