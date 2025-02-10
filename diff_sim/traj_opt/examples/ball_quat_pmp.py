import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from diff_sim.traj_opt.pmp_fd_indexes import build_fd_cache, make_loss_fn, PMP
from mujoco.mjx._src.math import quat_to_mat, axis_angle_to_quat
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')

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
        target_fields={"qpos", "qvel"},
        ctrl_dim=1,
        eps=1e-6
    )


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
        quat_ref =   axis_angle_to_quat(jnp.array([0.,1.,0.]), jnp.array([2.35]))
        # Chordal distance :
        # it complies with the four metric requirements while being more numerically stable
        # and simpler than the geodesic distance
        # https://arxiv.org/pdf/2401.05396
        costR = jnp.sum((quat_to_mat(dx.qpos[0:4])  - quat_to_mat(quat_ref))**2)

        # Geodesic distance : Log of the diff in rotation matrix, then skew-symmetric extraction, then norm.
        # It defines a smooth metric since both the logarithmic map and the Euclidean norm are smooth.
        # However, it brings more computational expense and numerical instability
        # from the logarithm map for small rotations
        # Note : Arccos function can be used but will only compute the abs value of the norm, might be problematic to get the
        # sign or direction for the gradient.
        # costR = rotation_distance(quat_to_mat(dx.qpos[3:7]),quat_to_mat(quat_ref))
        # R0 = quat_to_mat(quaternion_multiply(quaternion_conjugate(dx.qpos[3:7]), quat_ref))
        # error = jnp.tanh((jnp.trace(R0) -1)/2)
        # error = jnp.sum(R0,axis = -1)
        # error = jnp.arccos((jnp.trace(R0) -1)/2)
        # error = (R0 - jnp.identity(3))**2
        # error = jnp.sum((R0)**2)

        # return 0.01*jnp.array([jnp.sum(quat_diff**2)]) + 0.00001*dx.qfrc_applied[5]**2
        return 0.001*costR + 0.000001*dx.qfrc_applied[2]**2 + 0.000001*dx.qvel[2]**2
        # return 0.00001 * dx.qfrc_applied[5] ** 2


    def running_cost(dx: mjx.Data):
        cost = running_cost0(dx)
        return cost

    def terminal_cost(dx: mjx.Data):
        return 10*running_cost(dx)

    def set_control(dx, u):
        dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[1].set(u[0]))
        return dx

    @jax.jit
    def compute_loss_grad(u):
        jac_fun = jax.jacrev(lambda x: loss(x)[0])
        ad_grad = jac_fun(u)
        return ad_grad

    idata = mujoco.MjData(model)
    # qx0, qz0, qx1, qz1 = -0., 0.25, -0.2,  0.1s7] = qx0, qz0, qx1
    # idata.qpos[0], idata.qpos[2]= qx0, qz0
    Nlength = 100

    u0 = jnp.ones((Nlength, 1)) * 0.001
    qpos = jnp.array(idata.qpos)
    loss = make_loss_fn(mx, qpos, set_control, running_cost, terminal_cost, fd_cache)
    l, x = loss(u0)
    print("Loss: ", l)

    pmp = PMP(loss=lambda x: loss(x)[0])
    optimal_U = pmp.solve(U0=u0, learning_rate=0.00009, max_iter=100)
    l, x = loss(optimal_U)

    from diff_sim.utils.mj_viewers import visualise_traj_generic
    visualise_traj_generic(jnp.expand_dims(x, axis=0), idata, model, sleep=0.1)